"""Monte Carlo batch runner — sample bench offsets and run scenarios."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from satellite.config import (
    BenchOffsetConfig,
    ErrorDistributionConfig,
    GaussianErrorConfig,
    MonteCarloConfig,
    ScenarioInstance,
    UniformErrorConfig,
    build_scenario_config,
    load_monte_carlo_config,
    load_simulation_config,
)
from satellite.scenario import ScenarioResult, run_scenario


@dataclass(frozen=True)
class MonteCarloRunResult:
    run_index: int
    s1_theta: float
    s1_phi: float
    s2_theta: float
    s2_phi: float
    result: ScenarioResult


@dataclass(frozen=True)
class MonteCarloSummary:
    runs: int
    planned_runs: int
    interrupted: bool
    successes: int
    success_rate: float
    by_strategy: dict[str, int]
    run_results: tuple[MonteCarloRunResult, ...]


def _sample_component(rng: np.random.Generator, error: ErrorDistributionConfig) -> tuple[float, float]:
    if isinstance(error, UniformErrorConfig):
        theta = float(rng.uniform(error.theta_min, error.theta_max))
        phi = float(rng.uniform(error.phi_min, error.phi_max))
        return theta, phi
    if isinstance(error, GaussianErrorConfig):
        theta = float(rng.normal(error.theta_mean, error.theta_std))
        phi = float(rng.normal(error.phi_mean, error.phi_std))
        return theta, phi
    raise TypeError(f"Unsupported error config: {type(error)!r}")


def sample_offsets(
    error: ErrorDistributionConfig,
    rng: np.random.Generator,
) -> tuple[BenchOffsetConfig, BenchOffsetConfig]:
    s1_theta, s1_phi = _sample_component(rng, error)
    s2_theta, s2_phi = _sample_component(rng, error)
    return (
        BenchOffsetConfig(s1_theta, s1_phi),
        BenchOffsetConfig(s2_theta, s2_phi),
    )


def _format_sigfig(value: float, sigfigs: int = 3) -> str:
    return f"{value:.{sigfigs}g}"


def format_monte_carlo_run_start(
    run_number: int,
    total_runs: int,
    s1_theta: float,
    s1_phi: float,
    s2_theta: float,
    s2_phi: float,
) -> str:
    return (
        f"Running scenario {run_number}/{total_runs}: "
        f"S1_θ_off={_format_sigfig(s1_theta * 1e3)} mrad  "
        f"S1_φ_off={_format_sigfig(s1_phi * 1e3)} mrad  "
        f"S2_θ_off={_format_sigfig(s2_theta * 1e3)} mrad  "
        f"S2_φ_off={_format_sigfig(s2_phi * 1e3)} mrad"
    )


def format_monte_carlo_run_complete(
    run: MonteCarloRunResult,
    *,
    elapsed_ms: float,
) -> str:
    prefix = f"Completed in {elapsed_ms:.1f}ms:"
    result = run.result
    if result.success:
        strat = result.strategy_name or "unknown"
        q = result.hit_at_q
        q_str = f"{q:.3g}" if q is not None else "?"
        return f"{prefix} Success with {strat} at q={q_str}"
    total_q = result.schedule.total_duration
    tried = ", ".join(result.config.strategy.chain)
    return f"{prefix} Failed after q={total_q:.3g} timeout (tried {tried})"


def run_monte_carlo_single(
    mc: MonteCarloConfig,
    sim,
    rng: np.random.Generator,
    run_index: int,
    *,
    report: bool = False,
    total_runs: int | None = None,
) -> MonteCarloRunResult:
    s1_off, s2_off = sample_offsets(mc.error, rng)
    if report:
        if total_runs is None:
            raise ValueError("total_runs is required when report=True")
        print(
            format_monte_carlo_run_start(
                run_index + 1,
                total_runs,
                s1_off.bench_theta_offset,
                s1_off.bench_phi_offset,
                s2_off.bench_theta_offset,
                s2_off.bench_phi_offset,
            ),
            flush=True,
        )
    instance = ScenarioInstance(
        name=f"mc_run_{run_index}",
        distance=None,
        s1=s1_off,
        s2=s2_off,
    )
    config = build_scenario_config(sim, instance, strategy=mc.strategy)
    t0 = time.perf_counter()
    result = run_scenario(config)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    run_result = MonteCarloRunResult(
        run_index=run_index,
        s1_theta=s1_off.bench_theta_offset,
        s1_phi=s1_off.bench_phi_offset,
        s2_theta=s2_off.bench_theta_offset,
        s2_phi=s2_off.bench_phi_offset,
        result=result,
    )
    if report:
        print(
            format_monte_carlo_run_complete(run_result, elapsed_ms=elapsed_ms),
            flush=True,
        )
    return run_result


def _build_monte_carlo_summary(
    mc: MonteCarloConfig,
    run_results: list[MonteCarloRunResult],
    *,
    interrupted: bool,
) -> MonteCarloSummary:
    by_strategy: dict[str, int] = {}
    for run_result in run_results:
        result = run_result.result
        if result.success and result.strategy_name:
            by_strategy[result.strategy_name] = (
                by_strategy.get(result.strategy_name, 0) + 1
            )
    completed = len(run_results)
    successes = sum(1 for r in run_results if r.result.success)
    return MonteCarloSummary(
        runs=completed,
        planned_runs=mc.runs,
        interrupted=interrupted,
        successes=successes,
        success_rate=successes / completed if completed else 0.0,
        by_strategy=by_strategy,
        run_results=tuple(run_results),
    )


def run_monte_carlo(mc: MonteCarloConfig) -> MonteCarloSummary:
    sim = load_simulation_config(mc.simulation_path)
    rng = np.random.default_rng(mc.seed)
    run_results: list[MonteCarloRunResult] = []
    interrupted = False

    try:
        for run_index in range(mc.runs):
            run_result = run_monte_carlo_single(
                mc, sim, rng, run_index, report=True, total_runs=mc.runs
            )
            run_results.append(run_result)
    except KeyboardInterrupt:
        interrupted = True

    return _build_monte_carlo_summary(mc, run_results, interrupted=interrupted)


def run_monte_carlo_from_path(path: str) -> MonteCarloSummary:
    mc = load_monte_carlo_config(path)
    return run_monte_carlo(mc)


def format_monte_carlo_summary(summary: MonteCarloSummary) -> str:
    lines: list[str] = []
    if summary.interrupted:
        lines.append(
            f"Interrupted after {summary.runs}/{summary.planned_runs} runs."
        )
    lines.append(
        f"Monte Carlo: {summary.successes}/{summary.runs} succeeded "
        f"({100.0 * summary.success_rate:.1f}%)",
    )
    if summary.by_strategy:
        parts = ", ".join(
            f"{name}: {count}" for name, count in sorted(summary.by_strategy.items())
        )
        lines.append(f"Winning strategies: {parts}")
    return "\n".join(lines)
