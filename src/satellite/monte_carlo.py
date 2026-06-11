"""Monte Carlo batch runner — sample bench offsets and run scenarios."""

from __future__ import annotations

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


def run_monte_carlo(mc: MonteCarloConfig) -> MonteCarloSummary:
    sim = load_simulation_config(mc.simulation_path)
    rng = np.random.default_rng(mc.seed)
    run_results: list[MonteCarloRunResult] = []
    by_strategy: dict[str, int] = {}

    for run_index in range(mc.runs):
        s1_off, s2_off = sample_offsets(mc.error, rng)
        instance = ScenarioInstance(
            name=f"mc_run_{run_index}",
            distance=None,
            s1=s1_off,
            s2=s2_off,
        )
        config = build_scenario_config(sim, instance, strategy=mc.strategy)
        result = run_scenario(config)
        run_results.append(
            MonteCarloRunResult(
                run_index=run_index,
                s1_theta=s1_off.bench_theta_offset,
                s1_phi=s1_off.bench_phi_offset,
                s2_theta=s2_off.bench_theta_offset,
                s2_phi=s2_off.bench_phi_offset,
                result=result,
            )
        )
        if result.success and result.strategy_name:
            by_strategy[result.strategy_name] = by_strategy.get(result.strategy_name, 0) + 1

    successes = sum(1 for r in run_results if r.result.success)
    return MonteCarloSummary(
        runs=mc.runs,
        successes=successes,
        success_rate=successes / mc.runs if mc.runs else 0.0,
        by_strategy=by_strategy,
        run_results=tuple(run_results),
    )


def run_monte_carlo_from_path(path: str) -> MonteCarloSummary:
    mc = load_monte_carlo_config(path)
    return run_monte_carlo(mc)


def format_monte_carlo_summary(summary: MonteCarloSummary) -> str:
    lines = [
        f"Monte Carlo: {summary.successes}/{summary.runs} succeeded "
        f"({100.0 * summary.success_rate:.1f}%)",
    ]
    if summary.by_strategy:
        parts = ", ".join(
            f"{name}: {count}" for name, count in sorted(summary.by_strategy.items())
        )
        lines.append(f"Winning strategies: {parts}")
    return "\n".join(lines)
