"""Monte Carlo batch runner — sample bench offsets and run scenarios."""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import sys
import time
from dataclasses import dataclass

import numpy as np

from satellite.config import (
    BenchOffsetConfig,
    ErrorDistributionConfig,
    GaussianErrorConfig,
    MonteCarloChainConfig,
    MonteCarloConfig,
    ScenarioInstance,
    StrategyConfig,
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
    computation_time_ms: float


@dataclass(frozen=True)
class MonteCarloSummary:
    runs: int
    planned_runs: int
    interrupted: bool
    successes: int
    success_rate: float
    by_strategy: dict[str, int]
    run_results: tuple[MonteCarloRunResult, ...]
    mean_t: float | None
    median_t: float | None
    mean_computation_ms: float
    median_computation_ms: float
    total_computation_ms: float


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
        t = result.hit_at_t
        t_str = f"{t:.3g}" if t is not None else "?"
        return f"{prefix} Success with {strat} at t={t_str}"
    total_t = result.schedule.total_duration
    tried = ", ".join(result.config.strategy.chain)
    return f"{prefix} Failed after t={total_t:.3g} timeout (tried {tried})"


def run_monte_carlo_single(
    mc: MonteCarloConfig,
    sim,
    seed: int,
    run_index: int,
    *,
    report: bool = False,
    total_runs: int | None = None,
    strategy: StrategyConfig | None = None,
) -> MonteCarloRunResult:
    rng = np.random.default_rng(seed)
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
    config = build_scenario_config(sim, instance, strategy=strategy or mc.strategy)
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
        computation_time_ms=elapsed_ms,
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
    success_ts = []
    comp_times = []
    for run_result in run_results:
        result = run_result.result
        comp_times.append(run_result.computation_time_ms)
        if result.success:
            if result.strategy_name:
                by_strategy[result.strategy_name] = (
                    by_strategy.get(result.strategy_name, 0) + 1
                )
            if result.hit_at_t is not None:
                success_ts.append(result.hit_at_t)

    completed = len(run_results)
    successes = len(success_ts)
    
    mean_t = float(np.mean(success_ts)) if success_ts else None
    median_t = float(np.median(success_ts)) if success_ts else None
    
    mean_comp = float(np.mean(comp_times)) if comp_times else 0.0
    median_comp = float(np.median(comp_times)) if comp_times else 0.0
    total_comp = float(np.sum(comp_times)) if comp_times else 0.0

    return MonteCarloSummary(
        runs=completed,
        planned_runs=mc.runs,
        interrupted=interrupted,
        successes=successes,
        success_rate=successes / completed if completed else 0.0,
        by_strategy=by_strategy,
        run_results=tuple(run_results),
        mean_t=mean_t,
        median_t=median_t,
        mean_computation_ms=mean_comp,
        median_computation_ms=median_comp,
        total_computation_ms=total_comp,
    )


def _print_progress_bar(
    completed: int,
    total: int,
    successes: int,
    *,
    width: int = 40,
    finished: bool = False,
) -> None:
    fraction = completed / total if total > 0 else 0.0
    filled = int(width * fraction)
    bar = "#" * filled + " " * (width - filled)
    percent = 100.0 * fraction
    failures = completed - successes
    line = (
        f"[{bar}] {completed}/{total} ({percent:.1f}%) - "
        f"Success: {successes} | Failed: {failures}"
    )
    if finished:
        sys.stdout.write(f"\r{line}\n")
    else:
        sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


def run_monte_carlo(
    mc: MonteCarloConfig,
    max_workers: int | None = None,
) -> MonteCarloSummary:
    from satellite.cuda_monte_carlo import is_strategy_chain_supported_on_gpu, run_monte_carlo_cuda
    if is_strategy_chain_supported_on_gpu(mc):
        print("Running Monte Carlo simulation on GPU (RTX 3050)...", flush=True)
        return run_monte_carlo_cuda(mc)

    sim = load_simulation_config(mc.simulation_path)

    tasks = []
    global_run_idx = 0
    for chain_cfg in mc.chains:
        chain_rng = np.random.default_rng(mc.seed)
        seeds = chain_rng.integers(0, 2**32 - 1, size=chain_cfg.runs).tolist()
        
        chain_strategy = StrategyConfig(
            k=mc.strategy.k,
            chain=chain_cfg.chain,
            params=mc.strategy.params,
        )
        
        for run_in_chain_idx in range(chain_cfg.runs):
            tasks.append((seeds[run_in_chain_idx], global_run_idx, chain_strategy))
            global_run_idx += 1

    total_runs = len(tasks)
    run_results: list[MonteCarloRunResult] = []
    interrupted = False
    successes = 0

    # Number of workers (leave 1 core free for system if many cores)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    try:
        if max_workers > 1 and total_runs > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_index = {
                    executor.submit(
                        run_monte_carlo_single,
                        mc,
                        sim,
                        seed,
                        global_idx,
                        report=False,
                        strategy=strat,
                    ): global_idx
                    for seed, global_idx, strat in tasks
                }

                _print_progress_bar(0, total_runs, 0)

                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        res = future.result()
                        run_results.append(res)
                        if res.result.success:
                            successes += 1
                        _print_progress_bar(len(run_results), total_runs, successes)
                    except Exception as e:
                        print(f"\nRun failed with error: {e}", file=sys.stderr)
        else:
            # Serial execution (max_workers=1)
            _print_progress_bar(0, total_runs, 0)
            for seed, global_idx, strat in tasks:
                res = run_monte_carlo_single(
                    mc, sim, seed, global_idx, report=False, strategy=strat
                )
                run_results.append(res)
                if res.result.success:
                    successes += 1
                _print_progress_bar(len(run_results), total_runs, successes)

        _print_progress_bar(len(run_results), total_runs, successes, finished=True)

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Cleaning up workers...", file=sys.stderr)

    run_results.sort(key=lambda r: r.run_index)
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
    if summary.mean_t is not None:
        lines.append(
            f"Success sim-t: mean={summary.mean_t:.3f}, median={summary.median_t:.3f}"
        )
    lines.append(
        f"Computation: mean={summary.mean_computation_ms:.1f}ms, "
        f"median={summary.median_computation_ms:.1f}ms, "
        f"total={summary.total_computation_ms / 1000.0:.2f}s"
    )
    if summary.by_strategy:
        parts = ", ".join(
            f"{name}: {count}" for name, count in sorted(summary.by_strategy.items())
        )
        lines.append(f"Winning strategies: {parts}")
    return "\n".join(lines)
