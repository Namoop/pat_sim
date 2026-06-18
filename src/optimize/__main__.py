#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pointing Acquisition Strategy Parameter Optimizer.
Supports Grid Search, Random Search, and Optuna.

Physical speed validation
--------------------------
Every candidate parameter set is checked analytically against the hardware
``max_beam_speed`` limit *before* a simulation is run.  Invalid candidates are
counted as rejected trials — the trial counter still advances so the total
number of requested trials is always honoured — but they are not simulated and
do not consume process-pool resources.

Speed formulae used
~~~~~~~~~~~~~~~~~~~
* **spiral** (dual_spiral, concentric_shells):
    peak ≈ speed × √(w² + (k·sin(R))²)   where R = max_search_radius

* **lissajous_scan**:
    peak ≈ A × √(wx² + wy²)               where A = max_search_radius / √2

* **rosette_scan**:
    peak ≈ A × (|w1| + |w2|)

* **dual_raster** (SerpentineRaster):
    peak = 2·R·steps·speed_a / 10         (chord scanning at max-chord row)

* **random_curve / center_rebias**:
    velocity_a  is already a direct angular speed in rad/s.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing
import signal
import sys
import time
import numpy as np

# Set by the first Ctrl+C; a second Ctrl+C restores default handling and exits.
_interrupt_requested = False


def _handle_sigint(signum, frame) -> None:
    global _interrupt_requested
    if not _interrupt_requested:
        _interrupt_requested = True
        print(
            "\nInterrupt received — stopping after current work...",
            file=sys.stderr,
            flush=True,
        )
    else:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt


signal.signal(signal.SIGINT, _handle_sigint)


class OptimizationInterrupted(Exception):
    """User requested early termination (Ctrl+C)."""


def _check_interrupt() -> None:
    if _interrupt_requested:
        raise OptimizationInterrupted()


def _note_interrupt() -> None:
    if not _interrupt_requested:
        print(
            "\nInterrupted. Cleaning up workers...",
            file=sys.stderr,
            flush=True,
        )


# SearchResult: best_params, best_cost, interrupted, completed_trials
SearchResult = tuple[dict, float, bool, int]

from satellite.config import (
    load_simulation_config,
    load_monte_carlo_config,
    ScenarioInstance,
    build_scenario_config,
    StrategyConfig,
)
from satellite.monte_carlo import sample_offsets
from satellite.scenario import run_scenario

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def build_parameter_spaces(sim_cfg, mc_cfg) -> dict:
    """Derive per-strategy parameter bounds from the loaded simulation config.

    Upper bounds on speed-related parameters are set to 95% of the hardware
    max_beam_speed so the optimizer always has room to find valid candidates.
    All other (non-speed) parameters use generous fixed ranges.
    """
    R = sim_cfg.simulation.max_search_radius   # rad
    v_max = sim_cfg.satellite.max_beam_speed   # rad/s
    v_hi = v_max * 0.95                        # leave 5% headroom

    # --- random_curve / center_rebias ---
    # peak = velocity_a * velocity_ratio  (velocity_a already in rad/s)
    # With ratio_max = 3.0: velocity_a_max = v_hi / 3.0
    ratio_rc_max = 3.0
    vel_a_max = max(1e-4, v_hi / ratio_rc_max)

    # --- lissajous ---
    # peak ≈ A * sqrt(wx² + wy²),  A = R / sqrt(2)
    # worst case both axes equal: peak = A * wx * sqrt(2) => wx_max = v_hi / A
    A_lis = R / math.sqrt(2.0)
    wx_max = max(0.2, v_hi / max(A_lis, 1e-9))

    # --- rosette ---
    # peak ≈ A * (w1 + w2),  A = R
    # equal split: w_max = v_hi / (2 * R)
    w_ros_max = max(0.2, v_hi / max(2.0 * R, 1e-9))


    # --- dual_raster ---
    # peak = 2 * R * steps * speed / 10
    # With steps_max=100, ratio_max=1.5: speed_max = v_hi * 10 / (2*R*100*1.5)
    steps_max = 100
    ratio_raster_max = 1.5
    speed_raster_max = max(0.01, v_hi * 10.0 / max(2.0 * R * steps_max * ratio_raster_max, 1e-9))

    return {
        "random_curve": {
            "velocity_a":     ("float", 1e-4,       vel_a_max),
            "velocity_ratio": ("float", 1.0,        ratio_rc_max),
            "drift_sigma":    ("float", 0.001,      0.5),
            "max_turn_radius":("float", 0.1,        2.0),
        },
        "center_rebias": {
            "velocity_a":     ("float", 1e-4,       vel_a_max),
            "velocity_ratio": ("float", 1.0,        ratio_rc_max),
            "drift_sigma":    ("float", 0.001,      0.5),
            "max_turn_radius":("float", 0.1,        2.0),
            "bias_strength":  ("float", 0.0,        1.0),
        },
        "lissajous_scan": {
            "s1_wx": ("float", 0.1, wx_max),
            "s1_wy": ("float", 0.1, wx_max),
            "s2_wx": ("float", 0.1, wx_max),
            "s2_wy": ("float", 0.1, wx_max),
        },
        "rosette_scan": {
            "s1_w1": ("float", 0.1, w_ros_max),
            "s1_w2": ("float", 0.1, w_ros_max),
            "s2_w1": ("float", 0.1, w_ros_max),
            "s2_w2": ("float", 0.1, w_ros_max),
        },
        "dual_spiral": {
            "k_ratio":        ("float", 0.5, 2.0),
            "s2_hold_delay":  ("float", 0.0, 5.0),
            "phase_offset":   ("float", 0.0, 6.2831853),
        },
        "dual_raster": {
            "steps_a":     ("int",   5,     steps_max),
            "steps_b":     ("int",   5,     steps_max),
            "speed_a":     ("float", 0.001, speed_raster_max),
            "speed_ratio": ("float", 1.0,   ratio_raster_max),
        },
        "concentric_shells": {
            "num_shells":       ("int", 2, 5),
            "growth_exponent":  ("float", 0.5, 2.0),
            "s2_offset_shells": ("int", 0, 4),
        },
    }


# ---------------------------------------------------------------------------
# Analytic peak-speed estimators
# ---------------------------------------------------------------------------

def _spiral_peak_speed(speed: float, w: float, k: float, max_radius: float) -> float:
    """Archimedean spiral: worst-case angular speed.
    
    Since the spiral is parameterized to move at a constant linear speed, the peak
    angular speed is exactly equal to the constant physical speed: speed * c.
    """
    if w <= 0.0 or k <= 0.0 or max_radius <= 0.0:
        return speed
    x = (k * max_radius) / w
    sqrt_term = math.sqrt(1.0 + x * x)
    h_x = 0.5 * (x * sqrt_term + math.log(x + sqrt_term))
    c = h_x / x
    return speed * c


def peak_speed_for_strategy(
    strategy_name: str,
    params: dict,
    sim_cfg,
    mc_cfg,
) -> float:
    """Return an analytic upper-bound on the beam angular speed for *params*.

    The estimate is conservative (may slightly over-predict) which is the
    safe direction — it means we never let the optimizer propose parameters
    that could violate hardware limits.
    """
    sat = sim_cfg.satellite
    strat = mc_cfg.strategy
    max_radius = sim_cfg.simulation.max_search_radius
    k = strat.k
    w = strat.spiral_w(sat)

    if strategy_name in ("dual_spiral", "concentric_shells"):
        # Both satellites spiral at max_beam_speed
        return sat.max_beam_speed

    if strategy_name == "dual_raster":
        # SerpentineRaster: max scan speed at the widest chord (centre row)
        # duration = 10 / speed_a; peak = 2·R·steps / duration
        speed_a = params.get("speed_a", 1.0)
        ratio = params.get("speed_ratio", 1.41421356)
        speed_b = speed_a * ratio
        steps_a = params.get("steps_a", 20)
        steps_b = params.get("steps_b", 20)
        # duration = 10 / speed;  peak = 2 * R * steps / duration = 2*R*steps*speed/10
        peak_a = 2.0 * max_radius * steps_a * speed_a / 10.0
        peak_b = 2.0 * max_radius * steps_b * speed_b / 10.0
        return max(peak_a, peak_b)

    if strategy_name == "lissajous_scan":
        # u = A·sin(wx·t + delta),  v = A·sin(wy·t)
        # |du/dt|_max = A·wx,  |dv/dt|_max = A·wy
        # peak ≈ A·√(wx²+wy²)  (both reach max simultaneously at worst case)
        A = max_radius / math.sqrt(2.0)
        peak_s1 = A * math.sqrt(params.get("s1_wx", 1.0) ** 2 + params.get("s1_wy", 1.41421356) ** 2)
        peak_s2 = A * math.sqrt(params.get("s2_wx", 1.0) ** 2 + params.get("s2_wy", 1.41421356) ** 2)
        return max(peak_s1, peak_s2)

    if strategy_name == "rosette_scan":
        # r = A·cos(w2·t); u = r·cos(w1·t), v = r·sin(w1·t)
        # |dr/dt|_max = A·w2,  peak tangential ≈ A·w1
        # conservative bound: A·(w1 + w2)
        A = max_radius
        peak_s1 = A * (abs(params.get("s1_w1", 1.0)) + abs(params.get("s1_w2", 11.0)))
        peak_s2 = A * (abs(params.get("s2_w1", 1.0)) + abs(params.get("s2_w2", 1.0)))
        return max(peak_s1, peak_s2)

    if strategy_name in ("random_curve", "center_rebias"):
        # velocity_a is already an angular speed in rad/s
        velocity_a = params.get("velocity_a", 0.01)
        ratio = params.get("velocity_ratio", 1.41421356)
        return velocity_a * ratio  # faster satellite (S2)

    # Unknown strategy — no estimate available; assume valid
    return 0.0


def is_physically_valid(
    strategy_name: str,
    params: dict,
    sim_cfg,
    mc_cfg,
) -> bool:
    """Return True iff *params* do not analytically exceed max_beam_speed."""
    max_speed = sim_cfg.satellite.max_beam_speed
    if max_speed <= 0.0:
        return True  # no constraint configured
    peak = peak_speed_for_strategy(strategy_name, params, sim_cfg, mc_cfg)
    return peak <= max_speed


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

from optimize import evaluate_single_instance


def evaluate_candidate(
    params: dict,
    strategy_name: str,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    max_workers: int,
) -> tuple[float, float, float]:
    from satellite.strategy.base import CONFIG_PARSERS

    params = dict(params)
    if strategy_name == "lissajous_scan":
        params["s1_delta"] = 1.570796
        params["s2_delta"] = 1.570796

    parsed_params = params
    if strategy_name in CONFIG_PARSERS:
        parsed_params = CONFIG_PARSERS[strategy_name](params)

    strategy = StrategyConfig(
        k=mc_cfg.strategy.k,
        chain=(strategy_name,),
        params={strategy_name: parsed_params},
    )

    # If the strategy is GPU-compatible, perform evaluation in a single batch on the GPU
    import os
    from satellite.cuda_monte_carlo import is_strategy_chain_supported_on_gpu, run_monte_carlo_cuda
    from satellite.config import MonteCarloConfig

    mc = MonteCarloConfig(
        simulation_path=mc_cfg.simulation_path,
        seed=mc_cfg.seed,
        runs=len(fixed_offsets),
        chain=(strategy_name,),
        error=mc_cfg.error,
        strategy=strategy,
    )

    if os.environ.get("SATELLITE_NO_GPU") != "1" and is_strategy_chain_supported_on_gpu(mc):
        summary = run_monte_carlo_cuda(mc)
        success_rate = summary.success_rate
        timeout = sim_cfg.simulation.timeout
        mean_t = summary.mean_t if summary.mean_t is not None else timeout
        penalty = timeout * 2.0
        cost = ((1.0 - success_rate) * penalty) + mean_t
        return cost, success_rate, mean_t

    # Fallback to CPU parallel execution
    tasks = [
        (sim_cfg, ScenarioInstance(name=f"eval_{idx}", s1=s1_off, s2=s2_off), strategy)
        for idx, (s1_off, s2_off) in enumerate(fixed_offsets)
    ]

    successes = 0
    hit_times: list[float] = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single_instance, *task) for task in tasks]
            for fut in concurrent.futures.as_completed(futures):
                _check_interrupt()
                try:
                    success, hit_at_t = fut.result()
                except concurrent.futures.CancelledError:
                    continue
                if success:
                    successes += 1
                    if hit_at_t is not None:
                        hit_times.append(hit_at_t)
    except (KeyboardInterrupt, OptimizationInterrupted):
        _note_interrupt()
        raise OptimizationInterrupted() from None

    success_rate = successes / len(fixed_offsets)
    timeout = sim_cfg.simulation.timeout
    mean_t = float(np.mean(hit_times)) if hit_times else timeout

    # Cost: prioritise lock rate first, then speed
    penalty = timeout * 2.0
    cost = ((1.0 - success_rate) * penalty) + mean_t
    return cost, success_rate, mean_t


# ---------------------------------------------------------------------------
# Candidate sampling helpers
# ---------------------------------------------------------------------------

def _sample_candidate(space: dict, rng: np.random.Generator) -> dict:
    candidate = {}
    for param, spec in space.items():
        ptype, start, end = spec
        if ptype == "float":
            candidate[param] = float(rng.uniform(start, end))
        elif ptype == "int":
            candidate[param] = int(rng.integers(start, end + 1))
    return candidate


def _sample_valid_candidate(
    space: dict,
    strategy_name: str,
    sim_cfg,
    mc_cfg,
    rng: np.random.Generator,
    *,
    max_attempts: int = 1000,
) -> dict | None:
    """Sample a candidate that passes the physics check.

    Returns None if no valid candidate is found within *max_attempts* draws,
    which would indicate the entire search space violates the speed limit.
    """
    for _ in range(max_attempts):
        c = _sample_candidate(space, rng)
        if is_physically_valid(strategy_name, c, sim_cfg, mc_cfg):
            return c
    return None


# ---------------------------------------------------------------------------
# Search methods
# ---------------------------------------------------------------------------

def run_random_search(
    strategy_name: str,
    space: dict,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    trials: int,
    max_workers: int,
    seed: int | None = None,
) -> SearchResult:
    """Basic Random Search optimization.

    Unphysical candidates are resampled in place — the trial counter only
    advances when a candidate actually runs.
    """
    best_params: dict = {}
    best_cost = float("inf")
    rng = np.random.default_rng(seed)

    max_speed = sim_cfg.satellite.max_beam_speed
    print(f"Starting Random Search ({trials} trials, max_beam_speed={max_speed:.4f} rad/s)...")

    resampled = 0
    interrupted = False
    completed_trials = 0
    try:
        while completed_trials < trials:
            _check_interrupt()
            candidate = _sample_candidate(space, rng)
            peak = peak_speed_for_strategy(strategy_name, candidate, sim_cfg, mc_cfg)

            if peak > max_speed:
                resampled += 1
                continue  # draw again — does NOT advance completed_trials

            cost, success_rate, mean_t = evaluate_candidate(
                candidate, strategy_name, sim_cfg, mc_cfg, fixed_offsets, max_workers
            )
            completed_trials += 1
            print(
                f"Trial {completed_trials}/{trials}: params={candidate} -> "
                f"Cost: {cost:.4f} (SR: {success_rate * 100:.1f}%, Mean T: {mean_t:.2f}s)"
            )

            if cost < best_cost:
                best_cost = cost
                best_params = candidate
    except (KeyboardInterrupt, OptimizationInterrupted):
        _note_interrupt()
        interrupted = True

    if resampled:
        print(f"  ({resampled} unphysical candidate(s) resampled during search)")
    return best_params, best_cost, interrupted, completed_trials


def run_grid_search(
    strategy_name: str,
    space: dict,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    grid_points: int,
    max_workers: int,
) -> SearchResult:
    """Grid Search optimization. Best for 1–2 parameters.

    Grid points that exceed the speed limit are skipped (the grid is fixed so
    there is no notion of resampling — the skipped points are simply omitted).
    """
    print("Starting Grid Search...")
    interrupted = False
    keys = list(space.keys())

    if len(keys) > 3:
        print(
            f"Warning: Grid search on {len(keys)} parameters is very expensive. "
            f"Only optimizing the first 2 parameters: {keys[:2]}."
        )
        keys = keys[:2]

    grids = []
    for k in keys:
        ptype, start, end = space[k]
        if ptype == "float":
            grids.append(np.linspace(start, end, grid_points))
        elif ptype == "int":
            grids.append(np.unique(np.round(np.linspace(start, end, grid_points)).astype(int)))

    grid_coords = np.meshgrid(*grids)
    flat_coords = [c.flatten() for c in grid_coords]
    num_runs = len(flat_coords[0])

    best_params: dict = {}
    best_cost = float("inf")
    max_speed = sim_cfg.satellite.max_beam_speed
    skipped = 0
    completed_trials = 0

    try:
        for idx in range(num_runs):
            _check_interrupt()
            candidate: dict = {}
            for i, k in enumerate(keys):
                ptype = space[k][0]
                val = flat_coords[i][idx]
                candidate[k] = int(val) if ptype == "int" else float(val)

            # Fill non-grid keys with their minimum bound as default
            for k in space:
                if k not in candidate:
                    candidate[k] = space[k][1]

            peak = peak_speed_for_strategy(strategy_name, candidate, sim_cfg, mc_cfg)
            if peak > max_speed:
                skipped += 1
                continue  # skip this grid point entirely

            cost, success_rate, mean_t = evaluate_candidate(
                candidate, strategy_name, sim_cfg, mc_cfg, fixed_offsets, max_workers
            )
            completed_trials += 1
            print(
                f"Grid point {completed_trials} (of {num_runs - skipped} valid): params={candidate} -> "
                f"Cost: {cost:.4f} (SR: {success_rate * 100:.1f}%, Mean T: {mean_t:.2f}s)"
            )

            if cost < best_cost:
                best_cost = cost
                best_params = candidate
    except (KeyboardInterrupt, OptimizationInterrupted):
        _note_interrupt()
        interrupted = True

    if skipped:
        print(f"  ({skipped}/{num_runs} grid points skipped — exceeded max_beam_speed)")
    return best_params, best_cost, interrupted, completed_trials


def run_optuna_search(
    strategy_name: str,
    space: dict,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    trials: int,
    max_workers: int,
    seed: int | None = None,
) -> SearchResult:
    """Optuna study optimization. Intelligent Bayesian Search."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print(
            "Optuna is not installed. To use this method:\n"
            "    pip install optuna\n"
            "Falling back to Random Search instead..."
        )
        return run_random_search(
            strategy_name, space, sim_cfg, mc_cfg, fixed_offsets, trials, max_workers, seed=seed
        )

    interrupted = False

    max_speed = sim_cfg.satellite.max_beam_speed
    print(f"Starting Optuna Search ({trials} trials, max_beam_speed={max_speed:.4f} rad/s)...")

    import os
    import concurrent.futures
    import threading
    from satellite.config import MonteCarloConfig, StrategyConfig
    
    # Check if GPU batching is supported
    from satellite.cuda_monte_carlo import is_strategy_chain_supported_on_gpu, run_monte_carlo_cuda_batch, CUDA_AVAILABLE
    
    # Construct a valid dummy parameters object using midpoints of the search space
    dummy_params = {}
    for param, spec in space.items():
        ptype, start, end = spec
        dummy_params[param] = (start + end) / 2.0 if ptype == "float" else int((start + end) / 2)
        
    if strategy_name == "lissajous_scan":
        dummy_params["s1_delta"] = 1.570796
        dummy_params["s2_delta"] = 1.570796
        
    from satellite.strategy.base import CONFIG_PARSERS
    if strategy_name in CONFIG_PARSERS:
        dummy_params = CONFIG_PARSERS[strategy_name](dummy_params)
        
    dummy_strategy = StrategyConfig(
        k=mc_cfg.strategy.k,
        chain=(strategy_name,),
        params={strategy_name: dummy_params},
    )
    dummy_mc = MonteCarloConfig(
        simulation_path=mc_cfg.simulation_path,
        seed=mc_cfg.seed,
        runs=len(fixed_offsets),
        chain=(strategy_name,),
        error=mc_cfg.error,
        strategy=dummy_strategy,
    )
    
    use_gpu_batch = (os.environ.get("SATELLITE_NO_GPU") != "1") and is_strategy_chain_supported_on_gpu(dummy_mc)
    
    completed_trials = 0
    resampled = 0

    if seed is not None:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize")

    if use_gpu_batch:
        BATCH_SIZE = max_workers if max_workers > 1 else 8
        MAX_PRUNE_ATTEMPTS = BATCH_SIZE * 200  # give up filling a batch after this many consecutive pruned candidates
        print(f"Using GPU batch execution (batch size {BATCH_SIZE})...")
        try:
            while completed_trials < trials:
                _check_interrupt()
                batch_trials = []
                batch_configs = []
                consecutive_prunes = 0

                while len(batch_trials) < BATCH_SIZE and completed_trials + len(batch_trials) < trials:
                    if consecutive_prunes >= MAX_PRUNE_ATTEMPTS:
                        print(
                            f"ERROR: {MAX_PRUNE_ATTEMPTS} consecutive candidates rejected by speed check. "
                            "The entire search space may exceed max_beam_speed. Aborting."
                        )
                        batch_trials = []  # trigger the break below
                        break

                    trial = study.ask()
                    candidate = {}
                    for param, spec in space.items():
                        ptype, start, end = spec
                        if ptype == "float":
                            candidate[param] = trial.suggest_float(param, start, end)
                        elif ptype == "int":
                            candidate[param] = trial.suggest_int(param, start, end)

                    cand_eval = dict(candidate)
                    if strategy_name == "lissajous_scan":
                        cand_eval["s1_delta"] = 1.570796
                        cand_eval["s2_delta"] = 1.570796

                    peak = peak_speed_for_strategy(strategy_name, cand_eval, sim_cfg, mc_cfg)
                    if peak > max_speed:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        resampled += 1
                        consecutive_prunes += 1
                        continue

                    consecutive_prunes = 0
                    batch_trials.append(trial)

                    from satellite.strategy.base import CONFIG_PARSERS
                    parsed_params = cand_eval
                    if strategy_name in CONFIG_PARSERS:
                        parsed_params = CONFIG_PARSERS[strategy_name](cand_eval)

                    run_strat = StrategyConfig(
                        k=mc_cfg.strategy.k,
                        chain=(strategy_name,),
                        params={strategy_name: parsed_params},
                    )
                    mc = MonteCarloConfig(
                        simulation_path=mc_cfg.simulation_path,
                        seed=mc_cfg.seed,
                        runs=len(fixed_offsets),
                        chain=(strategy_name,),
                        error=mc_cfg.error,
                        strategy=run_strat,
                    )
                    batch_configs.append(mc)

                if not batch_trials:
                    break

                try:
                    summaries = run_monte_carlo_cuda_batch(batch_configs)
                    for trial, summary, mc_c in zip(batch_trials, summaries, batch_configs):
                        success_rate = summary.success_rate
                        timeout = sim_cfg.simulation.timeout
                        mean_t = summary.mean_t if summary.mean_t is not None else timeout
                        penalty = timeout * 2.0
                        cost = ((1.0 - success_rate) * penalty) + mean_t

                        study.tell(trial, cost)
                        completed_trials += 1

                        from dataclasses import asdict
                        log_params = asdict(mc_c.strategy.params[strategy_name])
                        print(f"Trial {completed_trials}/{trials}: params={log_params} -> Cost: {cost:.4f}")
                except Exception as e:
                    for trial in batch_trials:
                        try:
                            study.tell(trial, state=optuna.trial.TrialState.FAIL)
                            completed_trials += 1
                        except ValueError:
                            pass
                    print(f"Batch execution FAILED — {e}")
        except (KeyboardInterrupt, OptimizationInterrupted):
            _note_interrupt()
            interrupted = True

    else:
        # Fallback to multi-threaded CPU parallel execution
        lock = threading.Lock()
        
        def worker():
            nonlocal completed_trials, resampled
            while True:
                with lock:
                    if completed_trials >= trials or _interrupt_requested:
                        break
                    trial = study.ask()

                candidate: dict = {}
                for param, spec in space.items():
                    ptype, start, end = spec
                    if ptype == "float":
                        candidate[param] = trial.suggest_float(param, start, end)
                    elif ptype == "int":
                        candidate[param] = trial.suggest_int(param, start, end)

                peak = peak_speed_for_strategy(strategy_name, candidate, sim_cfg, mc_cfg)
                if peak > max_speed:
                    with lock:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        resampled += 1
                    continue

                try:
                    cost, _, _ = evaluate_candidate(
                        candidate, strategy_name, sim_cfg, mc_cfg, fixed_offsets, 1
                    )
                    with lock:
                        study.tell(trial, cost)
                        completed_trials += 1
                        print(f"Trial {completed_trials}/{trials}: params={candidate} -> Cost: {cost:.4f}")
                except OptimizationInterrupted:
                    return
                except Exception as e:
                    with lock:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        completed_trials += 1
                        print(f"Trial {completed_trials}/{trials}: FAILED — {e}")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(worker) for _ in range(max_workers)]
                concurrent.futures.wait(futures)
        except (KeyboardInterrupt, OptimizationInterrupted):
            _note_interrupt()
            interrupted = True
        if _interrupt_requested:
            interrupted = True

    if resampled:
        print(f"  ({resampled} unphysical candidate(s) resampled during search)")

    if study.best_trial is None:
        if not interrupted:
            print("WARNING: no valid parameters found.")
        return {}, float("inf"), interrupted, completed_trials

    return study.best_params, study.best_value, interrupted, completed_trials


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import tomllib
    from dataclasses import replace
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Optimize parameters for pointing acquisition strategies."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("config/Optimize.toml"),
        help="Optimize TOML (default: config/Optimize.toml)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["random_curve", "center_rebias", "lissajous_scan", "rosette_scan",
                 "dual_spiral", "dual_raster", "concentric_shells"],
        help="Strategy name to optimize. Overrides config value.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random", "optuna"],
        help="Optimization method to employ. Overrides config value.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Number of trials for random/optuna search. Overrides config value.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=5,
        help="Points per parameter axis for grid search.",
    )
    parser.add_argument(
        "--eval-runs",
        type=int,
        help="Number of pre-sampled initial offset scenarios for evaluation. Overrides config value.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        help="Path to Environment.toml config. Overrides config value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for pre-sampling pointing offsets. Overrides config value.",
    )
    parser.add_argument(
        "--trial-seed",
        type=int,
        help="Random seed for random/optuna search. Overrides config value.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum CPU processes to use (defaults to cpu_count - 1).",
    )

    args = parser.parse_args()

    # Locate and resolve the config file path
    requested_config = args.config
    config_path = requested_config
    if not config_path.exists():
        # Fallback to look in config folder or current folder
        for candidate in [Path("config") / config_path.name, Path(config_path.name)]:
            if candidate.exists():
                config_path = candidate
                break

    toml_data = {}
    if config_path.exists():
        print(f"Loading configuration from {config_path}")
        with config_path.open("rb") as f:
            toml_data = tomllib.load(f)
    else:
        # If user explicitly passed a custom config that doesn't exist, raise error
        if requested_config != Path("config/Optimize.toml"):
            print(f"ERROR: Configuration file not found at {requested_config}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Warning: Default configuration file '{requested_config}' not found. Using defaults.")

    opt_sec = toml_data.get("optimize", {})
    mc_sec = toml_data.get("monte_carlo", {})

    # Resolve strategy
    strategy = args.strategy or opt_sec.get("strategy")
    if not strategy:
        print("ERROR: Strategy must be specified either in the config file or via --strategy", file=sys.stderr)
        sys.exit(1)

    # Resolve optimization parameters
    method = args.method or opt_sec.get("method") or "random"
    trials = args.trials if args.trials is not None else opt_sec.get("trials", 20)
    grid_points = args.grid_points
    trial_seed = args.trial_seed if args.trial_seed is not None else opt_sec.get("trial_seed")

    # Load / resolve Monte Carlo configuration
    if config_path.exists() and "monte_carlo" in toml_data:
        mc_cfg = load_monte_carlo_config(config_path)
    else:
        # Construct a default MonteCarloConfig
        from satellite.config import StrategyConfig, GaussianErrorConfig, MonteCarloConfig
        
        env_path = args.env_config or mc_sec.get("environment") or "config/Environment.toml"
        env_path_obj = Path(env_path)
        if not env_path_obj.exists():
            for candidate in [Path("config") / env_path_obj.name, Path(env_path_obj.name)]:
                if candidate.exists():
                    env_path_obj = candidate
                    break
                    
        sim_bundle = load_simulation_config(env_path_obj)
        error_dist = GaussianErrorConfig(distribution="gaussian", mean=0.0, std=2.0 * 1e-3)
        
        mc_cfg = MonteCarloConfig(
            simulation_path=env_path_obj.resolve(),
            seed=42,
            error=error_dist,
            strategy=StrategyConfig(k=sim_bundle.satellite.k, chain=(strategy,), params={}),
            runs=30,
            chain=(strategy,),
        )

    # Apply command-line and config-level overrides to mc_cfg
    if args.env_config is not None:
        env_path = Path(args.env_config)
        if not env_path.exists():
            for candidate in [Path("config") / env_path.name, Path(env_path.name)]:
                if candidate.exists():
                    env_path = candidate
                    break
        mc_cfg = replace(mc_cfg, simulation_path=env_path.resolve())

    if args.seed is not None:
        mc_cfg = replace(mc_cfg, seed=args.seed)
    elif mc_cfg.seed == 0 and not ("seed" in mc_sec):
        mc_cfg = replace(mc_cfg, seed=42)

    if args.eval_runs is not None:
        mc_cfg = replace(mc_cfg, runs=args.eval_runs)
    elif mc_cfg.runs == 1 and not ("runs" in mc_sec):
        # load_monte_carlo_config sets runs=1 as default if not in TOML.
        # If it wasn't in TOML and not in CLI, set it to the default of 30.
        mc_cfg = replace(mc_cfg, runs=30)

    sim_cfg = load_simulation_config(mc_cfg.simulation_path)

    # Update mc_cfg's strategy to match the resolved strategy
    from satellite.config import StrategyConfig
    mc_cfg = replace(
        mc_cfg,
        chain=(strategy,),
        strategy=StrategyConfig(
            k=mc_cfg.strategy.k if (hasattr(mc_cfg, "strategy") and mc_cfg.strategy) else sim_cfg.satellite.k,
            chain=(strategy,),
            params={strategy: {}},
        ),
    )

    print(
        f"Pre-sampling {mc_cfg.runs} pointing offsets "
        f"from the configuration error distribution..."
    )
    rng = np.random.default_rng(mc_cfg.seed)
    fixed_offsets = [sample_offsets(mc_cfg.error, rng) for _ in range(mc_cfg.runs)]

    max_workers = args.workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using up to {max_workers} processes in parallel.")

    space = build_parameter_spaces(sim_cfg, mc_cfg)[strategy]
    print(f"\nOptimizing strategy: '{strategy}' with search space:")
    for param, spec in space.items():
        print(f"  {param}: {spec[0]} in [{spec[1]}, {spec[2]}]")

    # Warn early if the entire search space is physically infeasible
    max_speed = sim_cfg.satellite.max_beam_speed
    hi_params = {p: spec[2] for p, spec in space.items()}  # all parameters at upper bound
    hi_peak = peak_speed_for_strategy(strategy, hi_params, sim_cfg, mc_cfg)
    lo_params = {p: spec[1] for p, spec in space.items()}  # all parameters at lower bound
    lo_peak = peak_speed_for_strategy(strategy, lo_params, sim_cfg, mc_cfg)
    print(
        f"\nSpeed range across search space: {lo_peak:.4f} – {hi_peak:.4f} rad/s "
        f"(limit: {max_speed:.4f} rad/s)"
    )
    if lo_peak > max_speed:
        print(
            "ERROR: Even the minimum-speed parameters exceed the hardware speed limit.\n"
            "Adjust the bounds in build_parameter_spaces() in __main__.py before proceeding."
        )
        sys.exit(1)

    start_time = time.time()
    best_params: dict = {}
    best_cost = float("inf")
    interrupted = False
    completed_trials = 0

    try:
        if method == "grid":
            best_params, best_cost, interrupted, completed_trials = run_grid_search(
                strategy, space, sim_cfg, mc_cfg, fixed_offsets, grid_points, max_workers
            )
        elif method == "optuna":
            best_params, best_cost, interrupted, completed_trials = run_optuna_search(
                strategy, space, sim_cfg, mc_cfg, fixed_offsets, trials, max_workers, seed=trial_seed
            )
        else:
            best_params, best_cost, interrupted, completed_trials = run_random_search(
                strategy, space, sim_cfg, mc_cfg, fixed_offsets, trials, max_workers, seed=trial_seed
            )
    except (KeyboardInterrupt, OptimizationInterrupted):
        _note_interrupt()
        interrupted = True

    elapsed = time.time() - start_time
    if interrupted:
        print(
            f"\n--- Optimization interrupted after {elapsed:.1f}s "
            f"({completed_trials} trial(s) completed) ---"
        )
    else:
        print(f"\n--- Optimization completed in {elapsed:.1f}s ---")

    if best_params:
        print(f"Best objective cost score: {best_cost:.4f}")

        if strategy == "lissajous_scan":
            best_params["s1_delta"] = 1.570796
            best_params["s2_delta"] = 1.570796

        print("Optimal Parameters:")
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"  {k} = {v:.6f}")
            else:
                print(f"  {k} = {v}")
    elif interrupted and completed_trials == 0:
        print("Stopped before any trials finished.")

    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
