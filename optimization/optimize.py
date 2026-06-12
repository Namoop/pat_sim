#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pointing Acquisition Strategy Parameter Optimizer.
Supports Grid Search, Random Search, and Optuna.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import sys
import time
import numpy as np

from satellite.config import (
    load_simulation_config,
    load_monte_carlo_config,
    ScenarioInstance,
    build_scenario_config,
    StrategyConfig
)
from satellite.monte_carlo import sample_offsets
from satellite.scenario import run_scenario

# Define standard search spaces for built-in strategies
PARAMETER_SPACES = {
    "random_curve": {
        "velocity_a": ("float", 0.001, 0.05),
        "velocity_ratio": ("float", 1.0, 3.0),
        "drift_sigma": ("float", 0.001, 0.5),
        "max_turn_radius": ("float", 0.1, 2.0),
    },
    "center_rebias": {
        "velocity_a": ("float", 0.001, 0.05),
        "velocity_ratio": ("float", 1.0, 3.0),
        "drift_sigma": ("float", 0.001, 0.5),
        "max_turn_radius": ("float", 0.1, 2.0),
        "bias_strength": ("float", 0.0, 1.0),
    },
    "lissajous_scan": {
        "s1_wx": ("float", 0.1, 10.0),
        "s1_wy": ("float", 0.1, 10.0),
        "s1_delta": ("float", 0.0, 6.283185),
        "s2_wx": ("float", 0.1, 10.0),
        "s2_wy": ("float", 0.1, 10.0),
        "s2_delta": ("float", 0.0, 6.283185),
    },
    "rosette_scan": {
        "s1_w1": ("float", 0.1, 10.0),
        "s1_w2": ("float", 0.1, 10.0),
        "s2_w1": ("float", 0.1, 10.0),
        "s2_w2": ("float", 0.1, 10.0),
    },
    "random_walk": {
        "step_duration": ("float", 0.05, 5.0),
    },
    "dual_spiral": {
        "speed_a": ("float", 0.1, 5.0),
        "speed_b": ("float", 0.1, 5.0),
    },
    "dual_raster": {
        "steps_a": ("int", 5, 100),
        "steps_b": ("int", 5, 100),
    },
    "concentric_shells": {
        "spiral_speed_a": ("float", 0.1, 5.0),
        "speed_ratio": ("float", 1.0, 3.0),
    },
}


def evaluate_single_instance(sim_cfg, instance, strategy) -> tuple[bool, float | None]:
    """Runs a single scenario simulation. Runs in a process pool."""
    config = build_scenario_config(sim_cfg, instance, strategy=strategy)
    res = run_scenario(config)
    return res.success, res.hit_at_t


def evaluate_candidate(
    params: dict,
    strategy_name: str,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    max_workers: int
) -> tuple[float, float, float]:
    from satellite.strategy.base import CONFIG_PARSERS

    parsed_params = params
    if strategy_name in CONFIG_PARSERS:
        parsed_params = CONFIG_PARSERS[strategy_name](params)

    strategy = StrategyConfig(
        k=mc_cfg.strategy.k,
        chain=(strategy_name,),
        params={strategy_name: parsed_params}
    )

    tasks = []
    for idx, (s1_off, s2_off) in enumerate(fixed_offsets):
        instance = ScenarioInstance(
            name=f"eval_{idx}",
            distance=None,
            s1=s1_off,
            s2=s2_off
        )
        tasks.append((sim_cfg, instance, strategy))

    successes = 0
    hit_times = []

    # Run simulations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_instance, *task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            success, hit_at_t = fut.result()
            if success:
                successes += 1
                if hit_at_t is not None:
                    hit_times.append(hit_at_t)

    success_rate = successes / len(fixed_offsets)
    timeout = sim_cfg.simulation.timeout
    mean_t = float(np.mean(hit_times)) if hit_times else timeout
    
    # Cost function: prioritize lock rate first, then speed
    penalty = timeout * 2.0
    cost = ((1.0 - success_rate) * penalty) + mean_t

    return cost, success_rate, mean_t


def run_random_search(
    strategy_name: str,
    space: dict,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    trials: int,
    max_workers: int
) -> tuple[dict, float]:
    """Basic Random Search optimization."""
    best_params = {}
    best_cost = float("inf")

    print(f"Starting Random Search ({trials} trials)...")
    for t in range(trials):
        candidate = {}
        for param, spec in space.items():
            ptype, start, end = spec
            if ptype == "float":
                candidate[param] = float(np.random.uniform(start, end))
            elif ptype == "int":
                candidate[param] = int(np.random.randint(start, end + 1))

        cost, success_rate, mean_t = evaluate_candidate(
            candidate, strategy_name, sim_cfg, mc_cfg, fixed_offsets, max_workers
        )
        print(f"Trial {t+1}/{trials}: params={candidate} -> Cost: {cost:.4f} (SR: {success_rate * 100:.1f}%, Mean T: {mean_t:.2f}s)")
        
        if cost < best_cost:
            best_cost = cost
            best_params = candidate

    return best_params, best_cost


def run_grid_search(
    strategy_name: str,
    space: dict,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    grid_points: int,
    max_workers: int
) -> tuple[dict, float]:
    """Basic Grid Search optimization. Best for 1-2 parameters."""
    print("Starting Grid Search...")
    keys = list(space.keys())
    
    # We restrict grid search to at most 3 parameters to prevent combinatorics explosion
    if len(keys) > 3:
        print(f"Warning: Grid search on {len(keys)} parameters is very expensive. Only optimizing the first 2 parameters: {keys[:2]}.")
        keys = keys[:2]

    grids = []
    for k in keys:
        ptype, start, end = space[k]
        if ptype == "float":
            grids.append(np.linspace(start, end, grid_points))
        elif ptype == "int":
            grids.append(np.unique(np.round(np.linspace(start, end, grid_points)).astype(int)))

    # Compute cartesian product
    grid_coords = np.meshgrid(*grids)
    flat_coords = [c.flatten() for c in grid_coords]
    num_runs = len(flat_coords[0])

    best_params = {}
    best_cost = float("inf")

    for idx in range(num_runs):
        candidate = {}
        for i, k in enumerate(keys):
            ptype = space[k][0]
            val = flat_coords[i][idx]
            candidate[k] = int(val) if ptype == "int" else float(val)

        # Retain any parameters not in our grid keys as defaults
        for k in space:
            if k not in candidate:
                candidate[k] = space[k][1] # use min bound as default

        cost, success_rate, mean_t = evaluate_candidate(
            candidate, strategy_name, sim_cfg, mc_cfg, fixed_offsets, max_workers
        )
        print(f"Grid point {idx+1}/{num_runs}: params={candidate} -> Cost: {cost:.4f} (SR: {success_rate * 100:.1f}%, Mean T: {mean_t:.2f}s)")

        if cost < best_cost:
            best_cost = cost
            best_params = candidate

    return best_params, best_cost


def run_optuna_search(
    strategy_name: str,
    space: dict,
    sim_cfg,
    mc_cfg,
    fixed_offsets: list,
    trials: int,
    max_workers: int
) -> tuple[dict, float]:
    """Optuna study optimization. Intelligent Bayesian Search."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna is not installed in the current environment. To use this optimizer method:")
        print("    pip install optuna")
        print("Falling back to Random Search instead...")
        return run_random_search(
            strategy_name, space, sim_cfg, mc_cfg, fixed_offsets, trials, max_workers
        )

    print(f"Starting Optuna Search ({trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        candidate = {}
        for param, spec in space.items():
            ptype, start, end = spec
            if ptype == "float":
                candidate[param] = trial.suggest_float(param, start, end)
            elif ptype == "int":
                candidate[param] = trial.suggest_int(param, start, end)

        cost, _, _ = evaluate_candidate(
            candidate, strategy_name, sim_cfg, mc_cfg, fixed_offsets, max_workers
        )
        return cost

    study = optuna.create_study(direction="minimize")
    
    # Wrap in a loop so we can print trial-by-trial logs
    for t in range(trials):
        trial = study.ask()
        try:
            val = objective(trial)
            study.tell(trial, val)
            print(f"Trial {t+1}/{trials}: params={trial.params} -> Cost: {val:.4f}")
        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"Trial {t+1}/{trials} failed: {e}")

    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(description="Optimize parameters for pointing acquisition strategies.")
    parser.add_argument("--strategy", type=str, required=True, choices=list(PARAMETER_SPACES.keys()),
                        help="Strategy name to optimize.")
    parser.add_argument("--method", type=str, choices=["grid", "random", "optuna"], default="random",
                        help="Optimization method to employ.")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for random/optuna search.")
    parser.add_argument("--grid-points", type=int, default=5,
                        help="Points per parameter axis for grid search.")
    parser.add_argument("--eval-runs", type=int, default=30,
                        help="Number of pre-sampled initial offset scenarios for evaluation.")
    parser.add_argument("--sim-config", type=str, default="Simulation.toml",
                        help="Path to Simulation.toml config.")
    parser.add_argument("--mc-config", type=str, default="MonteCarlo_all.toml",
                        help="Path to MonteCarlo.toml config.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pre-sampling pointing offsets.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Maximum CPU processes to use (defaults to cpu_count - 1).")

    args = parser.parse_args()

    # Load configs
    sim_cfg = load_simulation_config(args.sim_config)
    mc_cfg = load_monte_carlo_config(args.mc_config)

    # Pre-sample initial offsets for evaluation (Common Random Numbers)
    print(f"Pre-sampling {args.eval_runs} pointing offsets from the configuration error distribution...")
    rng = np.random.default_rng(args.seed)
    fixed_offsets = [sample_offsets(mc_cfg.error, rng) for _ in range(args.eval_runs)]

    # Worker count selection
    max_workers = args.workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using up to {max_workers} processes in parallel.")

    space = PARAMETER_SPACES[args.strategy]
    print(f"\nOptimizing strategy: '{args.strategy}' with search space:")
    for param, spec in space.items():
        print(f"  {param}: {spec[0]} in [{spec[1]}, {spec[2]}]")

    start_time = time.time()

    if args.method == "grid":
        best_params, best_cost = run_grid_search(
            args.strategy, space, sim_cfg, mc_cfg, fixed_offsets, args.grid_points, max_workers
        )
    elif args.method == "optuna":
        best_params, best_cost = run_optuna_search(
            args.strategy, space, sim_cfg, mc_cfg, fixed_offsets, args.trials, max_workers
        )
    else:  # random
        best_params, best_cost = run_random_search(
            args.strategy, space, sim_cfg, mc_cfg, fixed_offsets, args.trials, max_workers
        )

    elapsed = time.time() - start_time
    print(f"\n--- Optimization completed in {elapsed:.1f}s ---")
    print(f"Best objective cost score: {best_cost:.4f}")
    print("Optimal Parameters:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")
        else:
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
