"""Monte Carlo config and runner tests."""

from __future__ import annotations

from pathlib import Path

from dataclasses import replace

import numpy as np

from satellite.config import (
    build_scenario_config,
    load_monte_carlo_config,
    load_scenario_config,
    load_simulation_config,
    positions_for_distance,
)
from satellite.monte_carlo import (
    format_monte_carlo_run_complete,
    format_monte_carlo_run_start,
    format_monte_carlo_summary,
    run_monte_carlo,
    run_monte_carlo_single,
    sample_offsets,
)
from satellite.scenario import run_scenario

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_uniform_sampling_respects_bounds():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    rng = np.random.default_rng(0)
    for _ in range(50):
        s1, s2 = sample_offsets(mc.error, rng)
        for theta, phi in (
            (s1.bench_theta_offset, s1.bench_phi_offset),
            (s2.bench_theta_offset, s2.bench_phi_offset),
        ):
            assert -0.001 <= theta <= 0.001
            assert -0.001 <= phi <= 0.001


def test_seeded_sampling_is_reproducible():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    draws_a = [sample_offsets(mc.error, rng_a) for _ in range(5)]
    draws_b = [sample_offsets(mc.error, rng_b) for _ in range(5)]
    assert draws_a == draws_b


def test_build_scenario_config_x_axis_positions():
    sim_path = REPO_ROOT / "Simulation.toml"
    sim = load_simulation_config(sim_path)
    instance = load_scenario_config(REPO_ROOT / "default.toml")
    mc = load_monte_carlo_config(REPO_ROOT / "MonteCarlo.toml")
    cfg = build_scenario_config(sim, instance, strategy=mc.strategy)
    s1_pos, s2_pos = positions_for_distance(sim.simulation.distance)
    np.testing.assert_allclose(cfg.s1.position, s1_pos)
    np.testing.assert_allclose(cfg.s2.position, s2_pos)


def test_monte_carlo_success_biased_fixture():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    mc = replace(mc, simulation_path=(REPO_ROOT / "Simulation.toml").resolve())
    summary = run_monte_carlo(mc, max_workers=1)
    assert summary.runs == 15
    assert summary.success_rate >= 0.8


def test_monte_carlo_failure_biased_fixture():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_fail.toml")
    mc = replace(mc, simulation_path=(REPO_ROOT / "Simulation.toml").resolve())
    summary = run_monte_carlo(mc, max_workers=1)
    assert summary.runs == 8
    assert summary.successes == 0


def test_monte_carlo_run_progress_messages():
    from satellite.monte_carlo import MonteCarloRunResult
    from tests.conftest import base_config

    result = run_scenario(base_config())
    mc_run = MonteCarloRunResult(
        run_index=0,
        s1_theta=0.01,
        s1_phi=-0.02,
        s2_theta=0.03,
        s2_phi=0.04,
        result=result,
        computation_time_ms=12.5,
    )
    start = format_monte_carlo_run_start(1, 10, 0.01, -0.02, 0.03, 0.04)
    assert start == (
        "Running scenario 1/10: "
        "S1_θ_off=10 mrad  S1_φ_off=-20 mrad  "
        "S2_θ_off=30 mrad  S2_φ_off=40 mrad"
    )
    complete = format_monte_carlo_run_complete(mc_run, elapsed_ms=12.5)
    assert complete.startswith("Completed in 12.5ms:")
    if result.success:
        assert " Success with " in complete
        assert " at t=" in complete
    else:
        assert " Failed after t=" in complete
        assert "timeout (tried " in complete


def test_monte_carlo_summary_statistics_formatting():
    from satellite.monte_carlo import MonteCarloSummary
    
    summary = MonteCarloSummary(
        runs=10,
        planned_runs=10,
        interrupted=False,
        successes=5,
        success_rate=0.5,
        by_strategy={"strat1": 5},
        run_results=(),
        mean_t=100.0,
        median_t=90.0,
        mean_computation_ms=50.0,
        median_computation_ms=45.0,
        total_computation_ms=500.0,
    )
    
    text = format_monte_carlo_summary(summary)
    assert "Monte Carlo: 5/10 succeeded (50.0%)" in text
    assert "Success sim-t: mean=100.000, median=90.000" in text
    assert "Computation: mean=50.0ms, median=45.0ms, total=0.50s" in text
    assert "Winning strategies: strat1: 5" in text


def test_monte_carlo_graceful_interrupt(monkeypatch):
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    from satellite.config import MonteCarloChainConfig
    mc = replace(
        mc,
        simulation_path=(REPO_ROOT / "Simulation.toml").resolve(),
        chains=(MonteCarloChainConfig(runs=5, chain=mc.strategy.chain),),
    )
    calls = {"n": 0}
    real_single = run_monte_carlo_single

    def interrupt_after_first(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt
        return real_single(*args, **kwargs)

    monkeypatch.setattr(
        "satellite.monte_carlo.run_monte_carlo_single",
        interrupt_after_first,
    )
    monkeypatch.setenv("SATELLITE_NO_GPU", "1")
    summary = run_monte_carlo(mc, max_workers=1)
    assert summary.interrupted is True
    assert summary.runs == 1
    assert summary.planned_runs == 5
    text = format_monte_carlo_summary(summary)
    assert "Interrupted after 1/5 runs." in text
    assert "Monte Carlo: " in text


def test_multiple_chains_toml_parsing(tmp_path):
    toml_content = """
[monte_carlo]
simulation = "Simulation.toml"
seed = 42

[[monte_carlo.chains]]
runs = 5
chain = ["minor_offset", "single_miss"]

[[monte_carlo.chains]]
runs = 3
chain = ["asymmetric_swap"]

[error]
distribution = "uniform"
theta_min = -0.02
theta_max = 0.02
phi_min = -0.02
phi_max = 0.02

[strategy]
k = 10.0
chain = ["minor_offset"]
"""
    p = tmp_path / "test_mc.toml"
    p.write_text(toml_content)
    
    mc = load_monte_carlo_config(p)
    assert mc.runs == 8
    assert mc.seed == 42
    assert len(mc.chains) == 2
    assert mc.chains[0].runs == 5
    assert mc.chains[0].chain == ("minor_offset", "single_miss")
    assert mc.chains[1].runs == 3
    assert mc.chains[1].chain == ("asymmetric_swap",)
    assert "single_miss" in mc.strategy.params
    assert "asymmetric_swap" in mc.strategy.params


def test_multiple_chains_prg_reproducibility(tmp_path):
    toml_a_content = """
[monte_carlo]
simulation = "Simulation.toml"
seed = 42

[[monte_carlo.chains]]
runs = 5
chain = ["minor_offset", "single_miss"]

[[monte_carlo.chains]]
runs = 3
chain = ["asymmetric_swap"]

[error]
distribution = "uniform"
theta_min = -0.02
theta_max = 0.02
phi_min = -0.02
phi_max = 0.02

[strategy]
k = 10.0
chain = ["minor_offset"]

[strategy.minor_offset]
max_spiral_radius = "fov"
spiral_speed = 0.4

[strategy.single_miss]
a_spiral_radius = 0.05
b_spiral_radius = 0.05
spiral_speed = 0.16

[strategy.asymmetric_swap]
spiral_radius = 0.05
lock_duration = 1.0
spiral_speed = 0.16
"""
    toml_b_content = """
[monte_carlo]
simulation = "Simulation.toml"
seed = 42

[[monte_carlo.chains]]
runs = 3
chain = ["asymmetric_swap"]

[[monte_carlo.chains]]
runs = 5
chain = ["minor_offset", "single_miss"]

[error]
distribution = "uniform"
theta_min = -0.02
theta_max = 0.02
phi_min = -0.02
phi_max = 0.02

[strategy]
k = 10.0
chain = ["minor_offset"]

[strategy.minor_offset]
max_spiral_radius = "fov"
spiral_speed = 0.4

[strategy.single_miss]
a_spiral_radius = 0.05
b_spiral_radius = 0.05
spiral_speed = 0.16

[strategy.asymmetric_swap]
spiral_radius = 0.05
lock_duration = 1.0
spiral_speed = 0.16
"""
    pa = tmp_path / "mc_a.toml"
    pa.write_text(toml_a_content)
    pb = tmp_path / "mc_b.toml"
    pb.write_text(toml_b_content)
    
    mc_a = load_monte_carlo_config(pa)
    mc_b = load_monte_carlo_config(pb)
    
    mc_a = replace(mc_a, simulation_path=(REPO_ROOT / "Simulation.toml").resolve())
    mc_b = replace(mc_b, simulation_path=(REPO_ROOT / "Simulation.toml").resolve())
    
    summary_a = run_monte_carlo(mc_a, max_workers=1)
    summary_b = run_monte_carlo(mc_b, max_workers=1)
    
    runs_a_swap = [r for r in summary_a.run_results if r.result.config.strategy.chain == ("asymmetric_swap",)]
    runs_a_spiral = [r for r in summary_a.run_results if r.result.config.strategy.chain == ("minor_offset", "single_miss")]
    
    runs_b_swap = [r for r in summary_b.run_results if r.result.config.strategy.chain == ("asymmetric_swap",)]
    runs_b_spiral = [r for r in summary_b.run_results if r.result.config.strategy.chain == ("minor_offset", "single_miss")]
    
    assert len(runs_a_swap) == 3
    assert len(runs_b_swap) == 3
    assert len(runs_a_spiral) == 5
    assert len(runs_b_spiral) == 5
    
    for r_a, r_b in zip(runs_a_swap, runs_b_swap):
        assert r_a.s1_theta == r_b.s1_theta
        assert r_a.s1_phi == r_b.s1_phi
        assert r_a.s2_theta == r_b.s2_theta
        assert r_a.s2_phi == r_b.s2_phi
        
    for r_a, r_b in zip(runs_a_spiral, runs_b_spiral):
        assert r_a.s1_theta == r_b.s1_theta
        assert r_a.s1_phi == r_b.s1_phi
        assert r_a.s2_theta == r_b.s2_theta
        assert r_a.s2_phi == r_b.s2_phi

