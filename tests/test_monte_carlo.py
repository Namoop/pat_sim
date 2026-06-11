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
    summary = run_monte_carlo(mc)
    assert summary.runs == 15
    assert summary.success_rate >= 0.8


def test_monte_carlo_failure_biased_fixture():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_fail.toml")
    mc = replace(mc, simulation_path=(REPO_ROOT / "Simulation.toml").resolve())
    summary = run_monte_carlo(mc)
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
        assert " at q=" in complete
    else:
        assert " Failed after q=" in complete
        assert "timeout (tried " in complete


def test_monte_carlo_graceful_interrupt(monkeypatch):
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    mc = replace(
        mc,
        simulation_path=(REPO_ROOT / "Simulation.toml").resolve(),
        runs=5,
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
    summary = run_monte_carlo(mc)
    assert summary.interrupted is True
    assert summary.runs == 1
    assert summary.planned_runs == 5
    text = format_monte_carlo_summary(summary)
    assert "Interrupted after 1/5 runs." in text
    assert "Monte Carlo: " in text
