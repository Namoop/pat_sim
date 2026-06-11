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
from satellite.monte_carlo import run_monte_carlo, sample_offsets

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
