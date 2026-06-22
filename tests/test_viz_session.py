"""Visualization session tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from montecarlo.run import load_monte_carlo_config, run_monte_carlo_single
from scenario.run import load_single_scenario, run_scenario
from satellite.config import load_simulation_config
from visualize.session import MonteCarloVizSession, SingleResultSession

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _mc_fixture():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    return replace(mc, simulation_path=(REPO_ROOT / "config/Environment.toml").resolve())


def test_single_result_session_advance_closes():
    cfg = load_single_scenario(
        REPO_ROOT / "config/Scenario.toml",
        REPO_ROOT / "config/Environment.toml",
    )
    result = run_scenario(cfg)
    session = SingleResultSession(result)
    assert session.current() is result
    assert session.has_next() is False
    assert session.advance() is None


def test_monte_carlo_viz_session_first_run_reproducible():
    mc = _mc_fixture()
    mc = replace(mc, runs=3, chain=mc.strategy.chain)
    sim = load_simulation_config(mc.simulation_path)
    # Match the logic in MonteCarloVizSession: 
    # first run uses first integer from rng seeded with mc.seed
    rng = np.random.default_rng(mc.seed)
    seed = int(rng.integers(0, 2**32 - 1))
    expected = run_monte_carlo_single(mc, sim, seed, 0)

    session = MonteCarloVizSession(mc)
    first = session.current()
    assert first.config.name == "mc_run_0"
    np.testing.assert_allclose(
        first.config.s1.bench_theta_offset,
        expected.s1_theta,
    )
    assert session.has_next() is True
    assert "1/3" in session.status_label()


def test_monte_carlo_viz_session_advance_and_last_returns_none():
    mc = _mc_fixture()
    mc = replace(mc, runs=3, chain=mc.strategy.chain)
    session = MonteCarloVizSession(mc)

    r0 = session.current()
    r1 = session.advance()
    assert r1 is not None
    assert r1.config.name == "mc_run_1"
    assert r1 is not r0

    r2 = session.advance()
    assert r2 is not None
    assert r2.config.name == "mc_run_2"
    assert session.has_next() is False

    assert session.advance() is None


def test_monte_carlo_viz_session_offsets_match_batch():
    mc = _mc_fixture()
    mc = replace(mc, runs=2, chain=mc.strategy.chain)
    sim = load_simulation_config(mc.simulation_path)

    rng = np.random.default_rng(mc.seed)
    seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(2)]
    batch = [run_monte_carlo_single(mc, sim, seeds[i], i) for i in range(2)]

    session = MonteCarloVizSession(mc)
    assert session.current().config.s1.bench_theta_offset == batch[0].s1_theta
    second = session.advance()
    assert second is not None
    assert second.config.s1.bench_theta_offset == batch[1].s1_theta


def test_monte_carlo_viz_session_start_run():
    mc = _mc_fixture()
    mc = replace(mc, runs=3, chain=mc.strategy.chain)
    sim = load_simulation_config(mc.simulation_path)

    rng = np.random.default_rng(mc.seed)
    seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(3)]
    expected = run_monte_carlo_single(mc, sim, seeds[1], 1)

    session = MonteCarloVizSession(mc, start_run=2)
    assert session._run_index == 1
    first = session.current()
    assert first.config.name == "mc_run_1"
    np.testing.assert_allclose(
        first.config.s1.bench_theta_offset,
        expected.s1_theta,
    )
    assert "2/3" in session.status_label()
    assert session.has_next() is True
