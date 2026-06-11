"""Dish detection regression tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from satellite.config import (
    BenchOffsetConfig,
    build_scenario_config,
    load_monte_carlo_config,
    load_scenario_config,
    load_simulation_config,
    positions_for_distance,
)
from satellite.detection import (
    beam_hits_dish,
    incoming_from_source,
    is_within_dish_fov,
)
from satellite.math3d import angle_between, normalize
from satellite.scenario import run_scenario
from satellite.strategy.base import beam_length_for, link_established


def test_incoming_from_source_points_at_transmitter():
    apex = np.array([1000.0, 0.0, 0.0])
    mount = np.array([0.0, 0.0, 0.5])
    incoming = incoming_from_source(apex, mount)
    expected = normalize(apex - mount)
    np.testing.assert_allclose(incoming, expected, atol=1e-12)


def test_off_axis_beam_rejected_when_source_outside_fov():
    sim = load_simulation_config("Simulation.toml")
    instance = load_scenario_config("default.toml")
    instance = replace(
        instance,
        s1=BenchOffsetConfig(0.02, 0.01),
        s2=BenchOffsetConfig(0.02, 0.015),
    )
    mc_strategy = load_monte_carlo_config("MonteCarlo.toml")
    cfg = build_scenario_config(sim, instance, strategy=mc_strategy.strategy)
    result = run_scenario(cfg)

    q = 6.09
    result.replay_to(q)
    aim2 = result.bench_aim("S2", q)
    geom1 = result.s1.receiver.geometry_snapshot()
    alpha = cfg.satellite.alpha
    beam_length = beam_length_for(result.s2, cfg)

    incoming = incoming_from_source(result.s2.position, geom1.mount)
    incident = angle_between(geom1.dish_boresight, incoming)
    assert incident > cfg.satellite.dish_fov
    assert not is_within_dish_fov(
        geom1.dish_boresight,
        incoming,
        cfg.satellite.dish_fov,
    )

    assert not beam_hits_dish(
        result.s2.position,
        geom1.mount,
        geom1.dish_boresight,
        cfg.satellite.dish_fov,
        aim2,
        alpha,
        beam_length,
    )
    assert not link_established(result.s2, result.s1, aim2, cfg)


def test_scenario_distance_override():
    sim = load_simulation_config("Simulation.toml")
    instance = load_scenario_config("default.toml")
    from dataclasses import replace

    overridden = replace(instance, distance=500.0)
    mc = load_monte_carlo_config("MonteCarlo.toml")
    cfg = build_scenario_config(sim, overridden, strategy=mc.strategy)
    s1_pos, s2_pos = positions_for_distance(500.0)
    np.testing.assert_allclose(cfg.s1.position, s1_pos)
    np.testing.assert_allclose(cfg.s2.position, s2_pos)
