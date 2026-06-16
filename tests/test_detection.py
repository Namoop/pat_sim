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

    t = 6.09
    result.replay_to(t)
    aim2 = result.bench_aim("S2", t)
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

    overridden = replace(instance, overrides={"simulation": {"distance": 500.0}})
    mc = load_monte_carlo_config("MonteCarlo.toml")
    cfg = build_scenario_config(sim, overridden, strategy=mc.strategy)
    s1_pos, s2_pos = positions_for_distance(500.0)
    np.testing.assert_allclose(cfg.s1.position, s1_pos)
    np.testing.assert_allclose(cfg.s2.position, s2_pos)


def test_scenario_generic_overrides():
    sim = load_simulation_config("Simulation.toml")
    instance = load_scenario_config("default.toml")
    from dataclasses import replace

    overridden = replace(
        instance,
        overrides={
            "simulation": {"t_step": 0.005},
            "satellite": {"max_fsm_radius": 0.5},
        }
    )
    mc = load_monte_carlo_config("MonteCarlo.toml")
    cfg = build_scenario_config(sim, overridden, strategy=mc.strategy)
    
    assert cfg.simulation.t_step == 0.005
    assert cfg.satellite.max_fsm_radius == 0.0005


def test_scenario_toml_simulation_path_and_override_parsing(tmp_path):
    toml_content = """
[scenario]
name = "override_test"
simulation_file = "Simulation.toml"
simulation.t_step = 0.005
chain = ["minor_offset"]

[s1]
bench_theta_offset = 1.0
bench_phi_offset = 1.0

[s2]
bench_theta_offset = 2.0
bench_phi_offset = 1.5
"""
    p = tmp_path / "scenario_test.toml"
    p.write_text(toml_content)
    
    # We also need a Simulation.toml in the same dir for resolving
    sim_content = """
[satellite]
body_radius = 0.5
dish_fov = 2.0
max_beam_speed = 87.0
max_fsm_speed = 1000.0
max_fsm_radius = 1.0
beam_width = 0.5
k = 10.0

[simulation]
distance = 1000.0
t_step = 0.01
boresight_extension = 5.0
max_search_radius = 5.0
profile_replay = false
timeout = 100.0
enforce_speed_limit = true

[visualization]
enabled = false
cone_u_steps = 24
cone_v_steps = 32
spiral_trail_steps = 120
ribbon_v_steps = 8
profile_frames = false

[map_visualization]
axis_limit = 10.0
slider_debounce_ms = 16
"""
    (tmp_path / "Simulation.toml").write_text(sim_content)
    
    # Load and build config
    from satellite.config import load_single_scenario
    cfg = load_single_scenario(p)
    
    # Check that simulation_path was resolved correctly and overrides were applied
    assert cfg.simulation.t_step == 0.005


