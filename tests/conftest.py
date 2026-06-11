# -*- coding: utf-8 -*-
"""Shared scenario fixtures for strategy tests."""

from __future__ import annotations

from satellite.config import (
    AsymmetricProbeStrategyConfig,
    MapVisualizationConfig,
    MinorOffsetStrategyConfig,
    SatelliteInstanceConfig,
    ScenarioConfig,
    SharedSatelliteConfig,
    SimulationConfig,
    SingleMissStrategyConfig,
    StrategyConfig,
    VisualizationConfig,
)
from satellite.math3d import as_vec3


def base_config(
    *,
    s1_theta: float = 0.001,
    s1_phi: float = 0.001,
    s2_theta: float = 0.001,
    s2_phi: float = 0.001,
    step_duration: float = 5.0,
    chain: tuple[str, ...] = ("minor_offset", "single_miss"),
    reset_duration: float = 0.3,
) -> ScenarioConfig:
    return ScenarioConfig(
        name="test",
        s1=SatelliteInstanceConfig(
            position=as_vec3([0.0, 0.0, 0.0]),
            bench_theta_offset=s1_theta,
            bench_phi_offset=s1_phi,
        ),
        s2=SatelliteInstanceConfig(
            position=as_vec3([1000.0, 0.0, 0.0]),
            bench_theta_offset=s2_theta,
            bench_phi_offset=s2_phi,
        ),
        satellite=SharedSatelliteConfig(
            body_radius=0.5,
            dish_fov=0.02,
            bench_slew_time=0.3,
            fsm_settle_time=0.0,
            beam_width_mrad=5.0,
        ),
        simulation=SimulationConfig(
            distance=1000.0,
            t_step=0.01,
            beam_length=None,
            boresight_extension=5.0,
            profile_replay=False,
        ),
        visualization=VisualizationConfig(
            enabled=False,
            cone_u_steps=8,
            cone_v_steps=8,
            spiral_trail_steps=20,
            ribbon_v_steps=4,
            profile_frames=False,
        ),
        map_visualization=MapVisualizationConfig(
            axis_limit=0.1,
            profile_frames=False,
            slider_debounce_ms=16,
        ),
        strategy=StrategyConfig(
            k=10.0,
            chain=chain,
            minor_offset=MinorOffsetStrategyConfig(
                duration=step_duration,
                max_spiral_radius="fov",
                spiral_speed=1.0,
            ),
            single_miss=SingleMissStrategyConfig(
                phase1_duration=step_duration,
                a_spiral_radius=0.05,
                reset_duration=reset_duration,
                phase2_duration=step_duration,
                b_spiral_radius=0.05,
                spiral_speed=1.0,
            ),
            asymmetric_probe=AsymmetricProbeStrategyConfig(
                probe_duration=step_duration,
                spiral_radius=0.05,
                spiral_speed=1.0,
                reset_duration=reset_duration,
            ),
        ),
    )
