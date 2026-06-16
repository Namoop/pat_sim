# -*- coding: utf-8 -*-
"""Shared scenario fixtures for strategy tests."""

from __future__ import annotations

from satellite.config import (
    MapVisualizationConfig,
    SatelliteInstanceConfig,
    ScenarioConfig,
    SharedSatelliteConfig,
    SimulationConfig,
    StrategyConfig,
    VisualizationConfig,
)
from satellite.math3d import as_vec3
from satellite.strategy.strategies.asymmetric_swap import AsymmetricSwapConfig
from satellite.strategy.strategies.minor_offset import MinorOffsetConfig
from satellite.strategy.strategies.single_miss import SingleMissConfig
from satellite.strategy.strategies.dual_spiral import DualSpiralConfig
from satellite.strategy.strategies.dual_raster import DualRasterConfig
from satellite.strategy.strategies.concentric_shells import ConcentricShellsConfig


def base_config(
    *,
    s1_theta: float = 0.001,
    s1_phi: float = 0.001,
    s2_theta: float = 0.001,
    s2_phi: float = 0.001,
    chain: tuple[str, ...] = ("minor_offset", "single_miss"),
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
            max_beam_speed=0.087,
            max_fsm_speed=1.0,
            max_fsm_radius=0.001,
            beam_width_mrad=5.0,
            k=10.0,
        ),
        simulation=SimulationConfig(
            distance=1000.0,
            t_step=0.01,
            beam_length=None,
            boresight_extension=5.0,
            max_search_radius=0.07,
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
            params={
                "minor_offset": MinorOffsetConfig(
                    max_spiral_radius="fov",
                    spiral_speed=0.4,
                ),
                "single_miss": SingleMissConfig(
                    a_spiral_radius=0.05,
                    b_spiral_radius=0.05,
                    spiral_speed=0.16,
                ),
                "asymmetric_swap": AsymmetricSwapConfig(
                    spiral_radius=0.05,
                    lock_duration=1.0,
                    spiral_speed=0.16,
                ),
                "dual_spiral": DualSpiralConfig(
                    speed_a=0.05,
                    speed_ratio=1.41421356,
                ),
                "dual_raster": DualRasterConfig(
                    steps_a=20,
                    steps_b=20,
                    speed_a=0.1,
                    speed_ratio=1.41421356,
                ),
                "concentric_shells": ConcentricShellsConfig(
                    radii_factors=(0.2, 0.5, 1.0),
                    spiral_speed_a=0.05,
                    speed_ratio=1.41421356,
                ),

            },
        ),
    )
