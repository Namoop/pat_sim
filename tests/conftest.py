# -*- coding: utf-8 -*-
"""Shared scenario fixtures for strategy tests."""

from __future__ import annotations

from satellite.config import (
    EyeVizConfig,
    MagVizConfig,
    SimulationConfig,
    SharedSatelliteConfig,
    ThreeDVizConfig,
)
from scenario.types import (
    SatelliteInstanceConfig,
    ScenarioConfig,
)
from strategy.config import StrategyConfig
from satellite.math.math3d import as_vec3
from strategy.strategies.asymmetric_swap import AsymmetricSwapConfig
from strategy.strategies.minor_offset import MinorOffsetConfig
from strategy.strategies.single_miss import SingleMissConfig
from strategy.strategies.dual_spiral import DualSpiralConfig
from strategy.strategies.dual_raster import DualRasterConfig
from strategy.strategies.concentric_shells import ConcentricShellsConfig


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
        three_d_viz=ThreeDVizConfig(
            cone_u_steps=8,
            cone_v_steps=8,
            spiral_trail_steps=20,
            ribbon_v_steps=4,
            profile_frames=False,
        ),
        eye_viz=EyeVizConfig(
            axis_limit=0.1,
            profile_frames=False,
            slider_debounce_ms=16,
            hover_correlation_enabled=True,
            correlation_blob_alpha=100,
            heatmap_grid_resolution=96,
            heatmap_diversity_floor=0.15,
        ),
        mag_viz=MagVizConfig(
            visual_limit_deg=1.0,
            fov_cone_length=1.0,
            beam_cone_length=1.0,
        ),
        strategy=StrategyConfig(
            k=10.0,
            chain=chain,
            params={
                "minor_offset": MinorOffsetConfig(
                    max_spiral_radius="fov",
                ),
                "single_miss": SingleMissConfig(
                    a_spiral_radius=0.05,
                    b_spiral_radius=0.05,
                ),
                "asymmetric_swap": AsymmetricSwapConfig(
                    spiral_radius=0.05,
                    lock_duration=1.0,
                ),
                "dual_spiral": DualSpiralConfig(),
                "dual_raster": DualRasterConfig(
                    steps_a=20,
                    steps_b=20,
                    speed_a=0.1,
                    speed_ratio=1.41421356,
                ),
                "concentric_shells": ConcentricShellsConfig(
                    num_shells=3,
                    growth_exponent=1.0,
                    s2_offset_shells=0,
                ),

            },
        ),
    )
