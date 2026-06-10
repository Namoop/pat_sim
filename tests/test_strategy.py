"""Strategy layer tests."""

from __future__ import annotations

import pytest

from satellite.config import (
    MapVisualizationConfig,
    MinorOffsetStrategyConfig,
    SatelliteInstanceConfig,
    ScenarioConfig,
    SdaConfig,
    SharedSatelliteConfig,
    SimulationConfig,
    SingleMissStrategyConfig,
    StrategyConfig,
    VisualizationConfig,
)
from satellite.detection import beam_hits_dish
from satellite.math3d import as_vec3
from satellite.scenario import run_scenario
from satellite.strategy.base import link_established
from satellite.strategy.schedule import LegSchedule
from satellite.sda.satellite import Satellite


def _base_config(
    *,
    s1_theta: float = 0.001,
    s1_phi: float = 0.001,
    s2_theta: float = 0.001,
    s2_phi: float = 0.001,
    q_max: float = 5.0,
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
            position=as_vec3([0.0, 50.0, 1000.0]),
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
        sda=SdaConfig(k=10.0, gamma=0.5, beta=0.5, omega_r=10.0, L_r=0.15),
        simulation=SimulationConfig(
            q_max=q_max,
            q_step=0.01,
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
            chain=chain,
            minor_offset=MinorOffsetStrategyConfig(
                duration=q_max,
                max_spiral_radius="fov",
                spiral_speed=1.0,
            ),
            single_miss=SingleMissStrategyConfig(
                epoch1_duration=q_max,
                a_spiral_radius=0.05,
                reset_duration=0.0,
                epoch2_duration=q_max,
                b_spiral_radius=0.05,
                spiral_speed=1.0,
            ),
        ),
    )


def test_link_established_matches_beam_hits_dish():
    cfg = _base_config()
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    aim = s1.bench.bench_boresight.copy()
    direct = beam_hits_dish(
        s1.position,
        s2.receiver.dish_mount,
        s2.receiver.dish_boresight,
        s2.receiver.dish_fov,
        aim,
        cfg.satellite.alpha,
        s1.transmitter.beam_length,
    )
    assert link_established(s1, s2, aim, cfg) == direct


def test_minor_offset_succeeds_with_small_offsets():
    cfg = _base_config()
    result = run_scenario(cfg)
    assert result.success
    assert result.strategy_name == "minor_offset"
    assert result.hit_at_q is not None


def test_escalation_to_single_miss_with_large_offsets():
    cfg = _base_config(s1_theta=0.02, s1_phi=0.015, s2_theta=0.02, s2_phi=0.01)
    result = run_scenario(cfg)
    assert result.success
    assert result.strategy_name == "single_miss"
    assert len(result.meta.attempts) >= 2
    assert not result.meta.attempts[0].success


def test_total_failure_when_strategy_times_out():
    cfg = _base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=0.08,
        s2_phi=0.08,
        q_max=0.5,
        chain=("minor_offset",),
    )
    cfg = ScenarioConfig(
        name=cfg.name,
        s1=cfg.s1,
        s2=cfg.s2,
        satellite=cfg.satellite,
        sda=cfg.sda,
        simulation=SimulationConfig(
            q_max=0.5,
            q_step=0.01,
            beam_length=None,
            boresight_extension=5.0,
            profile_replay=False,
        ),
        visualization=cfg.visualization,
        map_visualization=cfg.map_visualization,
        strategy=StrategyConfig(
            chain=("minor_offset",),
            minor_offset=MinorOffsetStrategyConfig(
                duration=0.5,
                max_spiral_radius="fov",
                spiral_speed=1.0,
            ),
            single_miss=cfg.strategy.single_miss,
        ),
    )
    result = run_scenario(cfg)
    assert not result.success


def test_leg_schedule_epoch_at_boundaries():
    cfg = _base_config()
    result = run_scenario(cfg)
    schedule: LegSchedule = result.schedule
    assert schedule.total_duration > 0.0
    epoch0, local0 = schedule.epoch_at(0.0)
    assert local0 == pytest.approx(0.0)
    mid = epoch0.q_start + epoch0.duration * 0.5
    _, local_mid = schedule.epoch_at(mid)
    assert local_mid == pytest.approx(epoch0.duration * 0.5, abs=0.02)


def test_replay_matches_headless_hit_at_q():
    cfg = _base_config(s1_theta=0.02, s1_phi=0.015, s2_theta=0.02, s2_phi=0.01)
    result = run_scenario(cfg)
    assert result.hit_at_q is not None
    result.replay_to(result.hit_at_q)
    assert result.active_in_cone(result.hit_at_q)
