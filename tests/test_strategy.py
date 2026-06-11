# -*- coding: utf-8 -*-
"""Strategy layer integration tests."""

from __future__ import annotations

import pytest

from satellite.detection import beam_hits_dish
from satellite.math3d import as_vec3
from satellite.scenario import run_scenario
from satellite.sda.satellite import Satellite
from satellite.strategy.base import link_established
from tests.conftest import base_config


def test_link_established_matches_beam_hits_dish():
    cfg = base_config()
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
    cfg = base_config()
    result = run_scenario(cfg)
    assert result.success
    assert result.strategy_name == "minor_offset"
    assert result.hit_at_q is not None
    assert result.mutual_lock(result.hit_at_q)


def test_escalation_to_single_miss_with_large_offsets():
    cfg = base_config(s1_theta=0.02, s1_phi=0.015, s2_theta=0.02, s2_phi=0.01)
    result = run_scenario(cfg)
    assert not result.success
    assert len(result.meta.attempts) >= 2
    assert not result.meta.attempts[0].success
    assert not result.meta.attempts[-1].success


def test_total_failure_when_strategy_times_out():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=0.08,
        s2_phi=0.08,
        step_duration=0.5,
        chain=("minor_offset",),
    )
    result = run_scenario(cfg)
    assert not result.success


def test_leg_schedule_step_at_boundaries():
    cfg = base_config()
    result = run_scenario(cfg)
    schedule = result.schedule
    assert schedule.total_duration > 0.0
    step0, local0 = schedule.step_at(0.0)
    assert local0 == pytest.approx(0.0)
    mid = step0.q_start + step0.duration * 0.5
    _, local_mid = schedule.step_at(mid)
    assert local_mid == pytest.approx(step0.duration * 0.5, abs=0.02)


def test_replay_matches_headless_hit_at_q():
    cfg = base_config()
    result = run_scenario(cfg)
    assert result.hit_at_q is not None
    result.replay_to(result.hit_at_q)
    assert result.mutual_lock(result.hit_at_q)


def test_event_log_includes_initial_conditions():
    cfg = base_config(
        s1_theta=0.01,
        s1_phi=-0.02,
        s2_theta=0.03,
        s2_phi=0.04,
    )
    result = run_scenario(cfg)
    result.ensure_replay_timeline()
    log: list[str] = []
    result.replay_to(0.0, event_log=log)
    assert log[0] == "Initial conditions:"
    assert "distance = 1 km" in log[1]
    assert "S1 bench theta=10 mrad" in log[2]
    assert "phi=-20 mrad" in log[2]
    assert "S2 bench theta=30 mrad" in log[3]
    assert "phi=40 mrad" in log[3]


def test_event_log_includes_acquisition_and_lock():
    cfg = base_config()
    result = run_scenario(cfg)
    assert result.hit_at_q is not None
    result.ensure_replay_timeline()
    log: list[str] = []
    result.replay_to(result.hit_at_q, event_log=log)
    joined = "\n".join(log)
    assert "acquisition started" in joined.lower() or "Mutual lock" in joined
    assert "Mutual lock" in joined or "Lock (both)" in joined
