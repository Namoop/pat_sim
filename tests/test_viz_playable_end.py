# -*- coding: utf-8 -*-
"""Playable timeline end and lock-capped replay tests."""

from __future__ import annotations

import math

import pytest

from scenario.run import run_scenario
from visualize.app import clamp_playable_t, play_reaches_end
from tests.conftest import base_config


def test_playable_t_end_on_success():
    cfg = base_config()
    result = run_scenario(cfg)
    assert result.hit_at_t is not None
    assert result.playable_t_end == result.hit_at_t
    assert result.playable_t_end < result.schedule.total_duration


def test_playable_t_end_on_failure():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=-0.08,
        s2_phi=-0.08,
        chain=("minor_offset",),
    )
    result = run_scenario(cfg)
    assert result.hit_at_t is None
    assert result.playable_t_end == result.schedule.total_duration


def test_replay_timeline_stops_at_lock():
    cfg = base_config()
    result = run_scenario(cfg)
    t_step = cfg.simulation.t_step
    timeline = result.ensure_replay_timeline()
    assert timeline.t_end == pytest.approx(result.playable_t_end, abs=cfg.simulation.t_step)
    full_steps = int(math.floor(result.schedule.total_duration / t_step)) + 1
    assert timeline.step_count < full_steps


def test_replay_timeline_runs_full_schedule_on_failure():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=-0.08,
        s2_phi=-0.08,
        chain=("minor_offset",),
    )
    result = run_scenario(cfg)
    timeline = result.ensure_replay_timeline()
    assert timeline.t_end == pytest.approx(result.playable_t_end, abs=cfg.simulation.t_step)
    assert timeline.t_end == pytest.approx(result.schedule.total_duration, abs=cfg.simulation.t_step)


def test_replay_to_clamps_past_playable_end():
    cfg = base_config()
    result = run_scenario(cfg)
    playable = result.playable_t_end
    result.ensure_replay_timeline()
    result.replay_to(playable + 10.0)
    assert result.mutual_lock(playable)


def test_clamp_playable_t():
    assert clamp_playable_t(3.0, 2.0) == 2.0
    assert clamp_playable_t(-1.0, 2.0) == 0.0
    assert clamp_playable_t(1.5, 2.0) == 1.5


def test_play_reaches_end():
    assert play_reaches_end(2.01, 2.0) is True
    assert play_reaches_end(2.0, 2.0) is False
    assert play_reaches_end(1.99, 2.0) is False
