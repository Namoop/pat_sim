# -*- coding: utf-8 -*-
"""Playable timeline end and lock-capped replay tests."""

from __future__ import annotations

import math

import pytest

from satellite.scenario import run_scenario
from satellite.visualize.app import clamp_playable_q, play_reaches_end
from tests.conftest import base_config


def test_playable_q_end_on_success():
    cfg = base_config(step_duration=5.0)
    result = run_scenario(cfg)
    assert result.hit_at_q is not None
    assert result.playable_q_end == result.hit_at_q
    assert result.playable_q_end < result.schedule.total_duration


def test_playable_q_end_on_failure():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=-0.08,
        s2_phi=-0.08,
        step_duration=0.5,
        chain=("minor_offset",),
    )
    result = run_scenario(cfg)
    assert result.hit_at_q is None
    assert result.playable_q_end == result.schedule.total_duration


def test_replay_timeline_stops_at_lock():
    cfg = base_config(step_duration=5.0)
    result = run_scenario(cfg)
    q_step = cfg.simulation.q_step
    timeline = result.ensure_replay_timeline()
    assert timeline.q_end == pytest.approx(result.playable_q_end)
    full_steps = int(math.floor(result.schedule.total_duration / q_step)) + 1
    assert timeline.step_count < full_steps


def test_replay_timeline_runs_full_schedule_on_failure():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=-0.08,
        s2_phi=-0.08,
        step_duration=0.5,
        chain=("minor_offset",),
    )
    result = run_scenario(cfg)
    timeline = result.ensure_replay_timeline()
    assert timeline.q_end == pytest.approx(result.playable_q_end)
    assert timeline.q_end == pytest.approx(result.schedule.total_duration)


def test_replay_to_clamps_past_playable_end():
    cfg = base_config(step_duration=5.0)
    result = run_scenario(cfg)
    playable = result.playable_q_end
    result.ensure_replay_timeline()
    result.replay_to(playable + 10.0)
    assert result.mutual_lock(playable)


def test_clamp_playable_q():
    assert clamp_playable_q(3.0, 2.0) == 2.0
    assert clamp_playable_q(-1.0, 2.0) == 0.0
    assert clamp_playable_q(1.5, 2.0) == 1.5


def test_play_reaches_end():
    assert play_reaches_end(2.01, 2.0) is True
    assert play_reaches_end(2.0, 2.0) is False
    assert play_reaches_end(1.99, 2.0) is False
