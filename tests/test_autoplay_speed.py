"""Autoplay speed step scaling."""

from __future__ import annotations

from satellite.visualize.app import play_step_delta


def test_play_step_delta_normal_play():
    assert play_step_delta(0.01, autoplay_active=False, autoplay_speed=2.0) == 0.01


def test_play_step_delta_autoplay_scales():
    assert play_step_delta(0.01, autoplay_active=True, autoplay_speed=2.0) == 0.02
    assert play_step_delta(0.01, autoplay_active=True, autoplay_speed=0.5) == 0.005


def test_play_step_delta_ignores_speed_when_autoplay_inactive():
    assert play_step_delta(0.01, autoplay_active=False, autoplay_speed=10.0) == 0.01
