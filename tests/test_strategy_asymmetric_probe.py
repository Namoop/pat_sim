# -*- coding: utf-8 -*-
"""Asymmetric probe strategy tests."""

from __future__ import annotations

from satellite.scenario import run_scenario
from tests.conftest import base_config


def test_asymmetric_probe_succeeds_with_small_offsets():
    cfg = base_config(
        s1_theta=0.001,
        s1_phi=0.001,
        s2_theta=0.001,
        s2_phi=0.001,
        chain=("asymmetric_probe",),
        step_duration=1.0,
    )
    result = run_scenario(cfg)
    assert result.success
    assert result.strategy_name == "asymmetric_probe"
    assert result.hit_at_q is not None
    hit_12, hit_21 = result.bidirectional_lock(result.hit_at_q)
    assert hit_12 and hit_21


def test_asymmetric_probe_fails_when_b_never_acquires():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=0.08,
        s2_phi=0.08,
        chain=("asymmetric_probe",),
        step_duration=0.5,
    )
    result = run_scenario(cfg)
    assert not result.success
    assert result.hit_at_q is None
