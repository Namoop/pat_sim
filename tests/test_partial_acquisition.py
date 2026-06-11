# -*- coding: utf-8 -*-
"""Partial acquisition hold and chain escalation tests."""

from __future__ import annotations

from satellite.scenario import run_scenario
from satellite.sda.satellite import Satellite
from satellite.strategy.base import StrategyContext
from tests.conftest import base_config


def test_partial_acquisition_continues_chain_for_naive_satellite():
    """Acquired satellite holds track; the other still runs the next strategy."""
    cfg = base_config(
        chain=("asymmetric_probe", "single_miss"),
        step_duration=2.0,
    )
    result = run_scenario(cfg)
    names = [a.strategy_name for a in result.meta.attempts]
    assert names[0] == "asymmetric_probe"
    assert "single_miss" in names


def test_partial_acquisition_preserves_tracker_state():
    cfg = base_config(
        chain=("asymmetric_probe", "single_miss"),
        step_duration=2.0,
    )
    result = run_scenario(cfg)
    assert result.s2.receiver.has_seen_beam or result.s1.receiver.has_seen_beam


def test_clone_for_next_attempt_resets_only_naive_satellite():
    cfg = base_config(step_duration=1.0)
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    ctx = StrategyContext(s1=s1, s2=s2, config=cfg)
    ctx.s2.receiver.bench.acquisition.has_seen_beam = True
    ctx.s2.receiver.bench.acquisition.slew_complete = True

    nxt = ctx.clone_for_next_attempt(
        end_hardware={
            "s1_beam": True,
            "s1_receiver": True,
            "s2_beam": True,
            "s2_receiver": True,
        },
    )
    assert nxt.s2 is ctx.s2
    assert nxt.s1 is not ctx.s1
    assert nxt.s2_frozen_hardware == (True, True)
    assert nxt.s1_frozen_hardware is None
