# -*- coding: utf-8 -*-
"""FrameRunner behavior tests."""

from __future__ import annotations

from satellite.sda.satellite import Satellite
from satellite.strategy.actions import beam, hold, receiver, strategy
from satellite.strategy.base import StrategyContext
from satellite.strategy.runner import FrameRunner
from tests.conftest import base_config


def _ctx(
    *,
    s1_theta: float = 0.001,
    s1_phi: float = 0.001,
    s2_theta: float = 0.001,
    s2_phi: float = 0.001,
) -> StrategyContext:
    cfg = base_config(s1_theta=s1_theta, s1_phi=s1_phi, s2_theta=s2_theta, s2_phi=s2_phi)
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    return StrategyContext(s1=s1, s2=s2, config=cfg)


def test_one_way_visibility_does_not_lock():
    ctx = _ctx(s2_theta=0.05, s2_phi=0.05)
    script = strategy("one_way")
    with script.satellite("S1"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    with script.satellite("S2"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    built = script.build()
    result = FrameRunner(ctx).execute(built, stop_on_lock=True)
    assert not result.success


def test_mutual_lock_requires_both_slews_complete():
    ctx = _ctx()
    script = strategy("mutual")
    with script.satellite("S1"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    with script.satellite("S2"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    built = script.build()
    result = FrameRunner(ctx).execute(built, stop_on_lock=True)
    assert result.success
    assert result.hit_at_q is not None


def test_disabled_receiver_blocks_lock():
    ctx = _ctx()
    script = strategy("rx_off")
    with script.satellite("S1"):
        beam.enable()
        receiver.disable()
        hold(duration=2.0)
    with script.satellite("S2"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    built = script.build()
    result = FrameRunner(ctx).execute(built, stop_on_lock=True)
    assert not result.success


def test_disabled_beam_blocks_lock():
    ctx = _ctx()
    script = strategy("tx_off")
    with script.satellite("S1"):
        beam.disable()
        receiver.enable()
        hold(duration=2.0)
    with script.satellite("S2"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    built = script.build()
    result = FrameRunner(ctx).execute(built, stop_on_lock=True)
    assert not result.success


def test_later_hardware_toggle_applies():
    ctx = _ctx()
    script = strategy("toggle")
    with script.satellite("S1"):
        beam.enable()
        receiver.enable()
        hold(duration=0.5)
        beam.disable()
        hold(duration=1.5)
    with script.satellite("S2"):
        beam.enable()
        receiver.enable()
        hold(duration=2.0)
    built = script.build()
    runtime = FrameRunner(ctx).begin(built)
    result_mid = FrameRunner(ctx).step(runtime, 0.25, ctx.q_step)
    assert any("beam enabled" in e for e in result_mid.events)
    result_late = FrameRunner(ctx).step(runtime, 0.75, ctx.q_step)
    assert any("beam disabled" in e for e in result_late.events)
