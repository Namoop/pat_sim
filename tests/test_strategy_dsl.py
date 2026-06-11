# -*- coding: utf-8 -*-
"""DSL and script validation tests."""

from __future__ import annotations

import pytest

from satellite.sda.satellite import Satellite
from satellite.strategy.actions import (
    beam,
    hold,
    receiver,
    reset,
    spiral,
    strategy,
    validate_movement_durations,
)
from satellite.strategy.base import StrategyContext
from tests.conftest import base_config


def _ctx() -> StrategyContext:
    cfg = base_config()
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    return StrategyContext(s1=s1, s2=s2, config=cfg)


def test_spiral_inside_satellite_context():
    script = strategy("test")
    with script.satellite("S1"):
        spiral(duration=1.0, w=1.0, k=10.0, max_radius=0.01)
    with script.satellite("S2"):
        hold(duration=1.0)
    built = script.build()
    assert built.s1.total_duration == pytest.approx(1.0)
    assert built.s2.total_duration == pytest.approx(1.0)


def test_dsl_outside_context_raises():
    with pytest.raises(RuntimeError, match="inside"):
        hold(duration=1.0)


def test_unequal_durations_raise():
    script = strategy("test")
    with script.satellite("S1"):
        hold(duration=1.0)
    with script.satellite("S2"):
        hold(duration=2.0)
    with pytest.raises(ValueError, match="same total duration"):
        script.build()


def test_negative_duration_raises():
    script = strategy("test")
    with script.satellite("S1"):
        with pytest.raises(ValueError, match="non-negative"):
            hold(duration=-1.0)


def test_zero_hold_duration_raises():
    script = strategy("test")
    with script.satellite("S1"):
        with pytest.raises(ValueError, match="positive"):
            hold(duration=0.0)


def test_reset_zero_duration_errors_when_not_at_target():
    ctx = _ctx()
    script = strategy("test")
    with script.satellite("S1"):
        spiral(duration=0.5, w=1.0, k=10.0, max_radius=0.05)
        reset(duration=0.0)
        hold(duration=0.5)
    with script.satellite("S2"):
        hold(duration=1.0)
    built = script.build()
    with pytest.raises(ValueError, match="not at initial boresight"):
        validate_movement_durations(ctx, built)


def test_reset_zero_duration_ok_when_at_target():
    ctx = _ctx()
    ctx.s1.bench.set_bench_aim(ctx.s1.bench.initial_boresight.copy())
    script = strategy("test")
    with script.satellite("S1"):
        reset(duration=0.0)
        hold(duration=1.0)
    with script.satellite("S2"):
        hold(duration=1.0)
    built = script.build()
    validate_movement_durations(ctx, built)


def test_same_time_hardware_toggles():
    script = strategy("test")
    with script.satellite("S1"):
        beam.enable()
        receiver.disable()
        hold(duration=1.0)
    with script.satellite("S2"):
        hold(duration=1.0)
    built = script.build()
    assert len(built.s1.hardware_steps) == 2
    assert built.s1.hardware_steps[0].time == built.s1.hardware_steps[1].time


def test_validate_slew_speed_invoked_from_try_run():
    from satellite.strategy.strategies.minor_offset import MinorOffsetStrategy

    ctx = _ctx()
    strat = MinorOffsetStrategy(
        duration=1.0,
        max_spiral_radius=0.02,
        spiral_speed=1.0,
        w=10.0,
        k=10.0,
    )
    result = strat.try_run(ctx, global_q_start=0.0)
    assert result.elapsed_q == pytest.approx(1.0)
