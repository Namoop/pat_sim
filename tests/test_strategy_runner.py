# -*- coding: utf-8 -*-
"""FrameRunner behavior tests."""

from __future__ import annotations

from satellite.physics.satellite import Satellite
from strategy.actions import beam, hold, receiver, strategy
from strategy.base import StrategyContext
from strategy.runner import FrameRunner
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
    assert result.hit_at_t is not None


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
    result_mid = FrameRunner(ctx).step(runtime, 0.25, ctx.t_step)
    assert any("beam enabled" in e for e in result_mid.events)
    result_late = FrameRunner(ctx).step(runtime, 0.75, ctx.t_step)
    assert any("beam disabled" in e for e in result_late.events)


def test_reset_slew_motion():
    from strategy.actions import reset, spiral, receiver
    from satellite.math.math3d import angle_between
    import math

    ctx = _ctx()
    script = strategy("test_reset")
    with script.satellite("S1"):
        receiver.disable()
        spiral(duration=1.0, w=10.0, k=20.0, max_radius=0.05)
        reset(duration=2.0)
    with script.satellite("S2"):
        hold(duration=3.0)
    built = script.build()

    runtime = FrameRunner(ctx).begin(built)
    
    # Step to the end of the spiral movement (t = 1.0)
    t = 0.0
    while t < 1.0:
        FrameRunner(ctx).step(runtime, t, ctx.t_step)
        t += ctx.t_step

    # Save the boresight at the end of the spiral / start of reset
    boresight_at_start_of_reset = ctx.s1.bench.bench_boresight.copy()
    center = ctx.s1.bench.initial_boresight.copy()

    # The spiral should have moved the boresight away from the initial boresight (center)
    offset_dist = angle_between(boresight_at_start_of_reset, center)
    assert offset_dist > 0.01, f"Boresight offset is too small: {math.degrees(offset_dist):.3f} deg"

    # Step slightly into the reset movement (t = 1.01)
    FrameRunner(ctx).step(runtime, 1.0 + ctx.t_step, ctx.t_step)
    boresight_after_small_step = ctx.s1.bench.bench_boresight.copy()

    # It should not have teleported to center immediately, but remain close to start boresight
    dist_to_center = angle_between(boresight_after_small_step, center)
    dist_to_start = angle_between(boresight_after_small_step, boresight_at_start_of_reset)

    assert dist_to_center > 0.01, f"Boresight jumped directly to center (teleported)! Dist: {math.degrees(dist_to_center):.3f} deg"
    assert dist_to_start < dist_to_center, "Boresight is not closer to starting position than center"


def test_inward_spiral_motion():
    from strategy.actions import hold, spiral, receiver
    from satellite.math.math3d import angle_between

    ctx = _ctx()
    script = strategy("test_inward")
    with script.satellite("S1"):
        receiver.disable()
        # Spiral out to 0.05
        spiral(duration=1.0, w=10.0, k=20.0, max_radius=0.05)
        # Spiral in to 0.0
        spiral(duration=1.0, w=10.0, k=20.0, max_radius=0.0)
    with script.satellite("S2"):
        hold(duration=2.0)
    built = script.build()

    runtime = FrameRunner(ctx).begin(built)
    
    # Step to the end of the outward spiral (t = 1.0)
    t = 0.0
    while t < 1.0:
        FrameRunner(ctx).step(runtime, t, ctx.t_step)
        t += ctx.t_step

    boresight_at_max_radius = ctx.s1.bench.bench_boresight.copy()
    center = ctx.s1.bench.initial_boresight.copy()

    # Verify we are deflected
    start_dist = angle_between(boresight_at_max_radius, center)
    assert start_dist > 0.01

    # Step into the inward spiral (t = 1.5, halfway in)
    t = 1.0
    while t <= 1.5 + 1e-12:
        FrameRunner(ctx).step(runtime, t, ctx.t_step)
        t += ctx.t_step
        
    boresight_halfway = ctx.s1.bench.bench_boresight.copy()
    dist_halfway = angle_between(boresight_halfway, center)
    
    # The radius should be smaller than start_dist but still deflected
    assert dist_halfway < start_dist
    assert dist_halfway > 0.005

    # Step to the end of the inward spiral (t = 2.0)
    while t <= 2.0 + 1e-12:
        FrameRunner(ctx).step(runtime, t, ctx.t_step)
        t += ctx.t_step
        
    boresight_end = ctx.s1.bench.bench_boresight.copy()
    end_dist = angle_between(boresight_end, center)
    assert end_dist < 1e-4

