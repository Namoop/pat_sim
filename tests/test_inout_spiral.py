# -*- coding: utf-8 -*-
"""InOutSpiral movement and dual_spiral timeline packing tests."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from satellite.physics.satellite import Satellite
from strategy.base import StrategyContext
from strategy.movements import Hold, InOutSpiral, Spiral, build_aim_context
from strategy.strategies.dual_spiral import DualSpiralConfig, DualSpiralStrategy
from tests.conftest import base_config


def _ctx(*, timeout: float = 100.0, radius: float = 0.001) -> StrategyContext:
    cfg = base_config(chain=("dual_spiral",))
    cfg = replace(
        cfg,
        simulation=replace(cfg.simulation, timeout=timeout, max_search_radius=radius),
        strategy=replace(
            cfg.strategy,
            chain=("dual_spiral",),
            params={
                **cfg.strategy.params,
                "dual_spiral": DualSpiralConfig(radius=radius),
            },
        ),
    )
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    return StrategyContext(s1=s1, s2=s2, config=cfg)


def test_dual_spiral_emits_few_legs_at_small_radius():
    ctx = _ctx(timeout=100.0, radius=0.001)
    script = DualSpiralStrategy.from_config(ctx.config).build_script(ctx)

    assert len(script.s1.movement_steps) <= 2
    assert len(script.s2.movement_steps) <= 3
    assert sum(isinstance(s.movement, InOutSpiral) for s in script.s1.movement_steps) == 1
    assert sum(isinstance(s.movement, InOutSpiral) for s in script.s2.movement_steps) == 1
    assert sum(isinstance(s.movement, Spiral) for s in script.s1.movement_steps) == 0
    assert sum(isinstance(s.movement, Spiral) for s in script.s2.movement_steps) == 0


def test_inout_spiral_matches_unrolled_spiral_sequence():
    ctx = _ctx(timeout=20.0, radius=0.002)
    cfg = ctx.config
    w = cfg.strategy.spiral_w(cfg.satellite)
    k = cfg.strategy.k
    max_radius = 0.002
    one_way = cfg.strategy.spiral_duration(max_radius, w, cfg.satellite.max_beam_speed)
    cycle = 2.0 * one_way
    assert cycle > 0.0

    n_cycles = 3
    total = n_cycles * cycle
    aim_ctx = build_aim_context(ctx.s1)
    combined = InOutSpiral(
        w=w, k=k, max_radius=max_radius, one_way_duration=one_way
    )

    # Unrolled reference: out then in, repeated.
    out = Spiral(w=w, k=k, max_radius=max_radius)
    inn = Spiral(w=w, k=k, max_radius=0.0)

    sample_ts = np.linspace(0.0, total, 40, endpoint=False)
    for t in sample_ts:
        got = combined.aim_at(float(t), total, aim_ctx)

        phase = float(t % cycle)
        if phase < one_way:
            expected = out.aim_at(phase, one_way, aim_ctx)
        else:
            # Reverse spiral uses step_start_aim at outer radius; synthesize that.
            from dataclasses import replace as dc_replace

            outer_aim = out.aim_at(one_way, one_way, aim_ctx)
            rev_ctx = dc_replace(aim_ctx, step_start_aim=outer_aim)
            expected = inn.aim_at(phase - one_way, one_way, rev_ctx)

        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_dual_spiral_hold_remainder_when_incomplete_cycle():
    ctx = _ctx(timeout=100.0, radius=0.001)
    script = DualSpiralStrategy.from_config(ctx.config).build_script(ctx)
    s1_steps = script.s1.movement_steps
    assert isinstance(s1_steps[-1].movement, Hold) or (
        isinstance(s1_steps[0].movement, InOutSpiral)
        and abs(s1_steps[0].end - 100.0) < 1e-9
    )
    if len(s1_steps) == 2:
        assert isinstance(s1_steps[0].movement, InOutSpiral)
        assert isinstance(s1_steps[1].movement, Hold)
        assert s1_steps[1].end == pytest.approx(100.0)
