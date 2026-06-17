# -*- coding: utf-8 -*-
"""Asymmetric swap strategy tests."""

from __future__ import annotations

from satellite.scenario import run_scenario
from tests.conftest import base_config


def test_asymmetric_swap_succeeds_with_small_offsets():
    cfg = base_config(
        s1_theta=0.001,
        s1_phi=0.001,
        s2_theta=0.001,
        s2_phi=0.001,
        chain=("asymmetric_swap",),
    )
    result = run_scenario(cfg)
    assert result.success
    assert result.strategy_name == "asymmetric_swap"
    assert result.hit_at_t is not None
    hit_12, hit_21 = result.bidirectional_lock(result.hit_at_t)
    assert hit_12 and hit_21


def test_asymmetric_swap_script_includes_reset_after_spiral():
    from satellite.sda.satellite import Satellite
    from satellite.strategy.base import StrategyContext
    from satellite.strategy.strategies.asymmetric_swap import AsymmetricSwapStrategy, AsymmetricSwapConfig

    cfg = base_config(chain=("asymmetric_swap",))
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    ctx = StrategyContext(s1=s1, s2=s2, config=cfg)
    script = AsymmetricSwapStrategy(
        config=AsymmetricSwapConfig(
            spiral_radius=0.05,
            lock_duration=1.0,
        ),
        w=10.0,
        k=10.0,
    ).build_script(ctx)
    s1_labels = [step.label for step in script.s1.movement_steps]
    assert "S1 probe spiral" in s1_labels
    assert "S1 reset" in s1_labels
    assert s1_labels.index("S1 reset") == s1_labels.index("S1 probe spiral") + 1


def test_asymmetric_swap_fails_when_b_never_acquires():
    cfg = base_config(
        s1_theta=0.08,
        s1_phi=0.08,
        s2_theta=0.08,
        s2_phi=0.08,
        chain=("asymmetric_swap",),
    )
    result = run_scenario(cfg)
    assert not result.success
    assert result.hit_at_t is None
