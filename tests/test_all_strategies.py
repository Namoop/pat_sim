# -*- coding: utf-8 -*-
"""Integration tests for all built-in search strategies."""

from __future__ import annotations

import pytest

from satellite.scenario import run_scenario
from tests.conftest import base_config


@pytest.mark.parametrize(
    "strategy_name",
    [
        "minor_offset",
        "single_miss",
        "asymmetric_swap",
        "dual_spiral",
        "dual_raster",
        "hex_scan",
        "lissajous_scan",
        "rosette_scan",
        "center_rebias",
        "concentric_shells",
        "random_walk",
        "random_curve",
        "nested_spiral",
        "golden_angle_spiral",
    ],
)
def test_strategy_execution_does_not_crash(strategy_name):
    # Use very small offsets so it succeeds quickly or executes correctly
    cfg = base_config(
        s1_theta=0.001,
        s1_phi=0.001,
        s2_theta=0.001,
        s2_phi=0.001,
        chain=(strategy_name,),
    )
    result = run_scenario(cfg)
    # The simulation should complete (success or failure, but not crash/exception)
    assert result is not None
    assert hasattr(result, "success")
