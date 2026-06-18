# -*- coding: utf-8 -*-
"""Eye view scene gating tests."""

from __future__ import annotations

from scenario.run import run_scenario
from visualize.scene import build_view
from tests.conftest import base_config


def test_eye_view_hides_beam_and_fov_when_disabled():
    cfg = base_config(chain=("asymmetric_swap",))
    result = run_scenario(cfg)
    s1 = build_view(result, "S1", 0.0)
    s2 = build_view(result, "S2", 0.0)
    assert s1.beam is not None
    assert s1.fov is None
    assert s2.beam is None
    assert s2.fov is not None
