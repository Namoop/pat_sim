# -*- coding: utf-8 -*-
"""Map visualization hardware gating tests."""

from __future__ import annotations

from satellite.mapviz.scene import build_panel
from satellite.scenario import run_scenario
from tests.conftest import base_config


def test_map_panel_hides_beam_and_fov_when_disabled():
    cfg = base_config(chain=("asymmetric_probe",), step_duration=1.0)
    result = run_scenario(cfg)
    s1 = build_panel(result, "S1", 0.0)
    s2 = build_panel(result, "S2", 0.0)
    assert s1.beam is not None
    assert s1.fov is None
    assert s2.beam is None
    assert s2.fov is not None
