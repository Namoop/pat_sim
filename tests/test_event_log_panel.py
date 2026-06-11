# -*- coding: utf-8 -*-
"""Event log panel splitting tests."""

from __future__ import annotations

from satellite.visualize.panels.event_log_panel import split_event_log


def test_split_event_log_partitions_by_satellite():
    lines = [
        "Initial conditions:",
        "  distance = 1 km",
        "  S1 bench theta=1 mrad phi=1 mrad",
        "minor_offset: timeline started",
        "S1 step: S1 FOV spiral at q=0.000",
        "S2 step: S2 hold at q=0.000",
        "S1 beam enabled at q=0.000",
        "S2 acquisition started at q=0.500",
        "Lock (both) at q=1.000",
    ]
    system, s1, s2 = split_event_log(lines)
    assert "Initial conditions:" in system
    assert "minor_offset: timeline started" in system
    assert "Lock (both) at q=1.000" in system
    assert any("S1 step" in line for line in s1)
    assert any("S1 beam" in line for line in s1)
    assert any("S2 step" in line for line in s2)
    assert any("S2 acquisition" in line for line in s2)
    assert not any(line.startswith("S2 ") for line in s1)
    assert not any(line.startswith("S1 ") for line in s2)
