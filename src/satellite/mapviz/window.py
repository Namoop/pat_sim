"""Angular map visualization (legacy entry — use satellite.visualize.run_visualizer)."""

from __future__ import annotations

from satellite.scenario import ScenarioResult
from satellite.visualize.app import run_visualizer as _run_unified
from satellite.visualize.session import SingleResultSession


def run_map_visualizer(result: ScenarioResult, start_t: float = 0.0) -> None:
    """Open unified visualizer on the map tab."""
    _run_unified(
        SingleResultSession(result),
        default_tab="map",
        start_t=start_t,
    )
