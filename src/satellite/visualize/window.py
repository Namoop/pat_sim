"""PyVista + Qt visualization (legacy entry — use satellite.visualize.run_visualizer)."""

from __future__ import annotations

from satellite.scenario import ScenarioResult
from satellite.visualize.app import run_visualizer as _run_unified
from satellite.visualize.session import SingleResultSession


def run_visualizer(result: ScenarioResult, start_t: float = 0.0) -> None:
    """Open unified visualizer on the 3D tab."""
    _run_unified(
        SingleResultSession(result),
        default_tab="3d",
        start_t=start_t,
    )
