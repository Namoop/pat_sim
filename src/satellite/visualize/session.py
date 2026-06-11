"""Visualization session — single scenario or lazy Monte Carlo stepping."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from satellite.config import MonteCarloConfig, load_simulation_config
from satellite.monte_carlo import run_monte_carlo_single
from satellite.scenario import ScenarioResult


class VizSession(Protocol):
    def current(self) -> ScenarioResult: ...

    def has_next(self) -> bool: ...

    def advance(self) -> ScenarioResult | None: ...

    def status_label(self) -> str: ...


class SingleResultSession:
    """One scenario; Next closes the window."""

    def __init__(self, result: ScenarioResult) -> None:
        self._result = result

    def current(self) -> ScenarioResult:
        return self._result

    def has_next(self) -> bool:
        return False

    def advance(self) -> ScenarioResult | None:
        return None

    def status_label(self) -> str:
        return self._result.config.name


class MonteCarloVizSession:
    """Lazy Monte Carlo — one run at a time for interactive visualization."""

    def __init__(self, mc: MonteCarloConfig) -> None:
        self._mc = mc
        self._sim = load_simulation_config(mc.simulation_path)
        self._rng = np.random.default_rng(mc.seed)
        self._run_index = 0
        self._current: ScenarioResult | None = None

    def current(self) -> ScenarioResult:
        if self._current is None:
            run = run_monte_carlo_single(
                self._mc,
                self._sim,
                self._rng,
                self._run_index,
                report=True,
                total_runs=self._mc.runs,
            )
            self._current = run.result
        return self._current

    def has_next(self) -> bool:
        return self._run_index < self._mc.runs - 1

    def advance(self) -> ScenarioResult | None:
        if self._run_index >= self._mc.runs - 1:
            return None
        self._run_index += 1
        run = run_monte_carlo_single(
            self._mc,
            self._sim,
            self._rng,
            self._run_index,
            report=True,
            total_runs=self._mc.runs,
        )
        self._current = run.result
        return self._current

    def status_label(self) -> str:
        name = (
            self._current.config.name
            if self._current is not None
            else f"mc_run_{self._run_index}"
        )
        return f"{name} ({self._run_index + 1}/{self._mc.runs})"
