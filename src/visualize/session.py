"""Visualization session — single scenario or lazy Monte Carlo stepping."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from montecarlo.types import MonteCarloConfig
from satellite.config import load_simulation_config
from strategy.config import StrategyConfig
from montecarlo.run import run_monte_carlo_single
from scenario.run import ScenarioResult


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
        
        self._runs_metadata = []
        chain_rng = np.random.default_rng(mc.seed)
        seeds = chain_rng.integers(0, 2**32 - 1, size=mc.runs).tolist()
        chain_strategy = StrategyConfig(
            k=mc.strategy.k,
            chain=mc.chain,
            params=mc.strategy.params,
        )
        for s in seeds:
            self._runs_metadata.append((s, chain_strategy))

        self._run_index = 0
        self._current: ScenarioResult | None = None

    def current(self) -> ScenarioResult:
        if self._current is None:
            seed, strategy = self._runs_metadata[self._run_index]
            run = run_monte_carlo_single(
                self._mc,
                self._sim,
                seed,
                self._run_index,
                report=False,
                strategy=strategy,
            )
            self._current = run.result
        return self._current

    def has_next(self) -> bool:
        return self._run_index < len(self._runs_metadata) - 1

    def advance(self) -> ScenarioResult | None:
        if self._run_index >= len(self._runs_metadata) - 1:
            return None
        self._run_index += 1
        seed, strategy = self._runs_metadata[self._run_index]
        run = run_monte_carlo_single(
            self._mc,
            self._sim,
            seed,
            self._run_index,
            report=False,
            strategy=strategy,
        )
        self._current = run.result
        return self._current

    def status_label(self) -> str:
        name = (
            self._current.config.name
            if self._current is not None
            else f"mc_run_{self._run_index}"
        )
        return f"{name} ({self._run_index + 1}/{len(self._runs_metadata)})"
