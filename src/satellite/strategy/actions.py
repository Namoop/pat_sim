"""Timed epoch action scripts."""

from __future__ import annotations

from dataclasses import dataclass

from satellite.strategy.movements import MovementPattern


@dataclass(frozen=True)
class SatelliteAction:
    movement: MovementPattern


@dataclass(frozen=True)
class Epoch:
    duration: float
    s1: SatelliteAction
    s2: SatelliteAction
    label: str = ""


@dataclass(frozen=True)
class ActionScript:
    epochs: tuple[Epoch, ...]
    strategy_name: str = ""

    @property
    def total_duration(self) -> float:
        return sum(epoch.duration for epoch in self.epochs)
