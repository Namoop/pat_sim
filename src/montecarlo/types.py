"""Monte Carlo batch runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from strategy.config import StrategyConfig


@dataclass(frozen=True)
class UniformErrorConfig:
    distribution: str
    min: float
    max: float


@dataclass(frozen=True)
class GaussianErrorConfig:
    distribution: str
    mean: float
    std: float


ErrorDistributionConfig = UniformErrorConfig | GaussianErrorConfig


@dataclass(frozen=True)
class MonteCarloConfig:
    simulation_path: Path
    seed: int
    error: ErrorDistributionConfig
    strategy: StrategyConfig
    runs: int
    chain: tuple[str, ...]
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
