"""Monte Carlo TOML parser ([monte_carlo] and error distribution)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    collect_overrides,
    resolve_environment_path,
)
from montecarlo.types import (
    ErrorDistributionConfig,
    GaussianErrorConfig,
    UniformErrorConfig,
)

_MC_SKIP = frozenset({"runs", "seed", "chain", "environment", "error"})


@dataclass(frozen=True)
class MonteCarloSettings:
    simulation_path: Path
    seed: int
    error: ErrorDistributionConfig
    runs: int
    chain: tuple[str, ...]
    overrides: dict[str, dict[str, Any]]


def _require_float(section: dict, key: str, context: str) -> float:
    if key not in section:
        raise ValueError(f"{context}.{key} is required")
    return float(section[key])


def _load_error(error_data: dict) -> ErrorDistributionConfig:
    distribution = str(error_data.get("distribution", "uniform")).lower()
    if distribution == "uniform":
        uniform_data = error_data.get("uniform", {})
        return UniformErrorConfig(
            distribution="uniform",
            min=float(uniform_data.get("min", 0.0)) * 1e-3,
            max=_require_float(uniform_data, "max", "[monte_carlo.error.uniform]") * 1e-3,
        )
    if distribution == "gaussian":
        gaussian_data = error_data.get("gaussian", {})
        return GaussianErrorConfig(
            distribution="gaussian",
            mean=float(gaussian_data.get("mean", 0.0)) * 1e-3,
            std=_require_float(gaussian_data, "std", "[monte_carlo.error.gaussian]") * 1e-3,
        )
    raise ValueError(f"Unsupported [monte_carlo.error].distribution: {distribution!r}")


def parse(data: dict, *, path: str | Path) -> MonteCarloSettings:
    config_path = Path(path)
    mc = data.get("monte_carlo", {})

    simulation_rel = mc.get("environment", "Environment.toml")
    simulation_path = resolve_environment_path(str(simulation_rel), config_path)

    chain_list = list(mc.get("chain", []))
    overrides = collect_overrides(mc, skip_keys=_MC_SKIP)

    return MonteCarloSettings(
        simulation_path=simulation_path,
        seed=int(mc.get("seed", 0)),
        error=_load_error(mc.get("error", {})),
        runs=int(mc.get("runs", 1)),
        chain=tuple(chain_list),
        overrides=overrides,
    )
