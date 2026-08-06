"""Optimize TOML parser ([optimize] section)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizeConfig:
    strategy: str | None = None
    method: str = "random"
    trials: int = 20
    trial_seed: int | None = None
    max_success_penalty: float = 1.0


def parse(data: dict) -> OptimizeConfig:
    opt = data.get("optimize", {})
    trial_seed = opt.get("trial_seed")
    return OptimizeConfig(
        strategy=opt.get("strategy"),
        method=str(opt.get("method", "random")),
        trials=int(opt.get("trials", 20)),
        trial_seed=int(trial_seed) if trial_seed is not None else None,
        max_success_penalty=float(opt.get("max_success_penalty", 1.0))
    )
