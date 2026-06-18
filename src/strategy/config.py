"""Strategy configuration types and [strategy] section parser."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from satellite.config import SharedSatelliteConfig


@dataclass(frozen=True)
class StrategyConfig:
    k: float
    chain: tuple[str, ...]
    params: Mapping[str, Any]

    def spiral_w(self, satellite: SharedSatelliteConfig) -> float:
        return self.k * satellite.alpha / math.pi

    def spiral_duration(
        self, radius: float, w: float, max_beam_speed: float
    ) -> float:
        """T = s(R / w) / max_beam_speed where s is the arc length of the spiral path."""
        if w <= 0 or max_beam_speed <= 0:
            return 0.0
        k = self.k
        if k <= 0.0:
            return radius / (w * max_beam_speed)
        x = (k * radius) / w
        sqrt_term = math.sqrt(1.0 + x * x)
        arc_len = (w / (2.0 * k)) * (x * sqrt_term + math.log(x + sqrt_term))
        return arc_len / max_beam_speed

    def reset_duration(
        self, radius: float, max_beam_speed: float
    ) -> float:
        """T_reset = R / max_beam_speed"""
        if max_beam_speed <= 0:
            return 0.0
        return radius / max_beam_speed

    @staticmethod
    def resolve_radius(value: str | float, dish_fov: float) -> float:
        if isinstance(value, str) and value.lower() == "fov":
            return dish_fov
        return float(value)


def parse(
    data: dict,
    *,
    chain: list[str] | tuple[str, ...] | None = None,
    default_k: float = 10.0,
) -> StrategyConfig:
    from strategy.base import CONFIG_PARSERS

    global_chain = data.get("chain")
    if global_chain is None:
        if chain is not None:
            global_chain = chain
        else:
            raise ValueError("strategy.chain is required")

    chain_list = list(global_chain)
    all_strategy_names = set(chain_list)
    if chain is not None:
        all_strategy_names.update(chain)

    params: dict[str, Any] = {}
    for name in all_strategy_names:
        section = data.get(name, {})
        if name in CONFIG_PARSERS:
            params[name] = CONFIG_PARSERS[name](section)
        else:
            params[name] = section

    return StrategyConfig(
        k=float(data.get("k", default_k)),
        chain=tuple(chain_list),
        params=params,
    )
