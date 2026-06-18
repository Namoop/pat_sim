"""Strategy 8: Rosette Scan — Flower-like pattern with center-heavy density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.actions import beam, receiver, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class RosetteScanConfig:
    s1_w1: float = 1.0
    s1_w2: float = 11.0
    s2_w1: float = 0.0
    s2_w2: float = 0.0
    duration: float = 10.0


def parse_rosette_scan_config(data: dict) -> RosetteScanConfig:
    return RosetteScanConfig(
        s1_w1=float(data.get("s1_w1", 1.0)),
        s1_w2=float(data.get("s1_w2", 11.0)),
        s2_w1=float(data.get("s2_w1", 0.0)),
        s2_w2=float(data.get("s2_w2", 0.0)),
        duration=float(data.get("duration", 10.0)),
    )


@register_strategy("rosette_scan", parse_rosette_scan_config)
class RosetteScanStrategy(SearchStrategy):
    def __init__(self, config: RosetteScanConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> RosetteScanStrategy:
        return cls(config=config.strategy.params.get("rosette_scan", RosetteScanConfig()))

    def build_script(self, ctx: StrategyContext):
        from strategy.movements import Rosette

        max_radius = ctx.config.simulation.max_search_radius
        duration = ctx.config.simulation.timeout

        script = strategy(self.name)
        
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                Rosette(A=max_radius, w1=self.config.s1_w1, w2=self.config.s1_w2),
                duration=duration,
                label="S1 rosette scan"
            )
            
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                Rosette(A=max_radius, w1=self.config.s2_w1, w2=self.config.s2_w2),
                duration=duration,
                label="S2 rosette scan"
            )
            
        return script.build()
