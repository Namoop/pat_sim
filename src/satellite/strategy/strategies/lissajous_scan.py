"""Strategy 7: Lissajous Scan — Sinusoidal orthogonal scanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.sim.config import ScenarioConfig


@dataclass(frozen=True)
class LissajousScanConfig:
    s1_wx: float = 1.0
    s1_wy: float = 1.41421356
    s1_delta: float = 0.0
    s2_wx: float = 1.0
    s2_wy: float = 1.41421356
    s2_delta: float = 0.0


def parse_lissajous_scan_config(data: dict) -> LissajousScanConfig:
    return LissajousScanConfig(
        s1_wx=float(data.get("s1_wx", data.get("wx", 1.0))),
        s1_wy=float(data.get("s1_wy", data.get("wy", 1.41421356))),
        s1_delta=float(data.get("s1_delta", data.get("delta", 0.0))),
        s2_wx=float(data.get("s2_wx", 1.0)),
        s2_wy=float(data.get("s2_wy", 1.41421356)),
        s2_delta=float(data.get("s2_delta", 0.0)),
    )


@register_strategy("lissajous_scan", parse_lissajous_scan_config)
class LissajousScanStrategy(SearchStrategy):
    def __init__(self, config: LissajousScanConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> LissajousScanStrategy:
        return cls(config=config.strategy.params.get("lissajous_scan", LissajousScanConfig()))

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.movements import Lissajous
        import math

        max_radius = ctx.config.simulation.max_search_radius
        A = max_radius / math.sqrt(2.0)
        duration = ctx.config.simulation.timeout

        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                Lissajous(
                    A=A,
                    wx=self.config.s1_wx,
                    wy=self.config.s1_wy,
                    delta=self.config.s1_delta,
                ),
                duration=duration,
                label="S1 lissajous scan"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                Lissajous(
                    A=A,
                    wx=self.config.s2_wx,
                    wy=self.config.s2_wy,
                    delta=self.config.s2_delta,
                ),
                duration=duration,
                label="S2 lissajous scan"
            )
        return script.build()
