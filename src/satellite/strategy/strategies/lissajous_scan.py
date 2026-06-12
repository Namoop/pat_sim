"""Strategy 7: Lissajous Scan — Sinusoidal orthogonal scanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class LissajousScanConfig:
    wx: float = 1.0
    wy: float = 1.41421356
    delta: float = 0.0
    duration: float = 10.0


def parse_lissajous_scan_config(data: dict) -> LissajousScanConfig:
    return LissajousScanConfig(
        wx=float(data.get("wx", 1.0)),
        wy=float(data.get("wy", 1.41421356)),
        delta=float(data.get("delta", 0.0)),
        duration=float(data.get("duration", 10.0)),
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

        max_radius = ctx.config.simulation.max_search_radius
        duration = self.config.duration

        script = strategy(self.name)
        for sat_name in ["S1", "S2"]:
            with script.satellite(sat_name):
                beam.enable(); receiver.enable()
                script._builders[sat_name].movement(
                    Lissajous(
                        A=max_radius,
                        wx=self.config.wx,
                        wy=self.config.wy,
                        delta=self.config.delta,
                    ),
                    duration=duration,
                    label=f"{sat_name} lissajous scan"
                )
        return script.build()
