"""Strategy 5: Dual Orthogonal Raster — Orthogonal boustrophedon scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class DualRasterConfig:
    steps_a: int = 20
    steps_b: int = 20
    speed: float = 1.0


def parse_dual_raster_config(data: dict) -> DualRasterConfig:
    return DualRasterConfig(
        steps_a=int(data.get("steps_a", 20)),
        steps_b=int(data.get("steps_b", 20)),
        speed=float(data.get("speed", 1.0)),
    )


@register_strategy("dual_raster", parse_dual_raster_config)
class DualRasterStrategy(SearchStrategy):
    def __init__(self, config: DualRasterConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> DualRasterStrategy:
        return cls(config=config.strategy.params.get("dual_raster", DualRasterConfig()))

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.actions import hold
        from satellite.strategy.movements import SerpentineRaster

        max_radius = ctx.config.simulation.max_search_radius
        # Duration is complex to calculate for raster, let's assume 10s for now
        # Actually, let's use the speed.
        duration = 10.0 / self.config.speed

        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                SerpentineRaster(radius=max_radius, steps=self.config.steps_a, horizontal=True),
                duration=duration,
                label="S1 horizontal raster"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                SerpentineRaster(radius=max_radius, steps=self.config.steps_b, horizontal=False),
                duration=duration,
                label="S2 vertical raster"
            )
        return script.build()
