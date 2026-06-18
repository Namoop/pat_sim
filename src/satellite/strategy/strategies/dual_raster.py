"""Strategy 5: Dual Orthogonal Raster — Orthogonal boustrophedon scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.sim.config import ScenarioConfig


@dataclass(frozen=True)
class DualRasterConfig:
    steps_a: int = 20
    steps_b: int = 20
    speed_a: float = 1.0
    speed_ratio: float = 1.41421356


def parse_dual_raster_config(data: dict) -> DualRasterConfig:
    return DualRasterConfig(
        steps_a=int(data.get("steps_a", 20)),
        steps_b=int(data.get("steps_b", 20)),
        speed_a=float(data.get("speed_a", 1.0)),
        speed_ratio=float(data.get("speed_ratio", 1.41421356)),
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
        speed_a = self.config.speed_a
        speed_b = speed_a * self.config.speed_ratio
        
        duration_a = 10.0 / speed_a
        duration_b = 10.0 / speed_b
        timeout = ctx.config.simulation.timeout

        script = strategy(self.name)
        
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            if duration_a > 0.0:
                while current_t + duration_a <= timeout:
                    script._builders["S1"].movement(
                        SerpentineRaster(radius=max_radius, steps=self.config.steps_a, horizontal=True),
                        duration=duration_a,
                        label="S1 horizontal raster"
                    )
                    current_t += duration_a
            if current_t < timeout:
                hold(duration=timeout - current_t)

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            if duration_b > 0.0:
                while current_t + duration_b <= timeout:
                    script._builders["S2"].movement(
                        SerpentineRaster(radius=max_radius, steps=self.config.steps_b, horizontal=False),
                        duration=duration_b,
                        label="S2 vertical raster"
                    )
                    current_t += duration_b
            if current_t < timeout:
                hold(duration=timeout - current_t)

        return script.build()
