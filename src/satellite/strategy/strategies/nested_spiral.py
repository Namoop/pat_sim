"""Strategy 13: Nested Spiral — Brute-force coordinated search for narrow beams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, hold, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class NestedSpiralConfig:
    outer_radius: float = 0.05
    inner_radius: float = 0.05
    spiral_speed: float = 1.0


def parse_nested_spiral_config(data: dict) -> NestedSpiralConfig:
    return NestedSpiralConfig(
        outer_radius=float(data.get("outer_radius", 0.05)),
        inner_radius=float(data.get("inner_radius", 0.05)),
        spiral_speed=float(data.get("spiral_speed", 1.0)),
    )


@register_strategy("nested_spiral", parse_nested_spiral_config)
class NestedSpiralStrategy(SearchStrategy):
    def __init__(self, config: NestedSpiralConfig, w: float, k: float) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> NestedSpiralStrategy:
        return cls(
            config=config.strategy.params.get("nested_spiral", NestedSpiralConfig()),
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        # S1 moves slowly (stays stationary for each S2 spiral)
        # S2 performs a complete spiral for every 'step' of S1
        
        # Simplified stub: S1 holds while S2 spirals
        duration_inner = ctx.config.strategy.spiral_duration(
            self.config.inner_radius, self.w, self.config.spiral_speed
        )
        
        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            hold(duration=duration_inner)
            
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            spiral(duration=duration_inner, w=self.w, k=self.k, max_radius=self.config.inner_radius, speed=self.config.spiral_speed)
            
        return script.build()
