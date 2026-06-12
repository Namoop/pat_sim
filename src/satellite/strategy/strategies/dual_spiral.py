"""Strategy 4: Dual Spiral — Simultaneous irrationally-related spirals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class DualSpiralConfig:
    speed_a: float = 1.0
    speed_ratio: float = 1.41421356  # sqrt(2)


def parse_dual_spiral_config(data: dict) -> DualSpiralConfig:
    return DualSpiralConfig(
        speed_a=float(data.get("speed_a", 1.0)),
        speed_ratio=float(data.get("speed_ratio", 1.41421356)),
    )


@register_strategy("dual_spiral", parse_dual_spiral_config)
class DualSpiralStrategy(SearchStrategy):
    def __init__(self, config: DualSpiralConfig, w: float, k: float) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> DualSpiralStrategy:
        return cls(
            config=config.strategy.params.get("dual_spiral", DualSpiralConfig()),
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.actions import hold

        max_radius = ctx.config.simulation.max_search_radius
        speed_a = self.config.speed_a
        speed_b = speed_a * self.config.speed_ratio
        strategy_config = ctx.config.strategy

        # Duration for out-and-back spiral
        duration_a = strategy_config.spiral_duration(max_radius, self.w, speed_a)
        duration_b = strategy_config.spiral_duration(max_radius, self.w, speed_b)
        timeout = ctx.config.simulation.timeout

        script = strategy(self.name)
        
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            cycle_a = 2 * duration_a
            if cycle_a > 0.0:
                while current_t + cycle_a <= timeout:
                    spiral(duration=duration_a, w=self.w, k=self.k, max_radius=max_radius, speed=speed_a)
                    spiral(duration=duration_a, w=self.w, k=self.k, max_radius=0, speed=speed_a)
                    current_t += cycle_a
            if current_t < timeout:
                hold(duration=timeout - current_t)

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            cycle_b = 2 * duration_b
            if cycle_b > 0.0:
                while current_t + cycle_b <= timeout:
                    spiral(duration=duration_b, w=self.w, k=self.k, max_radius=max_radius, speed=speed_b)
                    spiral(duration=duration_b, w=self.w, k=self.k, max_radius=0, speed=speed_b)
                    current_t += cycle_b
            if current_t < timeout:
                hold(duration=timeout - current_t)

        return script.build()
