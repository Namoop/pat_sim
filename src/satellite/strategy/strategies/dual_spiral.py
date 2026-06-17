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
    pass


def parse_dual_spiral_config(data: dict) -> DualSpiralConfig:
    return DualSpiralConfig()


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
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy

        # Duration for out-and-back spiral
        duration = strategy_config.spiral_duration(max_radius, self.w, max_beam_speed)
        timeout = ctx.config.simulation.timeout

        script = strategy(self.name)
        
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            cycle = 2 * duration
            if cycle > 0.0:
                while current_t + cycle <= timeout:
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=max_radius)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0)
                    current_t += cycle
            if current_t < timeout:
                hold(duration=timeout - current_t)

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            cycle = 2 * duration
            if cycle > 0.0:
                while current_t + cycle <= timeout:
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=max_radius)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0)
                    current_t += cycle
            if current_t < timeout:
                hold(duration=timeout - current_t)

        return script.build()
