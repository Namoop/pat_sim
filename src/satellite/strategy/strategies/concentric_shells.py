"""Strategy 10: Concentric Shells — Progressive depth search with center resets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, reset, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class ConcentricShellsConfig:
    radii_factors: tuple[float, ...] = (0.2, 0.5, 1.0)
    spiral_speed: float = 1.0


def parse_concentric_shells_config(data: dict) -> ConcentricShellsConfig:
    return ConcentricShellsConfig(
        radii_factors=tuple(data.get("radii_factors", (0.2, 0.5, 1.0))),
        spiral_speed=float(data.get("spiral_speed", 1.0)),
    )


@register_strategy("concentric_shells", parse_concentric_shells_config)
class ConcentricShellsStrategy(SearchStrategy):
    def __init__(self, config: ConcentricShellsConfig, w: float, k: float) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> ConcentricShellsStrategy:
        return cls(
            config=config.strategy.params.get("concentric_shells", ConcentricShellsConfig()),
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        max_radius = ctx.config.simulation.max_search_radius
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy

        script = strategy(self.name)
        for sat_name in ["S1", "S2"]:
            with script.satellite(sat_name):
                beam.enable(); receiver.enable()
                for factor in self.config.radii_factors:
                    r = factor * max_radius
                    duration = strategy_config.spiral_duration(
                        r, self.w, self.config.spiral_speed
                    )
                    reset_duration = strategy_config.reset_duration(
                        r, max_beam_speed
                    )
                    
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=r, speed=self.config.spiral_speed)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0, speed=self.config.spiral_speed)
                    reset(duration=reset_duration)
                    
        return script.build()
