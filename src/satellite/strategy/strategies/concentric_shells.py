"""Strategy 10: Concentric Shells — Progressive depth search with center resets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class ConcentricShellsConfig:
    radii_factors: tuple[float, ...] = (0.2, 0.5, 1.0)
    spiral_speed_a: float = 1.0
    speed_ratio: float = 1.41421356


def parse_concentric_shells_config(data: dict) -> ConcentricShellsConfig:
    return ConcentricShellsConfig(
        radii_factors=tuple(data.get("radii_factors", (0.2, 0.5, 1.0))),
        spiral_speed_a=float(data.get("spiral_speed_a", data.get("spiral_speed", 1.0))),
        speed_ratio=float(data.get("speed_ratio", 1.41421356)),
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
        from satellite.strategy.actions import hold

        max_radius = ctx.config.simulation.max_search_radius
        strategy_config = ctx.config.strategy
        
        spiral_speed_a = self.config.spiral_speed_a
        spiral_speed_b = spiral_speed_a * self.config.speed_ratio
        timeout = ctx.config.simulation.timeout

        script = strategy(self.name)
        import itertools

        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            factors_s1 = itertools.cycle(self.config.radii_factors)
            while current_t < timeout:
                factor = next(factors_s1)
                r = factor * max_radius
                duration = strategy_config.spiral_duration(r, self.w, spiral_speed_a)
                step_dur = 2 * duration
                if step_dur <= 0.0:
                    step_dur = 1.0
                    hold(duration=1.0)
                else:
                    if current_t + step_dur > timeout:
                        hold(duration=timeout - current_t)
                        break
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=r, speed=spiral_speed_a)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0, speed=spiral_speed_a)
                current_t += step_dur

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            factors_s2 = itertools.cycle(self.config.radii_factors)
            while current_t < timeout:
                factor = next(factors_s2)
                r = factor * max_radius
                duration = strategy_config.spiral_duration(r, self.w, spiral_speed_b)
                step_dur = 2 * duration
                if step_dur <= 0.0:
                    step_dur = 1.0
                    hold(duration=1.0)
                else:
                    if current_t + step_dur > timeout:
                        hold(duration=timeout - current_t)
                        break
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=r, speed=spiral_speed_b)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0, speed=spiral_speed_b)
                current_t += step_dur

        return script.build()
