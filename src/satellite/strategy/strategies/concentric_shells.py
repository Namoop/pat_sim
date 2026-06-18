"""Strategy 10: Concentric Shells — Progressive depth search with center resets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.sim.config import ScenarioConfig


@dataclass(frozen=True)
class ConcentricShellsConfig:
    num_shells: int = 3
    growth_exponent: float = 1.0
    s2_offset_shells: int = 0


def parse_concentric_shells_config(data: dict) -> ConcentricShellsConfig:
    return ConcentricShellsConfig(
        num_shells=int(data.get("num_shells", 3)),
        growth_exponent=float(data.get("growth_exponent", 1.0)),
        s2_offset_shells=int(data.get("s2_offset_shells", 0)),
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
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy
        timeout = ctx.config.simulation.timeout

        # Generate radii factors based on the parameters
        num_shells = max(1, self.config.num_shells)
        growth_exponent = self.config.growth_exponent
        
        factors = []
        for i in range(1, num_shells + 1):
            factors.append((i / num_shells) ** growth_exponent)

        script = strategy(self.name)
        import itertools

        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            factors_s1 = itertools.cycle(factors)
            while current_t < timeout:
                factor = next(factors_s1)
                r = factor * max_radius
                duration = strategy_config.spiral_duration(r, self.w, max_beam_speed)
                step_dur = 2 * duration
                if step_dur <= 0.0:
                    step_dur = 1.0
                    hold(duration=1.0)
                else:
                    if current_t + step_dur > timeout:
                        hold(duration=timeout - current_t)
                        break
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=r)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0)
                current_t += step_dur

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            
            # Roll factors for S2
            offset = self.config.s2_offset_shells % num_shells
            rolled_factors = factors[offset:] + factors[:offset]
            factors_s2 = itertools.cycle(rolled_factors)
            
            while current_t < timeout:
                factor = next(factors_s2)
                r = factor * max_radius
                duration = strategy_config.spiral_duration(r, self.w, max_beam_speed)
                step_dur = 2 * duration
                if step_dur <= 0.0:
                    step_dur = 1.0
                    hold(duration=1.0)
                else:
                    if current_t + step_dur > timeout:
                        hold(duration=timeout - current_t)
                        break
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=r)
                    spiral(duration=duration, w=self.w, k=self.k, max_radius=0)
                current_t += step_dur

        return script.build()
