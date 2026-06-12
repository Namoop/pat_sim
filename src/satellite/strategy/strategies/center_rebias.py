"""Strategy 9: Stochastic Center Re-bias — Random walk with center-heavy pull."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class CenterRebiasConfig:
    step_duration: float = 0.5
    bias_strength: float = 0.2
    total_duration: float = 20.0
    seed: int = 123


def parse_center_rebias_config(data: dict) -> CenterRebiasConfig:
    return CenterRebiasConfig(
        step_duration=float(data.get("step_duration", 0.5)),
        bias_strength=float(data.get("bias_strength", 0.2)),
        total_duration=float(data.get("total_duration", 20.0)),
        seed=int(data.get("seed", 123)),
    )


@register_strategy("center_rebias", parse_center_rebias_config)
class CenterRebiasStrategy(SearchStrategy):
    def __init__(self, config: CenterRebiasConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> CenterRebiasStrategy:
        return cls(config=config.strategy.params.get("center_rebias", CenterRebiasConfig()))

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.movements import DiscretePattern

        max_radius = ctx.config.simulation.max_search_radius
        beam_width = ctx.config.satellite.alpha
        
        rng = random.Random(self.config.seed)
        steps = int(self.config.total_duration / self.config.step_duration)
        
        points = [(0.0, 0.0)]
        curr_u, curr_v = 0.0, 0.0
        
        for _ in range(steps - 1):
            # Random step
            curr_u += beam_width * (rng.random() - 0.5)
            curr_v += beam_width * (rng.random() - 0.5)
            
            # Apply bias pull toward (0,0)
            curr_u *= (1.0 - self.config.bias_strength)
            curr_v *= (1.0 - self.config.bias_strength)
            
            # Bound check
            dist = (curr_u**2 + curr_v**2)**0.5
            if dist > max_radius:
                curr_u *= max_radius / dist
                curr_v *= max_radius / dist
                
            points.append((curr_u, curr_v))

        script = strategy(self.name)
        for sat_name in ["S1", "S2"]:
            with script.satellite(sat_name):
                beam.enable(); receiver.enable()
                script._builders[sat_name].movement(
                    DiscretePattern(points=tuple(points), step_duration=self.config.step_duration),
                    duration=self.config.total_duration,
                    label=f"{sat_name} center rebias"
                )
        return script.build()
