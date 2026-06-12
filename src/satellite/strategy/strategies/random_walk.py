"""Strategy 11: Random Walk — Stochastic discrete point search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class RandomWalkConfig:
    step_duration: float = 1.0
    total_duration: float = 20.0
    seed: int = 42


def parse_random_walk_config(data: dict) -> RandomWalkConfig:
    return RandomWalkConfig(
        step_duration=float(data.get("step_duration", 1.0)),
        total_duration=float(data.get("total_duration", 20.0)),
        seed=int(data.get("seed", 42)),
    )


@register_strategy("random_walk", parse_random_walk_config)
class RandomWalkStrategy(SearchStrategy):
    def __init__(self, config: RandomWalkConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> RandomWalkStrategy:
        return cls(config=config.strategy.params.get("random_walk", RandomWalkConfig()))

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.movements import DiscretePattern

        max_radius = ctx.config.simulation.max_search_radius
        beam_width = ctx.config.satellite.alpha
        
        rng = random.Random(self.config.seed)
        steps = int(self.config.total_duration / self.config.step_duration)
        
        points = [(0.0, 0.0)]
        curr_u, curr_v = 0.0, 0.0
        
        for _ in range(steps - 1):
            angle = rng.uniform(0, 2 * 3.14159)
            # move by roughly one beam width
            curr_u += beam_width * 0.8 * 2.0 * (rng.random() - 0.5)
            curr_v += beam_width * 0.8 * 2.0 * (rng.random() - 0.5)
            
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
                    label=f"{sat_name} random walk"
                )
        return script.build()
