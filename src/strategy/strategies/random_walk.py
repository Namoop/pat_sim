"""Strategy 11: Random Walk — Stochastic discrete point search."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.actions import beam, receiver, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class RandomWalkConfig:
    step_duration_a: float = 1.0
    step_duration_ratio: float = 1.41421356
    seed: int = 42


def parse_random_walk_config(data: dict) -> RandomWalkConfig:
    return RandomWalkConfig(
        step_duration_a=float(data.get("step_duration_a", data.get("step_duration", 1.0))),
        step_duration_ratio=float(data.get("step_duration_ratio", 1.41421356)),
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
        from strategy.movements import DiscretePattern

        max_radius = ctx.config.simulation.max_search_radius
        beam_width = ctx.config.satellite.alpha
        timeout = ctx.config.simulation.timeout
        
        # S1 points
        rng1 = random.Random(self.config.seed)
        step_duration_a = self.config.step_duration_a
        steps_s1 = max(1, int(timeout / step_duration_a))
        
        points_s1 = [(0.0, 0.0)]
        curr_u, curr_v = 0.0, 0.0
        for _ in range(steps_s1 - 1):
            curr_u += beam_width * 0.8 * 2.0 * (rng1.random() - 0.5)
            curr_v += beam_width * 0.8 * 2.0 * (rng1.random() - 0.5)
            dist = (curr_u**2 + curr_v**2)**0.5
            if dist > max_radius:
                curr_u *= max_radius / dist
                curr_v *= max_radius / dist
            points_s1.append((curr_u, curr_v))

        # S2 points
        rng2 = random.Random(self.config.seed + 1)
        step_duration_b = step_duration_a * self.config.step_duration_ratio
        steps_s2 = max(1, int(timeout / step_duration_b))
        
        points_s2 = [(0.0, 0.0)]
        curr_u, curr_v = 0.0, 0.0
        for _ in range(steps_s2 - 1):
            curr_u += beam_width * 0.8 * 2.0 * (rng2.random() - 0.5)
            curr_v += beam_width * 0.8 * 2.0 * (rng2.random() - 0.5)
            dist = (curr_u**2 + curr_v**2)**0.5
            if dist > max_radius:
                curr_u *= max_radius / dist
                curr_v *= max_radius / dist
            points_s2.append((curr_u, curr_v))

        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                DiscretePattern(points=tuple(points_s1), step_duration=step_duration_a),
                duration=timeout,
                label="S1 random walk"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                DiscretePattern(points=tuple(points_s2), step_duration=step_duration_b),
                duration=timeout,
                label="S2 random walk"
            )
        return script.build()
