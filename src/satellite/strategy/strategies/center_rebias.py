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
    step_duration_a: float = 0.5
    step_duration_ratio: float = 1.41421356
    bias_strength: float = 0.2
    seed: int = 123


def parse_center_rebias_config(data: dict) -> CenterRebiasConfig:
    return CenterRebiasConfig(
        step_duration_a=float(data.get("step_duration_a", data.get("step_duration", 0.5))),
        step_duration_ratio=float(data.get("step_duration_ratio", 1.41421356)),
        bias_strength=float(data.get("bias_strength", 0.2)),
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
        timeout = ctx.config.simulation.timeout
        
        # S1 points
        rng1 = random.Random(self.config.seed)
        step_duration_a = self.config.step_duration_a
        steps_s1 = max(1, int(timeout / step_duration_a))
        
        points_s1 = [(0.0, 0.0)]
        curr_u, curr_v = 0.0, 0.0
        for _ in range(steps_s1 - 1):
            curr_u += beam_width * (rng1.random() - 0.5)
            curr_v += beam_width * (rng1.random() - 0.5)
            curr_u *= (1.0 - self.config.bias_strength)
            curr_v *= (1.0 - self.config.bias_strength)
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
            curr_u += beam_width * (rng2.random() - 0.5)
            curr_v += beam_width * (rng2.random() - 0.5)
            curr_u *= (1.0 - self.config.bias_strength)
            curr_v *= (1.0 - self.config.bias_strength)
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
                label="S1 center rebias"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                DiscretePattern(points=tuple(points_s2), step_duration=step_duration_b),
                duration=timeout,
                label="S2 center rebias"
            )
        return script.build()
