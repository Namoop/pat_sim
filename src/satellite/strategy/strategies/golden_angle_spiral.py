"""Strategy 14: Golden Angle Spiral — Vogel's spiral discrete point distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class GoldenAngleConfig:
    num_points: int = 100
    step_duration: float = 0.5


def parse_golden_angle_config(data: dict) -> GoldenAngleConfig:
    return GoldenAngleConfig(
        num_points=int(data.get("num_points", 100)),
        step_duration=float(data.get("step_duration", 0.5)),
    )


def generate_vogel_points(radius: float, n: int) -> list[tuple[float, float]]:
    points = []
    # Golden angle in radians
    phi = (math.sqrt(5) + 1) / 2
    golden_angle = 2 * math.pi * (1 - 1/phi)
    
    for i in range(n):
        r = radius * math.sqrt(i / n)
        theta = i * golden_angle
        points.append((r * math.cos(theta), r * math.sin(theta)))
    return points


@register_strategy("golden_angle_spiral", parse_golden_angle_config)
class GoldenAngleStrategy(SearchStrategy):
    def __init__(self, config: GoldenAngleConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> GoldenAngleStrategy:
        return cls(config=config.strategy.params.get("golden_angle_spiral", GoldenAngleConfig()))

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.movements import DiscretePattern

        max_radius = ctx.config.simulation.max_search_radius
        points = generate_vogel_points(max_radius, self.config.num_points)
        total_duration = len(points) * self.config.step_duration

        script = strategy(self.name)
        for sat_name in ["S1", "S2"]:
            with script.satellite(sat_name):
                beam.enable(); receiver.enable()
                script._builders[sat_name].movement(
                    DiscretePattern(points=tuple(points), step_duration=self.config.step_duration),
                    duration=total_duration,
                    label=f"{sat_name} golden angle"
                )
        return script.build()
