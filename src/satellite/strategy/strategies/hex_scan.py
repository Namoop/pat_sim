"""Strategy 6: Hexagonal Grid Scan — Discrete points on a hexagonal lattice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class HexScanConfig:
    step_size: float = 0.003
    step_duration: float = 0.5


def parse_hex_scan_config(data: dict) -> HexScanConfig:
    return HexScanConfig(
        step_size=float(data.get("step_size", 0.003)),
        step_duration=float(data.get("step_duration", 0.5)),
    )


def generate_hex_points(radius: float, step_size: float) -> list[tuple[float, float]]:
    points = [(0.0, 0.0)]
    # Concentric rings
    n_rings = int(radius / (step_size * math.sqrt(3) / 2)) + 1
    for r in range(1, n_rings + 1):
        for i in range(6):
            # Hexagon vertex
            angle = i * math.pi / 3
            x = r * step_size * math.cos(angle)
            y = r * step_size * math.sin(angle)
            
            # Points along the edge
            next_angle = (i + 1) * math.pi / 3
            nx = r * step_size * math.cos(next_angle)
            ny = r * step_size * math.sin(next_angle)
            
            for j in range(r):
                px = x + (nx - x) * j / r
                py = y + (ny - y) * j / r
                if px*px + py*py <= radius*radius:
                    points.append((px, py))
    return points


@register_strategy("hex_scan", parse_hex_scan_config)
class HexScanStrategy(SearchStrategy):
    def __init__(self, config: HexScanConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> HexScanStrategy:
        return cls(config=config.strategy.params.get("hex_scan", HexScanConfig()))

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.movements import DiscretePattern

        max_radius = ctx.config.simulation.max_search_radius
        points = generate_hex_points(max_radius, self.config.step_size)
        total_duration = len(points) * self.config.step_duration

        script = strategy(self.name)
        for sat_name in ["S1", "S2"]:
            with script.satellite(sat_name):
                beam.enable(); receiver.enable()
                script._builders[sat_name].movement(
                    DiscretePattern(points=tuple(points), step_duration=self.config.step_duration),
                    duration=total_duration,
                    label=f"{sat_name} hex scan"
                )
        return script.build()
