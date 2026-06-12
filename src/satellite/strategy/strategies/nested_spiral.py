"""Strategy 13: Nested Spiral — Brute-force coordinated search for narrow beams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.strategy.actions import beam, hold, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class NestedSpiralConfig:
    outer_radius: float = 0.05
    inner_radius: float = 0.05
    spiral_speed: float = 1.0


def parse_nested_spiral_config(data: dict) -> NestedSpiralConfig:
    return NestedSpiralConfig(
        outer_radius=float(data.get("outer_radius", 0.05)),
        inner_radius=float(data.get("inner_radius", 0.05)),
        spiral_speed=float(data.get("spiral_speed", 1.0)),
    )


@register_strategy("nested_spiral", parse_nested_spiral_config)
class NestedSpiralStrategy(SearchStrategy):
    def __init__(self, config: NestedSpiralConfig, w: float, k: float) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> NestedSpiralStrategy:
        return cls(
            config=config.strategy.params.get("nested_spiral", NestedSpiralConfig()),
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        from satellite.strategy.movements import DiscretePattern
        import math

        max_radius = ctx.config.simulation.max_search_radius
        timeout = ctx.config.simulation.timeout
        
        # S2 performs a spiral out to max_radius (or back) in duration_inner
        duration_inner = ctx.config.strategy.spiral_duration(
            max_radius, self.w, self.config.spiral_speed
        )
        
        # Generate step points for S1 along a slow spiral of pitch and spacing equal to beam width (alpha)
        d = ctx.config.satellite.alpha
        if d <= 0.0:
            d = 0.005  # fallback
            
        num_steps_needed = max(1, int(timeout / duration_inner)) + 2
        
        base_points = [(0.0, 0.0)]
        i = 1
        while True:
            # Angle theta_i
            theta = 2.0 * math.sqrt(math.pi * i)
            # Radius r_i
            r = d * math.sqrt(i / math.pi)
            if r > max_radius:
                break
            u = r * math.cos(theta)
            v = r * math.sin(theta)
            base_points.append((u, v))
            i += 1
            
        # Repeat base points to cover the entire duration of the simulation
        points = []
        while len(points) < num_steps_needed:
            points.extend(base_points)
        points = points[:num_steps_needed]

        script = strategy(self.name)
        
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                DiscretePattern(points=tuple(points), step_duration=duration_inner),
                duration=timeout,
                label="S1 stepping"
            )
            
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            out_spiral = True
            if duration_inner > 0.0:
                while current_t + duration_inner <= timeout:
                    spiral(
                        duration=duration_inner,
                        w=self.w,
                        k=self.k,
                        max_radius=max_radius if out_spiral else 0.0,
                        speed=self.config.spiral_speed
                    )
                    out_spiral = not out_spiral
                    current_t += duration_inner
            if current_t < timeout:
                hold(duration=timeout - current_t)
            
        return script.build()
