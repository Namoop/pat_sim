"""Strategy 13: Nested Spiral — Brute-force coordinated search for narrow beams.

No user-configurable parameters. Both S1 and S2 always scan out to
``max_search_radius`` and the spiral speed is fixed at ``max_beam_speed``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from strategy.actions import beam, hold, receiver, spiral, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


def parse_nested_spiral_config(_data: dict) -> None:
    """No parameters — returns None."""
    return None


@register_strategy("nested_spiral", parse_nested_spiral_config)
class NestedSpiralStrategy(SearchStrategy):
    def __init__(self, w: float, k: float) -> None:
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> NestedSpiralStrategy:
        return cls(
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        from strategy.movements import DiscretePattern

        sat_cfg = ctx.config.satellite
        sim_cfg = ctx.config.simulation

        max_radius = sim_cfg.max_search_radius
        timeout = sim_cfg.timeout
        max_beam_speed = sat_cfg.max_beam_speed

        # Duration for S2 to complete one full spiral pass
        duration_inner = ctx.config.strategy.spiral_duration(
            max_radius, self.w, max_beam_speed
        )

        # Generate step points for S1 — a sunflower lattice covering the cone
        d = sat_cfg.alpha
        if d <= 0.0:
            d = 0.005  # fallback if alpha not set

        num_steps_needed = max(1, int(timeout / duration_inner)) + 2

        base_points: list[tuple[float, float]] = [(0.0, 0.0)]
        i = 1
        while True:
            theta = 2.0 * math.sqrt(math.pi * i)
            r = d * math.sqrt(i / math.pi)
            if r > max_radius:
                break
            base_points.append((r * math.cos(theta), r * math.sin(theta)))
            i += 1

        # Tile the lattice to span the full simulation timeout
        points: list[tuple[float, float]] = []
        while len(points) < num_steps_needed:
            points.extend(base_points)
        points = points[:num_steps_needed]

        script = strategy(self.name)

        with script.satellite("S1"):
            beam.enable()
            receiver.enable()
            script._builders["S1"].movement(
                DiscretePattern(points=tuple(points), step_duration=duration_inner),
                duration=timeout,
                label="S1 stepping",
            )

        with script.satellite("S2"):
            beam.enable()
            receiver.enable()
            current_t = 0.0
            out_spiral = True
            if duration_inner > 0.0:
                while current_t + duration_inner <= timeout:
                    spiral(
                        duration=duration_inner,
                        w=self.w,
                        k=self.k,
                        max_radius=max_radius if out_spiral else 0.0,
                    )
                    out_spiral = not out_spiral
                    current_t += duration_inner
            if current_t < timeout:
                hold(duration=timeout - current_t)

        return script.build()
