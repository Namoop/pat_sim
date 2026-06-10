"""Strategy 1 — FOV-limited spiral (minor offset assumption)."""

from __future__ import annotations

from satellite.strategy.actions import ActionScript, Epoch, SatelliteAction
from satellite.strategy.base import SearchStrategy, StrategyContext
from satellite.strategy.movements import Hold, Spiral


class MinorOffsetStrategy(SearchStrategy):
    name = "minor_offset"

    def __init__(
        self,
        *,
        duration: float,
        max_spiral_radius: float,
        spiral_speed: float,
        w: float,
        k: float,
    ) -> None:
        self.duration = duration
        self.max_spiral_radius = max_spiral_radius
        self.spiral_speed = spiral_speed
        self.w = w
        self.k = k

    def build_script(self, ctx: StrategyContext) -> ActionScript:
        spiral = Spiral(
            w=self.w,
            k=self.k,
            max_radius=self.max_spiral_radius,
            speed=self.spiral_speed,
        )
        return ActionScript(
            epochs=(
                Epoch(
                    duration=self.duration,
                    s1=SatelliteAction(movement=spiral),
                    s2=SatelliteAction(movement=Hold()),
                    label="S1 FOV spiral",
                ),
            ),
        )
