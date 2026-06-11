"""Strategy 1 — FOV-limited spiral (minor offset assumption)."""

from __future__ import annotations

from satellite.strategy.actions import beam, hold, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext


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

    def build_script(self, ctx: StrategyContext):
        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable()
            receiver.enable()
            spiral(
                duration=self.duration,
                w=self.w,
                k=self.k,
                max_radius=self.max_spiral_radius,
                speed=self.spiral_speed,
                label="S1 FOV spiral",
            )
        with script.satellite("S2"):
            beam.enable()
            receiver.enable()
            hold(duration=self.duration, label="S2 hold")
        return script.build()
