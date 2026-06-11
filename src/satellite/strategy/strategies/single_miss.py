"""Strategy 2: two-phase alternating spiral with bench reset."""

from __future__ import annotations

from satellite.strategy.actions import beam, hold, receiver, reset, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext


class SingleMissStrategy(SearchStrategy):
    name = "single_miss"

    def __init__(
        self,
        *,
        phase1_duration: float,
        a_spiral_radius: float,
        reset_duration: float,
        phase2_duration: float,
        b_spiral_radius: float,
        spiral_speed: float,
        w: float,
        k: float,
    ) -> None:
        self.phase1_duration = phase1_duration
        self.a_spiral_radius = a_spiral_radius
        self.reset_duration = reset_duration
        self.phase2_duration = phase2_duration
        self.b_spiral_radius = b_spiral_radius
        self.spiral_speed = spiral_speed
        self.w = w
        self.k = k

    def build_script(self, ctx: StrategyContext):
        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable()
            receiver.enable()
            spiral(
                duration=self.phase1_duration,
                w=self.w,
                k=self.k,
                max_radius=self.a_spiral_radius,
                speed=self.spiral_speed,
                label="S1 wide spiral",
            )
            reset(duration=self.reset_duration, label="S1 bench reset")
            hold(duration=self.phase2_duration, label="S1 hold")
        with script.satellite("S2"):
            beam.enable()
            receiver.enable()
            hold(duration=self.phase1_duration, label="S2 hold")
            hold(duration=self.reset_duration, label="S2 reset wait")
            spiral(
                duration=self.phase2_duration,
                w=self.w,
                k=self.k,
                max_radius=self.b_spiral_radius,
                speed=self.spiral_speed,
                label="S2 wide spiral",
            )
        return script.build()
