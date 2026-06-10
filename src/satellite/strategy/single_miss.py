"""Strategy 2 — two-phase alternating spiral with bench reset."""

from __future__ import annotations

from satellite.strategy.actions import ActionScript, Epoch, SatelliteAction
from satellite.strategy.base import SearchStrategy, StrategyContext
from satellite.strategy.movements import Hold, Reset, Spiral


class SingleMissStrategy(SearchStrategy):
    name = "single_miss"

    def __init__(
        self,
        *,
        epoch1_duration: float,
        a_spiral_radius: float,
        reset_duration: float,
        epoch2_duration: float,
        b_spiral_radius: float,
        spiral_speed: float,
        w: float,
        k: float,
    ) -> None:
        self.epoch1_duration = epoch1_duration
        self.a_spiral_radius = a_spiral_radius
        self.reset_duration = reset_duration
        self.epoch2_duration = epoch2_duration
        self.b_spiral_radius = b_spiral_radius
        self.spiral_speed = spiral_speed
        self.w = w
        self.k = k

    def build_script(self, ctx: StrategyContext) -> ActionScript:
        spiral_a = Spiral(
            w=self.w,
            k=self.k,
            max_radius=self.a_spiral_radius,
            speed=self.spiral_speed,
        )
        spiral_b = Spiral(
            w=self.w,
            k=self.k,
            max_radius=self.b_spiral_radius,
            speed=self.spiral_speed,
        )
        epochs: list[Epoch] = [
            Epoch(
                duration=self.epoch1_duration,
                s1=SatelliteAction(movement=spiral_a),
                s2=SatelliteAction(movement=Hold()),
                label="S1 wide spiral",
            ),
        ]
        reset_dur = (
            self.reset_duration
            if self.reset_duration > 0.0
            else max(ctx.q_step, 1e-9)
        )
        epochs.append(
            Epoch(
                duration=reset_dur,
                s1=SatelliteAction(movement=Reset()),
                s2=SatelliteAction(movement=Hold()),
                label="S1 bench reset",
            ),
        )
        epochs.append(
            Epoch(
                duration=self.epoch2_duration,
                s1=SatelliteAction(movement=Hold()),
                s2=SatelliteAction(movement=spiral_b),
                label="S2 wide spiral",
            ),
        )
        return ActionScript(epochs=tuple(epochs))
