"""Asymmetric probe strategy with delayed reciprocal beam."""

from __future__ import annotations

from satellite.strategy.actions import beam, hold, receiver, reset, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext


class AsymmetricProbeStrategy(SearchStrategy):
    name = "asymmetric_probe"

    def __init__(
        self,
        *,
        probe_duration: float,
        spiral_radius: float,
        spiral_speed: float,
        reset_duration: float,
        w: float,
        k: float,
    ) -> None:
        self.probe_duration = probe_duration
        self.spiral_radius = spiral_radius
        self.spiral_speed = spiral_speed
        self.reset_duration = reset_duration
        self.w = w
        self.k = k

    def build_script(self, ctx: StrategyContext):
        slew_timeout = max(ctx.config.satellite.bench_slew_time, ctx.t_step)
        script = strategy(self.name)
        with script.satellite("S1"):
            # Phase 1: Leader
            beam.enable()
            receiver.disable()
            spiral(
                duration=self.probe_duration,
                w=self.w,
                k=self.k,
                max_radius=self.spiral_radius,
                speed=self.spiral_speed,
                label="S1 probe spiral",
            )
            reset(duration=self.reset_duration, label="S1 bench reset")
            beam.disable()

            # Phase 2: Check
            receiver.enable()
            beam.enable()
            hold(duration=slew_timeout, label="S1 reciprocal receive")
        with script.satellite("S2"):
            # Phase 1: Follower
            beam.disable()
            receiver.enable()
            hold(duration=self.probe_duration, label="S2 receive probe")
            hold(duration=self.reset_duration, label="S2 reset wait")

            # Phase 2: Check
            beam.enable()
            hold(duration=slew_timeout, label="S2 reciprocal beam")
        return script.build()
