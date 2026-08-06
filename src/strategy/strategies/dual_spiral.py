"""Strategy 4: Dual Spiral — Simultaneous irrationally-related spirals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.actions import beam, hold, inout_spiral, receiver, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class DualSpiralConfig:
    k_ratio: float = 1.0
    s2_hold_delay: float = 0.0
    phase_offset: float = 0.0
    radius: float | None = None


def parse_dual_spiral_config(data: dict) -> DualSpiralConfig:
    radius_raw = data.get("radius")
    radius = None if radius_raw is None else float(radius_raw) * 1e-3
    return DualSpiralConfig(
        k_ratio=float(data.get("k_ratio", 1.0)),
        s2_hold_delay=float(data.get("s2_hold_delay", 0.0)),
        phase_offset=float(data.get("phase_offset", 0.0)),
        radius=radius,
    )


@register_strategy("dual_spiral", parse_dual_spiral_config)
class DualSpiralStrategy(SearchStrategy):
    def __init__(self, config: DualSpiralConfig, w: float, k: float) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> DualSpiralStrategy:
        return cls(
            config=config.strategy.params.get("dual_spiral", DualSpiralConfig()),
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        max_radius = (
            self.config.radius
            if self.config.radius is not None
            else ctx.config.simulation.max_search_radius
        )
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy
        timeout = ctx.config.simulation.timeout

        # S1 parameters
        w1 = self.w
        k1 = self.k
        duration_s1 = strategy_config.spiral_duration(max_radius, w1, max_beam_speed)

        # S2 parameters
        w2 = self.w * self.config.k_ratio
        k2 = self.k * self.config.k_ratio
        duration_s2 = strategy_config.spiral_duration(max_radius, w2, max_beam_speed)

        script = strategy(self.name)

        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            current_t = 0.0
            cycle = 2 * duration_s1
            if cycle > 0.0:
                n_cycles = int((timeout - current_t) // cycle)
                if n_cycles > 0:
                    inout_spiral(
                        duration=n_cycles * cycle,
                        w=w1,
                        k=k1,
                        max_radius=max_radius,
                        one_way_duration=duration_s1,
                    )
                    current_t += n_cycles * cycle
            if current_t < timeout:
                hold(duration=timeout - current_t)

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            current_t = 0.0

            hold_delay = min(self.config.s2_hold_delay, timeout)
            if hold_delay > 0.0:
                hold(duration=hold_delay)
                current_t += hold_delay

            cycle = 2 * duration_s2
            if cycle > 0.0:
                n_cycles = int((timeout - current_t) // cycle)
                if n_cycles > 0:
                    inout_spiral(
                        duration=n_cycles * cycle,
                        w=w2,
                        k=k2,
                        max_radius=max_radius,
                        one_way_duration=duration_s2,
                        phase_offset=self.config.phase_offset,
                    )
                    current_t += n_cycles * cycle
            if current_t < timeout:
                hold(duration=timeout - current_t)

        return script.build()
