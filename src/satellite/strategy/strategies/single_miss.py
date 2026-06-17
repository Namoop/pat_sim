"""Strategy 2: two-phase alternating spiral with bench reset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.config import StrategyConfig
from satellite.strategy.actions import beam, hold, receiver, reset, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class SingleMissConfig:
    a_spiral_radius: str | float
    b_spiral_radius: str | float


def parse_single_miss_config(data: dict) -> SingleMissConfig:
    a_radius = data.get("a_spiral_radius", 50.0)
    if isinstance(a_radius, (int, float)):
        a_radius = float(a_radius) * 1e-3
    b_radius = data.get("b_spiral_radius", 50.0)
    if isinstance(b_radius, (int, float)):
        b_radius = float(b_radius) * 1e-3
    return SingleMissConfig(
        a_spiral_radius=a_radius,
        b_spiral_radius=b_radius,
    )


@register_strategy("single_miss", parse_single_miss_config)
class SingleMissStrategy(SearchStrategy):
    def __init__(
        self,
        *,
        config: SingleMissConfig,
        w: float,
        k: float,
    ) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> SingleMissStrategy:
        return cls(
            config=config.strategy.params["single_miss"],
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        dish_fov = ctx.config.satellite.dish_fov
        a_radius = StrategyConfig.resolve_radius(self.config.a_spiral_radius, dish_fov)
        b_radius = StrategyConfig.resolve_radius(self.config.b_spiral_radius, dish_fov)
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy

        p1_duration = strategy_config.spiral_duration(
            a_radius, self.w, max_beam_speed
        )
        p2_duration = strategy_config.spiral_duration(
            b_radius, self.w, max_beam_speed
        )

        # Reset from max radius back to center
        reset_duration = strategy_config.reset_duration(
            max(a_radius, b_radius), max_beam_speed
        )

        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable()
            receiver.enable()
            spiral(
                duration=p1_duration,
                w=self.w,
                k=self.k,
                max_radius=a_radius,
                label="S1 wide spiral",
            )
            reset(duration=reset_duration, label="S1 bench reset")
            hold(duration=p2_duration, label="S1 hold")
        with script.satellite("S2"):
            beam.enable()
            receiver.enable()
            hold(duration=p1_duration, label="S2 hold")
            hold(duration=reset_duration, label="S2 reset wait")
            spiral(
                duration=p2_duration,
                w=self.w,
                k=self.k,
                max_radius=b_radius,
                label="S2 wide spiral",
            )
        return script.build()
