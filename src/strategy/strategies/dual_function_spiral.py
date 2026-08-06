"""Strategy 3: Dual Function Spiral — Role-reversing probe to mitigate mutual blindness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.config import StrategyConfig
from strategy.actions import beam, hold, receiver, reset, spiral, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class DualFunctionConfig:
    spiral_radius: str | float
    lock_duration: float
    s2_radius_mod: float = 1.0


def parse_dual_function_config(data: dict) -> DualFunctionConfig:
    radius = data.get("spiral_radius", 50.0)
    if isinstance(radius, (int, float)):
        radius = float(radius) * 1e-3
    return DualFunctionConfig(
        spiral_radius=radius,
        lock_duration=float(data.get("lock_duration", 1.0)),
        s2_radius_mod=float(data.get("s2_radius_mod", 1.0)),
    )


@register_strategy("dual_function_spiral", parse_dual_function_config)
class DualFunctionStrategy(SearchStrategy):
    def __init__(
        self,
        *,
        config: DualFunctionConfig,
        w: float,
        k: float,
    ) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> DualFunctionStrategy:
        params = config.strategy.params.get("dual_function_spiral")
        if params is None:
            params = DualFunctionConfig(
                spiral_radius=0.05,
                lock_duration=1.0,
                s2_radius_mod=1.0,
            )

        return cls(
            config=params,
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        dish_fov_s1 = ctx.config.satellite.dish_fov
        dish_fov_s2 = dish_fov_s1 * ctx.config.satellite.s2_fov_mod
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy

        radius_s1 = StrategyConfig.resolve_radius(self.config.spiral_radius, dish_fov_s1)
        radius_s2 = StrategyConfig.resolve_radius(self.config.spiral_radius, dish_fov_s2) * self.config.s2_radius_mod

        probe_duration = strategy_config.spiral_duration(
            radius_s1, self.w, max_beam_speed
        )
        reset_duration = strategy_config.reset_duration(radius_s1, max_beam_speed)

        s2_probe_duration = strategy_config.spiral_duration(
            radius_s2, self.w, max_beam_speed
        )
        s2_reset_duration = strategy_config.reset_duration(radius_s2, max_beam_speed)

        script = strategy(self.name)
        
        # S1 (Initial Leader)
        with script.satellite("S1"):
            # Phase 1: Probing
            beam.enable(); receiver.enable()    # BOTH FUNCTIONS ENABLED
            spiral(duration=probe_duration, w=self.w, k=self.k, max_radius=radius_s1, 
                   label="S1 probe spiral")
            reset(duration=reset_duration, label="S1 reset")

            # Phase 2: Listening (Swap)
            hold(duration=s2_probe_duration, label="S1 listen")
            reset(duration=s2_reset_duration, label="S1 reset 2")

            # Phase 3: Reciprocal Lock
            hold(duration=self.config.lock_duration, label="S1 lock verify")

        # S2 (Initial Follower)
        with script.satellite("S2"):
            # Phase 1: Listening
            beam.enable(); receiver.enable()    # BOTH FUNCTIONS ENABLED
            hold(duration=probe_duration, label="S2 listen")
            hold(duration=reset_duration, label="S2 wait")

            # Phase 2: Probing (Swap)
            spiral(duration=s2_probe_duration, w=self.w, k=self.k, max_radius=radius_s2, 
                   label="S2 probe spiral")
            reset(duration=s2_reset_duration, label="S2 reset")

            # Phase 3: Reciprocal Lock
            hold(duration=self.config.lock_duration, label="S2 lock verify")

        return script.build()
