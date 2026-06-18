"""Strategy 3: Asymmetric Swap — Role-reversing probe to mitigate mutual blindness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.sim.config import StrategyConfig
from satellite.strategy.actions import beam, hold, receiver, reset, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.sim.config import ScenarioConfig


@dataclass(frozen=True)
class AsymmetricSwapConfig:
    spiral_radius: str | float
    lock_duration: float


def parse_asymmetric_swap_config(data: dict) -> AsymmetricSwapConfig:
    radius = data.get("spiral_radius", 50.0)
    if isinstance(radius, (int, float)):
        radius = float(radius) * 1e-3
    return AsymmetricSwapConfig(
        spiral_radius=radius,
        lock_duration=float(data.get("lock_duration", 1.0)),
    )


@register_strategy("asymmetric_swap", parse_asymmetric_swap_config)
class AsymmetricSwapStrategy(SearchStrategy):
    def __init__(
        self,
        *,
        config: AsymmetricSwapConfig,
        w: float,
        k: float,
    ) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> AsymmetricSwapStrategy:
        # Fallback to asymmetric_probe if swap not present in config yet
        params = config.strategy.params.get("asymmetric_swap")
        if params is None:
            params = config.strategy.params.get("asymmetric_probe")
        if params is None:
            params = AsymmetricSwapConfig(spiral_radius=0.05, lock_duration=1.0)
        
        return cls(
            config=params,
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        dish_fov = ctx.config.satellite.dish_fov
        radius = StrategyConfig.resolve_radius(self.config.spiral_radius, dish_fov)
        max_beam_speed = ctx.config.satellite.max_beam_speed
        strategy_config = ctx.config.strategy

        probe_duration = strategy_config.spiral_duration(
            radius, self.w, max_beam_speed
        )
        reset_duration = strategy_config.reset_duration(radius, max_beam_speed)

        script = strategy(self.name)
        
        # S1 (Initial Leader)
        with script.satellite("S1"):
            # Phase 1: Probing
            beam.enable(); receiver.disable()
            spiral(duration=probe_duration, w=self.w, k=self.k, max_radius=radius, 
                   label="S1 probe spiral")
            reset(duration=reset_duration, label="S1 reset")

            # Phase 2: Listening (Swap)
            beam.disable(); receiver.enable()
            hold(duration=probe_duration, label="S1 listen")
            reset(duration=reset_duration, label="S1 reset 2")

            # Phase 3: Reciprocal Lock
            beam.enable(); receiver.enable()
            hold(duration=self.config.lock_duration, label="S1 lock verify")

        # S2 (Initial Follower)
        with script.satellite("S2"):
            # Phase 1: Listening
            beam.disable(); receiver.enable()
            hold(duration=probe_duration, label="S2 listen")
            hold(duration=reset_duration, label="S2 wait")

            # Phase 2: Probing (Swap)
            beam.enable(); receiver.disable()
            spiral(duration=probe_duration, w=self.w, k=self.k, max_radius=radius, 
                   label="S2 probe spiral")
            reset(duration=reset_duration, label="S2 reset")

            # Phase 3: Reciprocal Lock
            beam.enable(); receiver.enable()
            hold(duration=self.config.lock_duration, label="S2 lock verify")

        return script.build()
