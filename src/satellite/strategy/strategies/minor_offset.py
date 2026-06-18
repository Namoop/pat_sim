"""Strategy 1 — FOV-limited spiral (minor offset assumption)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.sim.config import StrategyConfig
from satellite.strategy.actions import beam, hold, receiver, spiral, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from satellite.sim.config import ScenarioConfig


@dataclass(frozen=True)
class MinorOffsetConfig:
    max_spiral_radius: str | float


def parse_minor_offset_config(data: dict) -> MinorOffsetConfig:
    radius = data.get("max_spiral_radius", "fov")
    if isinstance(radius, (int, float)):
        radius = float(radius) * 1e-3
    return MinorOffsetConfig(
        max_spiral_radius=radius,
    )


@register_strategy("minor_offset", parse_minor_offset_config)
class MinorOffsetStrategy(SearchStrategy):
    def __init__(
        self,
        *,
        config: MinorOffsetConfig,
        w: float,
        k: float,
    ) -> None:
        self.config = config
        self.w = w
        self.k = k

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> MinorOffsetStrategy:
        return cls(
            config=config.strategy.params["minor_offset"],
            w=config.strategy.spiral_w(config.satellite),
            k=config.strategy.k,
        )

    def build_script(self, ctx: StrategyContext):
        radius = StrategyConfig.resolve_radius(
            self.config.max_spiral_radius, ctx.config.satellite.dish_fov
        )
        max_beam_speed = ctx.config.satellite.max_beam_speed
        duration = ctx.config.strategy.spiral_duration(
            radius, self.w, max_beam_speed
        )

        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable()
            receiver.enable()
            spiral(
                duration=duration,
                w=self.w,
                k=self.k,
                max_radius=radius,
                label="S1 FOV spiral",
            )
        with script.satellite("S2"):
            beam.enable()
            receiver.enable()
            hold(duration=duration, label="S2 hold")
        return script.build()
