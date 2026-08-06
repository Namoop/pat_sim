"""Strategy 8: Rosette Scan — Flower-like pattern with center-heavy density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.actions import beam, receiver, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class RosetteScanConfig:
    s1_w1: float = 1.0
    s1_w2: float = 11.0
    s2_w1: float = 0.0
    s2_w2: float = 0.0
    duration: float = 10.0
    s2_hold_delay: float = 0.0
    radius: float | None = None


def parse_rosette_scan_config(data: dict) -> RosetteScanConfig:
    radius_raw = data.get("radius")
    radius = None if radius_raw is None else float(radius_raw) * 1e-3
    return RosetteScanConfig(
        s1_w1=float(data.get("s1_w1", 1.0)),
        s1_w2=float(data.get("s1_w2", 11.0)),
        s2_w1=float(data.get("s2_w1", 0.0)),
        s2_w2=float(data.get("s2_w2", 0.0)),
        duration=float(data.get("duration", 10.0)),
        s2_hold_delay=float(data.get("s2_hold_delay", 0.0)),
        radius=radius,
    )


@register_strategy("rosette_scan", parse_rosette_scan_config)
class RosetteScanStrategy(SearchStrategy):
    def __init__(self, config: RosetteScanConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> RosetteScanStrategy:
        return cls(config=config.strategy.params.get("rosette_scan", RosetteScanConfig()))

    def build_script(self, ctx: StrategyContext):
        from strategy.actions import hold
        from strategy.movements import Rosette

        max_radius = (
            self.config.radius
            if self.config.radius is not None
            else ctx.config.simulation.max_search_radius
        )
        timeout = ctx.config.simulation.timeout

        script = strategy(self.name)

        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                Rosette(A=max_radius, w1=self.config.s1_w1, w2=self.config.s1_w2),
                duration=timeout,
                label="S1 rosette scan"
            )

        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            hold_delay = min(self.config.s2_hold_delay, timeout)
            if hold_delay > 0.0:
                hold(duration=hold_delay, label="S2 hold")
            remaining = timeout - hold_delay
            if remaining > 0.0:
                script._builders["S2"].movement(
                    Rosette(A=max_radius, w1=self.config.s2_w1, w2=self.config.s2_w2),
                    duration=remaining,
                    label="S2 rosette scan"
                )

        return script.build()
