"""Strategy 7: Lissajous Scan — Sinusoidal orthogonal scanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.actions import beam, receiver, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class LissajousScanConfig:
    s1_wx: float = 1.0
    s1_wy: float = 1.41421356
    s1_delta: float = 0.0
    s2_wx: float = 1.0
    s2_wy: float = 1.41421356
    s2_delta: float = 0.0
    s2_hold_delay: float = 0.0
    radius: float | None = None


def parse_lissajous_scan_config(data: dict) -> LissajousScanConfig:
    radius_raw = data.get("radius")
    radius = None if radius_raw is None else float(radius_raw) * 1e-3
    return LissajousScanConfig(
        s1_wx=float(data.get("s1_wx", data.get("wx", 1.0))),
        s1_wy=float(data.get("s1_wy", data.get("wy", 1.41421356))),
        s1_delta=float(data.get("s1_delta", data.get("delta", 0.0))),
        s2_wx=float(data.get("s2_wx", 1.0)),
        s2_wy=float(data.get("s2_wy", 1.41421356)),
        s2_delta=float(data.get("s2_delta", 0.0)),
        s2_hold_delay=float(data.get("s2_hold_delay", 0.0)),
        radius=radius,
    )


@register_strategy("lissajous_scan", parse_lissajous_scan_config)
class LissajousScanStrategy(SearchStrategy):
    def __init__(self, config: LissajousScanConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> LissajousScanStrategy:
        return cls(config=config.strategy.params.get("lissajous_scan", LissajousScanConfig()))

    def build_script(self, ctx: StrategyContext):
        from strategy.actions import hold
        from strategy.movements import Lissajous
        import math

        max_radius = (
            self.config.radius
            if self.config.radius is not None
            else ctx.config.simulation.max_search_radius
        )
        A = max_radius / math.sqrt(2.0)
        timeout = ctx.config.simulation.timeout

        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                Lissajous(
                    A=A,
                    wx=self.config.s1_wx,
                    wy=self.config.s1_wy,
                    delta=self.config.s1_delta,
                ),
                duration=timeout,
                label="S1 lissajous scan"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            hold_delay = min(self.config.s2_hold_delay, timeout)
            if hold_delay > 0.0:
                hold(duration=hold_delay, label="S2 hold")
            remaining = timeout - hold_delay
            if remaining > 0.0:
                script._builders["S2"].movement(
                    Lissajous(
                        A=A,
                        wx=self.config.s2_wx,
                        wy=self.config.s2_wy,
                        delta=self.config.s2_delta,
                    ),
                    duration=remaining,
                    label="S2 lissajous scan"
                )
        return script.build()
