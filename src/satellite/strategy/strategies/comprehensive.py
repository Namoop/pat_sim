"""Comprehensive search strategy (stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from satellite.strategy.actions import hold, strategy
from satellite.strategy.base import (
    SearchStrategy,
    StrategyContext,
    StrategyResult,
    register_strategy,
)

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@register_strategy("comprehensive")
class ComprehensiveStrategy(SearchStrategy):
    @classmethod
    def from_config(cls, config: ScenarioConfig) -> ComprehensiveStrategy:
        return cls()

    def build_script(self, ctx: StrategyContext):
        # Placeholder duration
        duration = max(ctx.config.simulation.t_step, 0.1)

        script = strategy(self.name)
        with script.satellite("S1"):
            hold(duration=duration)
        with script.satellite("S2"):
            hold(duration=duration)
        return script.build()

    def try_run(self, ctx: StrategyContext, global_t_start: float) -> StrategyResult:
        script = self.build_script(ctx)
        return StrategyResult(
            success=False,
            strategy_name=self.name,
            hit_at_t=None,
            script=script,
            elapsed_t=script.total_duration,
            skipped_reason="not_implemented",
        )
