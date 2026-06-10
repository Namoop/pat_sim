"""Comprehensive search strategy (stub)."""

from __future__ import annotations

from satellite.strategy.actions import ActionScript
from satellite.strategy.base import SearchStrategy, StrategyContext, StrategyResult


class ComprehensiveStrategy(SearchStrategy):
    name = "comprehensive"

    def build_script(self, ctx: StrategyContext) -> ActionScript:
        return ActionScript(epochs=())

    def try_run(self, ctx: StrategyContext, global_q_start: float) -> StrategyResult:
        return StrategyResult(
            success=False,
            strategy_name=self.name,
            hit_at_q=None,
            script=ActionScript(epochs=(), strategy_name=self.name),
            elapsed_q=0.0,
            skipped_reason="not_implemented",
        )
