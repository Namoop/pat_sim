"""Comprehensive search strategy (stub)."""

from __future__ import annotations

from satellite.strategy.actions import hold, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, StrategyResult


class ComprehensiveStrategy(SearchStrategy):
    name = "comprehensive"

    def build_script(self, ctx: StrategyContext):
        script = strategy(self.name)
        with script.satellite("S1"):
            hold(duration=max(ctx.q_step, 1e-9))
        with script.satellite("S2"):
            hold(duration=max(ctx.q_step, 1e-9))
        return script.build()

    def try_run(self, ctx: StrategyContext, global_q_start: float) -> StrategyResult:
        script = self.build_script(ctx)
        return StrategyResult(
            success=False,
            strategy_name=self.name,
            hit_at_q=None,
            script=script,
            elapsed_q=script.total_duration,
            skipped_reason="not_implemented",
        )
