"""Meta-strategy chain runner."""

from __future__ import annotations

from dataclasses import dataclass, field

from satellite.config import ScenarioConfig, StrategyConfig
from satellite.strategy.actions import StrategyScript
from satellite.strategy.base import SearchStrategy, StrategyContext, StrategyResult
from satellite.strategy.schedule import LegSchedule, compile_trace
from satellite.strategy.strategies import (
    AsymmetricProbeStrategy,
    ComprehensiveStrategy,
    MinorOffsetStrategy,
    SingleMissStrategy,
)


@dataclass
class MetaStrategyResult:
    success: bool
    hit_at_q: float | None
    winning_strategy: str | None
    attempts: list[StrategyResult]
    schedule: LegSchedule
    s1: object
    s2: object
    lock_direction: str | None = None
    metadata: dict = field(default_factory=dict)


class MetaStrategy:
    def __init__(self, strategies: list[SearchStrategy]) -> None:
        self.strategies = strategies

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> MetaStrategy:
        sc = config.strategy
        w = sc.spiral_w(config.satellite)
        k = sc.k
        dish_fov = config.satellite.dish_fov

        registry: dict[str, SearchStrategy] = {
            "minor_offset": MinorOffsetStrategy(
                duration=sc.minor_offset.duration,
                max_spiral_radius=StrategyConfig.resolve_radius(
                    sc.minor_offset.max_spiral_radius,
                    dish_fov,
                ),
                spiral_speed=sc.minor_offset.spiral_speed,
                w=w,
                k=k,
            ),
            "single_miss": SingleMissStrategy(
                phase1_duration=sc.single_miss.phase1_duration,
                a_spiral_radius=StrategyConfig.resolve_radius(
                    sc.single_miss.a_spiral_radius,
                    dish_fov,
                ),
                reset_duration=sc.single_miss.reset_duration,
                phase2_duration=sc.single_miss.phase2_duration,
                b_spiral_radius=StrategyConfig.resolve_radius(
                    sc.single_miss.b_spiral_radius,
                    dish_fov,
                ),
                spiral_speed=sc.single_miss.spiral_speed,
                w=w,
                k=k,
            ),
            "asymmetric_probe": AsymmetricProbeStrategy(
                probe_duration=sc.asymmetric_probe.probe_duration,
                spiral_radius=StrategyConfig.resolve_radius(
                    sc.asymmetric_probe.spiral_radius,
                    dish_fov,
                ),
                spiral_speed=sc.asymmetric_probe.spiral_speed,
                reset_duration=sc.asymmetric_probe.reset_duration,
                w=w,
                k=k,
            ),
            "comprehensive": ComprehensiveStrategy(),
        }

        chain = [
            registry[name]
            for name in sc.chain
            if name in registry
        ]
        return cls(chain)

    def run(self, ctx: StrategyContext) -> MetaStrategyResult:
        attempts: list[StrategyResult] = []
        scripts: list[StrategyScript] = []
        global_q = 0.0
        winning: StrategyResult | None = None
        attempt_ctx = ctx.clone_fresh()

        for strategy in self.strategies:
            result = strategy.try_run(attempt_ctx, global_q_start=global_q)
            attempts.append(result)
            scripts.append(result.script)
            global_q += result.elapsed_q

            if result.success:
                winning = result
                break

            end_hardware = result.metadata.get("hardware")
            attempt_ctx = attempt_ctx.clone_for_next_attempt(
                end_hardware=end_hardware,
            )

        schedule = compile_trace(scripts)
        final_ctx = attempt_ctx

        if winning is not None:
            return MetaStrategyResult(
                success=True,
                hit_at_q=winning.hit_at_q,
                winning_strategy=winning.strategy_name,
                attempts=attempts,
                schedule=schedule,
                s1=final_ctx.s1,
                s2=final_ctx.s2,
                lock_direction=winning.metadata.get("direction"),
                metadata=dict(winning.metadata),
            )

        return MetaStrategyResult(
            success=False,
            hit_at_q=None,
            winning_strategy=None,
            attempts=attempts,
            schedule=schedule,
            s1=final_ctx.s1,
            s2=final_ctx.s2,
        )
