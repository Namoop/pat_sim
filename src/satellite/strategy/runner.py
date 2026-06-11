"""Frame-based bidirectional lock runner."""

from __future__ import annotations

from dataclasses import dataclass, field

from satellite.strategy.actions import ActionScript, Epoch
from satellite.strategy.base import StrategyContext, link_established
from satellite.strategy.movements import AimContext, MovementPattern, Reset, build_aim_context


@dataclass
class FrameRunResult:
    success: bool
    hit_at_q: float | None
    global_q_end: float
    metadata: dict = field(default_factory=dict)


@dataclass
class _SatEpochState:
    ctx: AimContext


class FrameRunner:
    def __init__(self, strategy_ctx: StrategyContext) -> None:
        self.ctx = strategy_ctx

    def execute(
        self,
        script: ActionScript,
        *,
        global_q_start: float = 0.0,
        stop_on_lock: bool = True,
    ) -> FrameRunResult:
        q_step = self.ctx.q_step
        global_q = global_q_start

        for epoch_index, epoch in enumerate(script.epochs):
            result = self._run_epoch(
                epoch,
                epoch_index=epoch_index,
                strategy_name=script.strategy_name,
                global_q=global_q,
                q_step=q_step,
                stop_on_lock=stop_on_lock,
            )
            if result.success and stop_on_lock:
                return result
            global_q += epoch.duration

        return FrameRunResult(
            success=False,
            hit_at_q=None,
            global_q_end=global_q,
        )

    def _run_epoch(
        self,
        epoch: Epoch,
        *,
        epoch_index: int,
        strategy_name: str,
        global_q: float,
        q_step: float,
        stop_on_lock: bool,
    ) -> FrameRunResult:
        s1_state = self._begin_satellite(self.ctx.s1, epoch.s1.movement)
        s2_state = self._begin_satellite(self.ctx.s2, epoch.s2.movement)

        local_t = 0.0
        while local_t < epoch.duration - 1e-12:
            aim1 = self._apply_aim(
                self.ctx.s1,
                epoch.s1.movement,
                s1_state,
                local_t,
                epoch.duration,
            )
            aim2 = self._apply_aim(
                self.ctx.s2,
                epoch.s2.movement,
                s2_state,
                local_t,
                epoch.duration,
            )

            hit_12, hit_21 = self._bidirectional_lock(aim1, aim2, q_step)

            if hit_12 or hit_21:
                direction = (
                    "both"
                    if hit_12 and hit_21
                    else ("S1→S2" if hit_12 else "S2→S1")
                )
                if stop_on_lock:
                    return FrameRunResult(
                        success=True,
                        hit_at_q=global_q + local_t,
                        global_q_end=global_q + local_t + q_step,
                        metadata={
                            "direction": direction,
                            "epoch_index": epoch_index,
                            "strategy_name": strategy_name,
                            "epoch_label": epoch.label,
                        },
                    )

            local_t += q_step

        return FrameRunResult(
            success=False,
            hit_at_q=None,
            global_q_end=global_q + epoch.duration,
        )

    def _begin_satellite(self, sat, movement: MovementPattern) -> _SatEpochState:
        reset = isinstance(movement, Reset)
        aim_ctx = build_aim_context(sat, reset=reset)
        return _SatEpochState(ctx=aim_ctx)

    def _apply_aim(
        self,
        sat,
        movement: MovementPattern,
        state: _SatEpochState,
        local_t: float,
        duration: float,
    ):
        if sat.receiver.has_seen_beam:
            return sat.bench.bench_boresight.copy()
        aim = movement.aim_at(local_t, duration, state.ctx)
        sat.bench.set_bench_aim(aim)
        return sat.bench.bench_boresight.copy()

    def _bidirectional_lock(self, aim1, aim2, q_step: float) -> tuple[bool, bool]:
        cfg = self.ctx.config
        s1 = self.ctx.s1
        s2 = self.ctx.s2

        geom1 = s1.receiver.geometry_snapshot()
        geom2 = s2.receiver.geometry_snapshot()

        hit_12 = link_established(s1, s2, aim1, cfg)
        hit_21 = link_established(s2, s1, aim2, cfg)

        if hit_12:
            s2.receiver.observe_beam(
                True,
                s1.position,
                q_step,
                dish_at_step_start=geom2.dish_boresight,
            )
        if hit_21:
            s1.receiver.observe_beam(
                True,
                s2.position,
                q_step,
                dish_at_step_start=geom1.dish_boresight,
            )

        return hit_12, hit_21
