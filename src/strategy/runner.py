"""Frame-based runner for independent satellite timelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from strategy.actions import (
    HardwareStep,
    MovementStep,
    SatelliteTimeline,
    StrategyScript,
)
from strategy.base import StrategyContext, link_established
from strategy.movements import AimContext, Reset, build_aim_context


class HardwareSnapshot(TypedDict):
    s1_beam: bool
    s1_receiver: bool
    s2_beam: bool
    s2_receiver: bool


@dataclass
class FrameRunResult:
    success: bool
    hit_at_t: float | None
    global_t_end: float
    metadata: dict = field(default_factory=dict)


@dataclass
class FrameStepResult:
    visible_12: bool
    visible_21: bool
    locked: bool
    events: list[str] = field(default_factory=list)


@dataclass
class SatelliteRuntime:
    timeline: SatelliteTimeline
    beam_enabled: bool = True
    receiver_enabled: bool = True
    next_hardware_index: int = 0
    current_movement_index: int | None = None
    current_aim_step_index: int | None = None
    aim_ctx: AimContext | None = None
    acquisition_hold: bool = False

    def movement_at(self, local_t: float) -> tuple[MovementStep, float]:
        steps = self.timeline.movement_steps
        if not steps:
            raise RuntimeError(f"{self.timeline.name} timeline has no movement steps")
        t = min(max(local_t, 0.0), self.timeline.total_duration)
        
        idx = self.current_movement_index or 0
        if idx < len(steps):
            step = steps[idx]
            if step.start - 1e-9 <= t < step.end - 1e-9:
                return step, t - step.start
                
        for i in range(idx, len(steps)):
            step = steps[i]
            if t < step.end - 1e-9:
                self.current_movement_index = i
                return step, t - step.start
                
        for i in range(len(steps)):
            step = steps[i]
            if t < step.end - 1e-9:
                self.current_movement_index = i
                return step, t - step.start
                
        last = steps[-1]
        self.current_movement_index = len(steps) - 1
        return last, last.duration

    def apply_hardware(self, local_t: float) -> list[str]:
        if self.acquisition_hold:
            return []
        events: list[str] = []
        steps = self.timeline.hardware_steps
        while self.next_hardware_index < len(steps):
            step = steps[self.next_hardware_index]
            if step.time > local_t + 1e-12:
                break
            if step.target == "beam":
                self.beam_enabled = step.enabled
            else:
                self.receiver_enabled = step.enabled
            state = "enabled" if step.enabled else "disabled"
            events.append(f"{self.timeline.name} {step.target} {state}")
            self.next_hardware_index += 1
        return events


class FrameRunner:
    def __init__(self, strategy_ctx: StrategyContext) -> None:
        self.ctx = strategy_ctx

    def execute(
        self,
        script: StrategyScript,
        *,
        global_t_start: float = 0.0,
        stop_on_lock: bool = True,
    ) -> FrameRunResult:
        t_step = self.ctx.t_step
        runtime = self.begin(script)
        local_t = 0.0
        all_events: list[str] = []

        while local_t <= script.total_duration + 1e-12:
            # Optimization: inline step logic to avoid object creation
            s1_runtime = runtime["S1"]
            s2_runtime = runtime["S2"]
            
            s1_hw = s1_runtime.apply_hardware(local_t)
            s2_hw = s2_runtime.apply_hardware(local_t)
            if s1_hw: all_events.extend(s1_hw)
            if s2_hw: all_events.extend(s2_hw)

            aim1, events1 = self._apply_satellite(
                self.ctx.s1,
                self.ctx.s2.position,
                s1_runtime,
                local_t,
                t_step,
            )
            aim2, events2 = self._apply_satellite(
                self.ctx.s2,
                self.ctx.s1.position,
                s2_runtime,
                local_t,
                t_step,
            )
            if events1: all_events.extend(events1)
            if events2: all_events.extend(events2)

            # _evaluate_lock return (locked, more_events)
            locked, ev = self._evaluate_lock_fast(
                aim1,
                aim2,
                s1_runtime.beam_enabled,
                s1_runtime.receiver_enabled,
                s2_runtime.beam_enabled,
                s2_runtime.receiver_enabled,
                t_step,
            )
            if ev: all_events.extend(ev)

            if locked and stop_on_lock:
                return FrameRunResult(
                    success=True,
                    hit_at_t=global_t_start + local_t,
                    global_t_end=global_t_start + local_t + t_step,
                    metadata={
                        "direction": "both",
                        "strategy_name": script.strategy_name,
                        "events": all_events,
                        "hardware": self._hardware_snapshot(runtime),
                        "elapsed_t": local_t + t_step,
                    },
                )
            if local_t >= script.total_duration - 1e-12:
                break
            local_t += t_step

        return FrameRunResult(
            success=False,
            hit_at_t=None,
            global_t_end=global_t_start + script.total_duration,
            metadata={
                "strategy_name": script.strategy_name,
                "events": all_events,
                "hardware": self._hardware_snapshot(runtime),
                "elapsed_t": script.total_duration,
            },
        )

    def _evaluate_lock_fast(
        self,
        aim1,
        aim2,
        s1_beam: bool,
        s1_receiver: bool,
        s2_beam: bool,
        s2_receiver: bool,
        t_step: float,
    ) -> tuple[bool, list[str]]:
        events: list[str] = []
        visible_12 = (
            s1_beam
            and s2_receiver
            and link_established(self.ctx.s1, self.ctx.s2, aim1, self.ctx.config)
        )
        visible_21 = (
            s2_beam
            and s1_receiver
            and link_established(self.ctx.s2, self.ctx.s1, aim2, self.ctx.config)
        )

        if not visible_12 and not visible_21:
            return False, events

        if visible_12 and not self.ctx.s2.receiver.has_seen_beam:
            events.append("S2 acquisition started")
        if visible_21 and not self.ctx.s1.receiver.has_seen_beam:
            events.append("S1 acquisition started")

        s2_updated = False
        s1_updated = False

        if visible_12:
            # Use dish_boresight instead of geometry_snapshot for speed
            _, acq_events2 = self.ctx.s2.receiver.observe_beam(
                True,
                self.ctx.s1.position,
                t_step,
                dish_at_step_start=self.ctx.s2.receiver.dish_boresight,
            )
            for event in acq_events2:
                if event == "Slew complete":
                    events.append("S2 slew complete")
            s2_updated = True
            
        if visible_21:
            _, acq_events1 = self.ctx.s1.receiver.observe_beam(
                True,
                self.ctx.s2.position,
                t_step,
                dish_at_step_start=self.ctx.s1.receiver.dish_boresight,
            )
            for event in acq_events1:
                if event == "Slew complete":
                    events.append("S1 slew complete")
            s1_updated = True

        if s1_updated:
            final_12 = (
                s1_beam
                and s2_receiver
                and link_established(
                    self.ctx.s1,
                    self.ctx.s2,
                    self.ctx.s1.receiver.fsm.effective_receive_boresight(
                        self.ctx.s1.bench.bench_boresight
                    ),
                    self.ctx.config,
                )
            )
        else:
            final_12 = visible_12

        if s2_updated:
            final_21 = (
                s2_beam
                and s1_receiver
                and link_established(
                    self.ctx.s2,
                    self.ctx.s1,
                    self.ctx.s2.receiver.fsm.effective_receive_boresight(
                        self.ctx.s2.bench.bench_boresight
                    ),
                    self.ctx.config,
                )
            )
        else:
            final_21 = visible_21

        locked = (
            final_12
            and final_21
            and self.ctx.s1.bench.acquisition.slew_complete
            and self.ctx.s2.bench.acquisition.slew_complete
        )
        if locked:
            events.append("Mutual lock")
        return locked, events

    def begin(self, script: StrategyScript) -> dict[str, SatelliteRuntime]:
        return {
            "S1": self._begin_satellite_runtime(script.s1, self.ctx.s1),
            "S2": self._begin_satellite_runtime(script.s2, self.ctx.s2),
        }

    def _begin_satellite_runtime(self, timeline, sat) -> SatelliteRuntime:
        runtime = SatelliteRuntime(timeline)
        if not sat.receiver.has_seen_beam:
            return runtime
        runtime.acquisition_hold = True
        frozen = (
            self.ctx.s1_frozen_hardware
            if sat.name == "S1"
            else self.ctx.s2_frozen_hardware
        )
        if frozen is not None:
            runtime.beam_enabled, runtime.receiver_enabled = frozen
        runtime.next_hardware_index = len(timeline.hardware_steps)
        return runtime

    def step(
        self,
        runtime: dict[str, SatelliteRuntime],
        local_t: float,
        t_step: float,
    ) -> FrameStepResult:
        events: list[str] = []
        s1_runtime = runtime["S1"]
        s2_runtime = runtime["S2"]
        
        s1_hw = s1_runtime.apply_hardware(local_t)
        s2_hw = s2_runtime.apply_hardware(local_t)
        if s1_hw: events.extend(s1_hw)
        if s2_hw: events.extend(s2_hw)

        aim1, events1 = self._apply_satellite(
            self.ctx.s1,
            self.ctx.s2.position,
            s1_runtime,
            local_t,
            t_step,
        )
        aim2, events2 = self._apply_satellite(
            self.ctx.s2,
            self.ctx.s1.position,
            s2_runtime,
            local_t,
            t_step,
        )
        if events1: events.extend(events1)
        if events2: events.extend(events2)

        locked, ev = self._evaluate_lock_fast(
            aim1,
            aim2,
            s1_runtime.beam_enabled,
            s1_runtime.receiver_enabled,
            s2_runtime.beam_enabled,
            s2_runtime.receiver_enabled,
            t_step,
        )
        if ev: events.extend(ev)

        # Re-check visibility for compatibility with FrameStepResult
        # (This is slightly slow but only used by tests and non-inlined callers)
        v12 = s1_runtime.beam_enabled and s2_runtime.receiver_enabled and link_established(
            self.ctx.s1, self.ctx.s2, self.ctx.s1.bench.bench_boresight, self.ctx.config
        )
        v21 = s2_runtime.beam_enabled and s1_runtime.receiver_enabled and link_established(
            self.ctx.s2, self.ctx.s1, self.ctx.s2.bench.bench_boresight, self.ctx.config
        )

        return FrameStepResult(
            visible_12=v12,
            visible_21=v21,
            locked=locked,
            events=events,
        )

    def _hold_or_track(
        self,
        sat,
        partner_position,
        t_step: float,
        events: list[str],
    ):
        if sat.receiver.has_seen_beam:
            _, acq_events = sat.receiver.observe_beam(
                False,
                partner_position,
                t_step,
            )
            for event in acq_events:
                if event == "Slew complete":
                    events.append(f"{sat.name} slew complete")
            return sat.receiver.fsm.effective_receive_boresight(sat.bench.bench_boresight)
        return sat.bench.bench_boresight

    def _apply_satellite(
        self,
        sat,
        partner_position,
        runtime: SatelliteRuntime,
        local_t: float,
        t_step: float,
    ) -> tuple[object, list[str]]:
        events: list[str] = []
        if sat.receiver.has_seen_beam:
            aim = self._hold_or_track(sat, partner_position, t_step, events)
            return aim, events

        step, step_t = runtime.movement_at(local_t)
        step_events = self._ensure_movement_context(sat, runtime, step)
        events.extend(step_events)
        assert runtime.aim_ctx is not None
        aim = step.movement.aim_at(step_t, step.duration, runtime.aim_ctx)
        sat.bench.set_bench_aim(aim)
        return sat.bench.bench_boresight, events

    def _ensure_movement_context(
        self,
        sat,
        runtime: SatelliteRuntime,
        step: MovementStep,
    ) -> list[str]:
        if runtime.current_aim_step_index == step.index:
            return []
        runtime.current_aim_step_index = step.index
        runtime.aim_ctx = build_aim_context(
            sat,
            reset=isinstance(step.movement, Reset),
        )
        label = step.label or step.movement.__class__.__name__.lower()
        return [f"{runtime.timeline.name} step: {label}"]

    @staticmethod
    def _hardware_snapshot(
        runtime: dict[str, SatelliteRuntime],
    ) -> HardwareSnapshot:
        return {
            "s1_beam": runtime["S1"].beam_enabled,
            "s1_receiver": runtime["S1"].receiver_enabled,
            "s2_beam": runtime["S2"].beam_enabled,
            "s2_receiver": runtime["S2"].receiver_enabled,
        }
