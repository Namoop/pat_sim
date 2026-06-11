"""Frame-based runner for independent satellite timelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from satellite.strategy.actions import (
    HardwareStep,
    MovementStep,
    SatelliteTimeline,
    StrategyScript,
)
from satellite.strategy.base import StrategyContext, link_established
from satellite.strategy.movements import AimContext, Reset, build_aim_context


class HardwareSnapshot(TypedDict):
    s1_beam: bool
    s1_receiver: bool
    s2_beam: bool
    s2_receiver: bool


@dataclass
class FrameRunResult:
    success: bool
    hit_at_q: float | None
    global_q_end: float
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
    aim_ctx: AimContext | None = None
    acquisition_hold: bool = False

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
        global_q_start: float = 0.0,
        stop_on_lock: bool = True,
    ) -> FrameRunResult:
        q_step = self.ctx.q_step
        runtime = self.begin(script)
        local_t = 0.0
        all_events: list[str] = []

        while local_t <= script.total_duration + 1e-12:
            result = self.step(runtime, local_t, q_step)
            all_events.extend(result.events)
            if result.locked and stop_on_lock:
                return FrameRunResult(
                    success=True,
                    hit_at_q=global_q_start + local_t,
                    global_q_end=global_q_start + local_t + q_step,
                    metadata={
                        "direction": "both",
                        "strategy_name": script.strategy_name,
                        "events": all_events,
                        "hardware": self._hardware_snapshot(runtime),
                        "elapsed_q": local_t + q_step,
                    },
                )
            if local_t >= script.total_duration - 1e-12:
                break
            local_t += q_step

        return FrameRunResult(
            success=False,
            hit_at_q=None,
            global_q_end=global_q_start + script.total_duration,
            metadata={
                "strategy_name": script.strategy_name,
                "events": all_events,
                "hardware": self._hardware_snapshot(runtime),
                "elapsed_q": script.total_duration,
            },
        )

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
        q_step: float,
    ) -> FrameStepResult:
        events: list[str] = []
        s1_runtime = runtime["S1"]
        s2_runtime = runtime["S2"]
        events.extend(s1_runtime.apply_hardware(local_t))
        events.extend(s2_runtime.apply_hardware(local_t))

        aim1, events1 = self._apply_satellite(
            self.ctx.s1,
            self.ctx.s2.position,
            s1_runtime,
            local_t,
            q_step,
        )
        aim2, events2 = self._apply_satellite(
            self.ctx.s2,
            self.ctx.s1.position,
            s2_runtime,
            local_t,
            q_step,
        )
        events.extend(events1)
        events.extend(events2)

        return self._evaluate_lock(
            aim1,
            aim2,
            s1_runtime.beam_enabled,
            s1_runtime.receiver_enabled,
            s2_runtime.beam_enabled,
            s2_runtime.receiver_enabled,
            q_step,
            events,
        )

    def _evaluate_lock(
        self,
        aim1,
        aim2,
        s1_beam: bool,
        s1_receiver: bool,
        s2_beam: bool,
        s2_receiver: bool,
        q_step: float,
        events: list[str],
    ) -> FrameStepResult:
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

        if visible_12 and not self.ctx.s2.receiver.has_seen_beam:
            events.append("S2 acquisition started")
        if visible_21 and not self.ctx.s1.receiver.has_seen_beam:
            events.append("S1 acquisition started")

        acq_events2: list[str] = []
        acq_events1: list[str] = []
        if visible_12:
            geom2 = self.ctx.s2.receiver.geometry_snapshot()
            _, acq_events2 = self.ctx.s2.receiver.observe_beam(
                True,
                self.ctx.s1.position,
                q_step,
                dish_at_step_start=geom2.dish_boresight,
            )
        if visible_21:
            geom1 = self.ctx.s1.receiver.geometry_snapshot()
            _, acq_events1 = self.ctx.s1.receiver.observe_beam(
                True,
                self.ctx.s2.position,
                q_step,
                dish_at_step_start=geom1.dish_boresight,
            )
        for event in acq_events2:
            if event == "Slew complete":
                events.append("S2 slew complete")
        for event in acq_events1:
            if event == "Slew complete":
                events.append("S1 slew complete")

        final_12 = (
            s1_beam
            and s2_receiver
            and link_established(
                self.ctx.s1,
                self.ctx.s2,
                self.ctx.s1.bench.bench_boresight,
                self.ctx.config,
            )
        )
        final_21 = (
            s2_beam
            and s1_receiver
            and link_established(
                self.ctx.s2,
                self.ctx.s1,
                self.ctx.s2.bench.bench_boresight,
                self.ctx.config,
            )
        )
        locked = (
            final_12
            and final_21
            and self.ctx.s1.bench.acquisition.slew_complete
            and self.ctx.s2.bench.acquisition.slew_complete
        )
        if locked:
            events.append("Mutual lock")
        return FrameStepResult(
            visible_12=final_12,
            visible_21=final_21,
            locked=locked,
            events=events,
        )

    def _hold_or_track(
        self,
        sat,
        partner_position,
        q_step: float,
        events: list[str],
    ):
        if sat.receiver.has_seen_beam:
            _, acq_events = sat.receiver.observe_beam(
                False,
                partner_position,
                q_step,
            )
            for event in acq_events:
                if event == "Slew complete":
                    events.append(f"{sat.name} slew complete")
            return sat.bench.bench_boresight.copy()
        return sat.bench.bench_boresight.copy()

    def _apply_satellite(
        self,
        sat,
        partner_position,
        runtime: SatelliteRuntime,
        local_t: float,
        q_step: float,
    ) -> tuple[object, list[str]]:
        events: list[str] = []
        if sat.receiver.has_seen_beam:
            aim = self._hold_or_track(sat, partner_position, q_step, events)
            return aim, events

        step, step_t = runtime.timeline.movement_at(local_t)
        step_events = self._ensure_movement_context(sat, runtime, step)
        events.extend(step_events)
        assert runtime.aim_ctx is not None
        aim = step.movement.aim_at(step_t, step.duration, runtime.aim_ctx)
        sat.bench.set_bench_aim(aim)
        return sat.bench.bench_boresight.copy(), events

    def _ensure_movement_context(
        self,
        sat,
        runtime: SatelliteRuntime,
        step: MovementStep,
    ) -> list[str]:
        if runtime.current_movement_index == step.index:
            return []
        runtime.current_movement_index = step.index
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
