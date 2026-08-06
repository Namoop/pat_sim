"""Independent satellite timelines and strategy authoring DSL."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterator, Literal

from satellite.math.math3d import angle_between
from strategy.movements import (
    Circle,
    Grid,
    Hold,
    Line,
    MovementPattern,
    Reset,
    Spiral,
    build_aim_context,
)

if TYPE_CHECKING:
    from strategy.base import StrategyContext

SatelliteName = Literal["S1", "S2"]
HardwareTarget = Literal["beam", "receiver"]

_DURATION_TOLERANCE = 1e-9
_ANGULAR_TOLERANCE = 1e-7
_current_timeline: ContextVar[object | None] = ContextVar(
    "current_timeline",
    default=None,
)


@dataclass(frozen=True)
class MovementStep:
    start: float
    duration: float
    movement: MovementPattern
    index: int
    label: str = ""

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class HardwareStep:
    time: float
    target: HardwareTarget
    enabled: bool
    index: int
    label: str = ""


@dataclass(frozen=True)
class SatelliteTimeline:
    name: SatelliteName
    movement_steps: tuple[MovementStep, ...]
    hardware_steps: tuple[HardwareStep, ...]
    total_duration: float

    def movement_at(self, local_t: float) -> tuple[MovementStep, float]:
        if not self.movement_steps:
            raise RuntimeError(f"{self.name} timeline has no movement steps")
        t = min(max(local_t, 0.0), self.total_duration)
        for step in self.movement_steps:
            if t < step.end - _DURATION_TOLERANCE:
                return step, t - step.start
        last = self.movement_steps[-1]
        return last, last.duration

    def hardware_state_at(self, local_t: float) -> tuple[bool, bool]:
        beam_enabled = True
        receiver_enabled = True
        for step in self.hardware_steps:
            if step.time > local_t + _DURATION_TOLERANCE:
                break
            if step.target == "beam":
                beam_enabled = step.enabled
            else:
                receiver_enabled = step.enabled
        return beam_enabled, receiver_enabled


@dataclass(frozen=True)
class StrategyScript:
    s1: SatelliteTimeline
    s2: SatelliteTimeline
    strategy_name: str = ""

    @property
    def total_duration(self) -> float:
        return self.s1.total_duration

    def validate_slew_speed(
        self,
        ctx: StrategyContext,
        *,
        strict: bool = False,
    ) -> list[str]:
        """Verify all point-to-point transitions and internal movement speeds are physical."""
        errors: list[str] = []
        max_beam_speed = ctx.config.satellite.max_beam_speed
        if max_beam_speed <= 0:
            return []

        # We need to simulate the state to know where the bench is
        check_ctx = ctx.clone_fresh()
        for timeline, sat in [
            (self.s1, check_ctx.s1),
            (self.s2, check_ctx.s2),
        ]:
            aim_ctx = None
            prev_end_aim = sat.bench.bench_boresight.copy()

            fastest_speed = 0

            for step in timeline.movement_steps:
                # Transition check: move from prev_end_aim to step's start aim
                if aim_ctx is None or isinstance(step.movement, Reset):
                    aim_ctx = build_aim_context(
                        sat, reset=isinstance(step.movement, Reset)
                    )
                
                start_aim = step.movement.aim_at(0.0, step.duration, aim_ctx)
                dist = angle_between(prev_end_aim, start_aim)
                
                # If there's a jump, it must happen in "zero" time which is impossible
                # unless the jump distance is 0. 
                # Practically, we allow a small buffer or assume the jump takes 
                # some of the step's duration.
                if dist > _ANGULAR_TOLERANCE:
                    msg = (
                        f"{timeline.name} jump of {math.degrees(dist):.3f} deg "
                        f"detected at start of '{step.label}' (t={step.start:.2f})"
                    )
                    if strict:
                        errors.append(msg)
                        raise ValueError(msg)

                # Continuous internal speed check within the step
                if step.duration > 0.0:
                    n_samples = 100
                    dt = step.duration / n_samples
                    last_aim = start_aim
                    enforce = ctx.config.simulation.enforce_speed_limit
                    #fastest_speed = 0
                    for i in range(1, n_samples + 1):
                        t_curr = i * dt
                        curr_aim = step.movement.aim_at(t_curr, step.duration, aim_ctx)
                        step_dist = angle_between(last_aim, curr_aim)
                        step_speed = step_dist / dt
                        if step_speed > fastest_speed:
                            fastest_speed = step_speed

                        # 1.05 tolerance to account for discretization/floating point variations
                        if step_speed > max_beam_speed * 1.05:
                            msg = (
                                f"{timeline.name} movement '{step.label}' requires speed "
                                f"{step_speed:.4f} rad/s (at t={step.start + t_curr:.2f}s), "
                                f"which exceeds the maximum physical beam speed limit "
                                f"of {max_beam_speed:.4f} rad/s."
                            )
                            errors.append(msg)
                            if enforce:
                                raise ValueError(msg)
                            print(f"WARNING: {msg}")
                            break  # record once per step, then move on
                        last_aim = curr_aim

                prev_end_aim = step.movement.aim_at(step.duration, step.duration, aim_ctx)
                sat.bench.set_bench_aim(prev_end_aim)
                aim_ctx = build_aim_context(sat)
            
            print("Fastest speed: %.3g" % (fastest_speed*1000))

        return errors


def validate_movement_durations(ctx: StrategyContext, script: StrategyScript) -> None:
    """Validate movement step durations against timeline and bench state."""
    check_ctx = ctx.clone_fresh()
    pairs = (
        (script.s1, ctx.s1, check_ctx.s1),
        (script.s2, ctx.s2, check_ctx.s2),
    )
    for timeline, live, sat in pairs:
        if live.receiver.has_seen_beam:
            continue
        aim_ctx = None
        for step in timeline.movement_steps:
            if step.duration < 0.0:
                raise ValueError(
                    f"{timeline.name} movement duration must be non-negative; "
                    f"got {step.duration!r}"
                )
            if step.duration == 0.0:
                if not isinstance(step.movement, Reset):
                    raise ValueError(
                        f"{timeline.name} movement {step.label!r} requires positive "
                        f"duration; got 0"
                    )
                target = sat.bench.initial_boresight
                current = sat.bench.bench_boresight
                if angle_between(current, target) > _ANGULAR_TOLERANCE:
                    raise ValueError(
                        f"{timeline.name} reset step {step.label!r} has duration=0 but "
                        f"bench is not at initial boresight; set an explicit duration "
                        f"or wait for auto-compute from beam-director max slew rate"
                    )
                continue

            if aim_ctx is None or isinstance(step.movement, Reset):
                aim_ctx = build_aim_context(
                    sat,
                    reset=isinstance(step.movement, Reset),
                )
            end_aim = step.movement.aim_at(step.duration, step.duration, aim_ctx)
            sat.bench.set_bench_aim(end_aim)
            aim_ctx = build_aim_context(sat)


class TimelineBuilder:
    def __init__(self, name: SatelliteName) -> None:
        self.name = name
        self._cursor = 0.0
        self._index = 0
        self._movements: list[MovementStep] = []
        self._hardware: list[HardwareStep] = []

    @property
    def total_duration(self) -> float:
        return self._cursor

    def movement(
        self,
        movement: MovementPattern,
        *,
        duration: float,
        label: str = "",
    ) -> None:
        if duration < 0.0:
            raise ValueError(
                f"{self.name} movement duration must be non-negative; got {duration!r}"
            )
        if duration == 0.0 and not isinstance(movement, Reset):
            raise ValueError(
                f"{self.name} movement duration must be positive; got {duration!r}"
            )
        self._movements.append(
            MovementStep(
                start=self._cursor,
                duration=float(duration),
                movement=movement,
                index=self._index,
                label=label,
            )
        )
        self._cursor += float(duration)
        self._index += 1

    def hardware(
        self,
        target: HardwareTarget,
        enabled: bool,
        *,
        label: str = "",
    ) -> None:
        self._hardware.append(
            HardwareStep(
                time=self._cursor,
                target=target,
                enabled=enabled,
                index=self._index,
                label=label,
            )
        )
        self._index += 1

    def build(self) -> SatelliteTimeline:
        if not self._movements:
            raise ValueError(f"{self.name} timeline must contain at least one movement")
        return SatelliteTimeline(
            name=self.name,
            movement_steps=tuple(self._movements),
            hardware_steps=tuple(self._hardware),
            total_duration=self._cursor,
        )


class StrategyBuilder:
    def __init__(self, name: str) -> None:
        self.name = name
        self._builders: dict[SatelliteName, TimelineBuilder] = {
            "S1": TimelineBuilder("S1"),
            "S2": TimelineBuilder("S2"),
        }

    @contextmanager
    def satellite(self, name: SatelliteName) -> Iterator[TimelineBuilder]:
        if name not in self._builders:
            raise ValueError(f"Unknown satellite {name!r}; expected 'S1' or 'S2'")
        token = _current_timeline.set(self._builders[name])
        try:
            yield self._builders[name]
        finally:
            _current_timeline.reset(token)

    def build(self) -> StrategyScript:
        s1 = self._builders["S1"].build()
        s2 = self._builders["S2"].build()
        if abs(s1.total_duration - s2.total_duration) > _DURATION_TOLERANCE:
            raise ValueError(
                "Satellite timelines must have the same total duration: "
                f"S1={s1.total_duration:.12g}, S2={s2.total_duration:.12g}"
            )
        return StrategyScript(s1=s1, s2=s2, strategy_name=self.name)


def strategy(name: str) -> StrategyBuilder:
    return StrategyBuilder(name)


def _active_builder() -> TimelineBuilder:
    builder = _current_timeline.get()
    if builder is None:
        raise RuntimeError(
            "Strategy DSL calls must be made inside "
            "`with script.satellite('S1'):` or `with script.satellite('S2'):`"
        )
    return builder  # type: ignore[return-value]


def hold(*, duration: float, label: str = "hold") -> None:
    _active_builder().movement(Hold(), duration=duration, label=label)


def reset(*, duration: float, label: str = "reset") -> None:
    _active_builder().movement(Reset(), duration=duration, label=label)


def spiral(
    *,
    duration: float,
    w: float,
    k: float,
    max_radius: float,
    phase_offset: float = 0.0,
    label: str = "spiral",
) -> None:
    _active_builder().movement(
        Spiral(w=w, k=k, max_radius=max_radius, phase_offset=phase_offset),
        duration=duration,
        label=label,
    )


def circle(
    *,
    duration: float,
    radius: float,
    period: float | None = None,
    label: str = "circle",
) -> None:
    _active_builder().movement(
        Circle(radius=radius, period=period),
        duration=duration,
        label=label,
    )


def line(
    *,
    duration: float,
    extent: float,
    axis_angle: float = 0.0,
    label: str = "line",
) -> None:
    _active_builder().movement(
        Line(extent=extent, axis_angle=axis_angle),
        duration=duration,
        label=label,
    )


def grid(
    *,
    duration: float,
    spacing: float,
    extent: float,
    label: str = "grid",
) -> None:
    _active_builder().movement(
        Grid(spacing=spacing, extent=extent),
        duration=duration,
        label=label,
    )


class _HardwareControl:
    def __init__(self, target: HardwareTarget) -> None:
        self.target = target

    def enable(self, *, label: str = "") -> None:
        _active_builder().hardware(
            self.target,
            True,
            label=label or f"{self.target} enable",
        )

    def disable(self, *, label: str = "") -> None:
        _active_builder().hardware(
            self.target,
            False,
            label=label or f"{self.target} disable",
        )


beam = _HardwareControl("beam")
receiver = _HardwareControl("receiver")
