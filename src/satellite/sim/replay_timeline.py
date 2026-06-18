"""Precomputed replay cache for O(1) viz scrubbing up to playable end."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from satellite.strategy.base import StrategyContext
from satellite.strategy.runner import FrameRunner
from satellite.strategy.schedule import ScheduledScript

if TYPE_CHECKING:
    from satellite.sim.scenario import ScenarioResult
    from satellite.physics.receiver import ReceiverSDA


@dataclass
class ReceiverSnapshot:
    bench_boresight: np.ndarray
    fsm_theta: float
    fsm_phi: float
    fsm_locked: bool
    fsm_track_target: np.ndarray | None
    has_seen_beam: bool
    incident_angle: float | None
    acq_track_target: np.ndarray | None
    bench_slew_rate: float | None
    acq_fsm_locked: bool
    slew_complete: bool


@dataclass
class _ReplayDriver:
    ctx: StrategyContext
    t_step: float
    runtime: dict | None = None
    current_key: tuple[str, int] | None = None
    global_t: float = 0.0

    def _ensure_script(self, scheduled: ScheduledScript) -> None:
        key = (scheduled.strategy_name, scheduled.attempt_index)
        if self.current_key == key:
            return
        if self.current_key is not None:
            if not self.ctx.s1.receiver.has_seen_beam:
                self.ctx.s1.receiver.reset_dish_tracking()
            if not self.ctx.s2.receiver.has_seen_beam:
                self.ctx.s2.receiver.reset_dish_tracking()
        self.current_key = key
        self.runtime = FrameRunner(self.ctx).begin(scheduled.script)

    def step_frame(self, scheduled: ScheduledScript, local_t: float):
        self._ensure_script(scheduled)
        assert self.runtime is not None
        return FrameRunner(self.ctx).step(self.runtime, local_t, self.t_step)


def _capture_receiver(rx: ReceiverSDA) -> ReceiverSnapshot:
    acq = rx.bench.acquisition
    fsm = rx.fsm
    return ReceiverSnapshot(
        bench_boresight=rx.bench.bench_boresight.copy(),
        fsm_theta=fsm.theta_offset,
        fsm_phi=fsm.phi_offset,
        fsm_locked=fsm.locked,
        fsm_track_target=(
            fsm.track_target.copy() if fsm.track_target is not None else None
        ),
        has_seen_beam=acq.has_seen_beam,
        incident_angle=acq.incident_angle,
        acq_track_target=(
            acq.track_target.copy() if acq.track_target is not None else None
        ),
        bench_slew_rate=acq.bench_slew_rate,
        acq_fsm_locked=acq.fsm_locked,
        slew_complete=acq.slew_complete,
    )


def _restore_receiver(rx: ReceiverSDA, snap: ReceiverSnapshot) -> None:
    bench = rx.bench
    fsm = rx.fsm
    acq = bench.acquisition

    bench.bench_boresight = snap.bench_boresight.copy()
    fsm.theta_offset = snap.fsm_theta
    fsm.phi_offset = snap.fsm_phi
    fsm.locked = snap.fsm_locked
    fsm.track_target = (
        snap.fsm_track_target.copy()
        if snap.fsm_track_target is not None
        else None
    )

    acq.has_seen_beam = snap.has_seen_beam
    acq.incident_angle = snap.incident_angle
    acq.track_target = (
        snap.acq_track_target.copy()
        if snap.acq_track_target is not None
        else None
    )
    acq.bench_slew_rate = snap.bench_slew_rate
    acq.fsm_locked = snap.acq_fsm_locked
    acq.slew_complete = snap.slew_complete
    bench._invalidate_geometry_cache()


@dataclass
class ReplayTimeline:
    t_values: np.ndarray
    s1_snapshots: list[ReceiverSnapshot]
    s2_snapshots: list[ReceiverSnapshot]
    hit_at_t: float | None
    event_steps: list[int] = field(default_factory=list)
    event_lines: list[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.t_values)

    @property
    def t_end(self) -> float:
        return float(self.t_values[-1]) if self.step_count else 0.0

    def index_for_t(self, t: float) -> int:
        if self.step_count == 0:
            return 0
        if self.step_count == 1:
            return 0
        step = float(self.t_values[1] - self.t_values[0])
        idx = int(np.floor((t + 1e-12) / step))
        return int(np.clip(idx, 0, self.step_count - 1))

    def memory_bytes(self) -> int:
        n = self.step_count
        vec_bytes = n * 3 * 8 * 4
        snap_bytes = n * 2 * 224
        event_bytes = sum(len(s.encode("utf-8")) for s in self.event_lines)
        return int(self.t_values.nbytes + vec_bytes + snap_bytes + event_bytes)

    def memory_summary(self) -> str:
        mb = self.memory_bytes() / (1024 * 1024)
        return f"{self.step_count} steps, ~{mb:.2f} MiB"

    def restore(self, result: ScenarioResult, t_end: float) -> int:
        idx = self.index_for_t(t_end)
        _restore_receiver(result.s1.receiver, self.s1_snapshots[idx])
        _restore_receiver(result.s2.receiver, self.s2_snapshots[idx])
        return idx

    def event_log_up_to(
        self,
        step_index: int,
        *,
        include_final: bool = False,
    ) -> list[str]:
        limit = self.step_count if include_final else step_index
        return [
            line
            for step, line in zip(self.event_steps, self.event_lines, strict=True)
            if step <= limit
        ]


def _append_event(timeline: ReplayTimeline, step_index: int, line: str) -> None:
    timeline.event_steps.append(step_index)
    timeline.event_lines.append(line)


def _format_sigfig(value: float, sigfigs: int = 3) -> str:
    return f"{value:.{sigfigs}g}"


def initial_conditions_lines(result: ScenarioResult) -> list[str]:
    """Event-log header lines for scenario geometry and bench offsets."""
    cfg = result.config
    s1 = cfg.s1
    s2 = cfg.s2
    distance_km = cfg.simulation.distance / 1000.0
    return [
        "Initial conditions:",
        f"  distance = {_format_sigfig(distance_km)} km",
        (
            f"  S1 bench theta={_format_sigfig(s1.bench_theta_offset * 1e3)} mrad"
            f" phi={_format_sigfig(s1.bench_phi_offset * 1e3)} mrad"
        ),
        (
            f"  S2 bench theta={_format_sigfig(s2.bench_theta_offset * 1e3)} mrad"
            f" phi={_format_sigfig(s2.bench_phi_offset * 1e3)} mrad"
        ),
    ]


def _append_initial_conditions(timeline: ReplayTimeline, result: ScenarioResult) -> None:
    for line in initial_conditions_lines(result):
        _append_event(timeline, 0, line)


def _fresh_context(result: ScenarioResult) -> StrategyContext:
    from satellite.physics.satellite import Satellite

    cfg = result.config
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    return StrategyContext(s1=s1, s2=s2, config=cfg)


def replay_to_t(
    result: ScenarioResult,
    t_end: float,
    *,
    event_log: list[str] | None = None,
) -> None:
    """Replay coupled simulation from t=0 through t_end."""
    t_end = float(np.clip(t_end, 0.0, result.playable_t_end))
    if result._replay_timeline is not None:
        timeline = result._replay_timeline
        idx = timeline.restore(result, t_end)
        if event_log is not None:
            event_log.clear()
            include_final = t_end >= result.playable_t_end - 1e-12
            event_log.extend(
                timeline.event_log_up_to(idx, include_final=include_final)
            )
        return

    if result._stepper is None:
        ctx = _fresh_context(result)
        result._stepper = _ReplayDriver(
            ctx=ctx,
            t_step=result.config.simulation.t_step,
        )
        result.s1.receiver.reset_dish_tracking()
        result.s2.receiver.reset_dish_tracking()
        ctx.s1.receiver.reset_dish_tracking()
        ctx.s2.receiver.reset_dish_tracking()

    driver: _ReplayDriver = result._stepper
    t_step = driver.t_step

    if abs(driver.global_t - t_end) < 1e-9:
        return

    if driver.global_t > t_end + 1e-12:
        driver.ctx = _fresh_context(result)
        driver.runtime = None
        driver.current_key = None
        driver.global_t = 0.0
        result.s1.receiver.reset_dish_tracking()
        result.s2.receiver.reset_dish_tracking()
        driver.ctx.s1.receiver.reset_dish_tracking()
        driver.ctx.s2.receiver.reset_dish_tracking()

    while True:
        scheduled, local_t = result.schedule.script_at(driver.global_t)
        driver.step_frame(scheduled, local_t)
        if driver.global_t >= t_end - 1e-12:
            break
        driver.global_t += t_step

    for src, dst in (
        (driver.ctx.s1, result.s1),
        (driver.ctx.s2, result.s2),
    ):
        dst.bench.bench_boresight = src.bench.bench_boresight.copy()
        dst.bench._invalidate_geometry_cache()
        dst.receiver.fsm.theta_offset = src.receiver.fsm.theta_offset
        dst.receiver.fsm.phi_offset = src.receiver.fsm.phi_offset
        dst.receiver.fsm.locked = src.receiver.fsm.locked
        dst.receiver.fsm.track_target = (
            src.receiver.fsm.track_target.copy()
            if src.receiver.fsm.track_target is not None
            else None
        )
        acq_src = src.bench.acquisition
        acq_dst = dst.bench.acquisition
        acq_dst.has_seen_beam = acq_src.has_seen_beam
        acq_dst.incident_angle = acq_src.incident_angle
        acq_dst.track_target = (
            acq_src.track_target.copy() if acq_src.track_target is not None else None
        )
        acq_dst.bench_slew_rate = acq_src.bench_slew_rate
        acq_dst.fsm_locked = acq_src.fsm_locked
        acq_dst.slew_complete = acq_src.slew_complete

    driver.global_t = t_end


def build_replay_timeline(result: ScenarioResult) -> ReplayTimeline:
    """Run schedule up to playable end and record receiver state at every step."""
    t_step = result.config.simulation.t_step
    stop_t = result.playable_t_end

    ctx = _fresh_context(result)
    driver = _ReplayDriver(ctx=ctx, t_step=t_step)

    t_values: list[float] = []
    s1_snaps: list[ReceiverSnapshot] = []
    s2_snaps: list[ReceiverSnapshot] = []
    hit_at_t: float | None = None

    timeline = ReplayTimeline(
        t_values=np.array([], dtype=np.float64),
        s1_snapshots=[],
        s2_snapshots=[],
        hit_at_t=None,
    )
    _append_initial_conditions(timeline, result)

    step_index = 0
    t = 0.0
    prev_script_key: tuple[str, int] | None = None

    while t <= stop_t + 1e-12:
        scheduled, local_t = result.schedule.script_at(t)
        script_key = (scheduled.strategy_name, scheduled.attempt_index)
        if prev_script_key != script_key:
            _append_event(
                timeline,
                step_index,
                f"{scheduled.strategy_name}: timeline started",
            )
            prev_script_key = script_key

        step_result = driver.step_frame(scheduled, local_t)
        for event in step_result.events:
            _append_event(timeline, step_index, f"{event} at t={t:.3f}")
        if hit_at_t is None and step_result.locked:
            hit_at_t = t
            _append_event(timeline, step_index, f"Lock (both) at t={t:.3f}")

        t_values.append(t)
        s1_snaps.append(_capture_receiver(ctx.s1.receiver))
        s2_snaps.append(_capture_receiver(ctx.s2.receiver))

        if step_result.locked:
            break
        if t >= stop_t - 1e-12:
            break
        step_index += 1
        t += t_step

    timeline.t_values = np.asarray(t_values, dtype=np.float64)
    timeline.s1_snapshots = s1_snaps
    timeline.s2_snapshots = s2_snaps
    timeline.hit_at_t = hit_at_t
    return timeline
