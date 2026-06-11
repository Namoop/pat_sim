"""Precomputed full-timeline replay cache for O(1) viz scrubbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from satellite.strategy.base import StrategyContext, link_established
from satellite.strategy.movements import Reset, build_aim_context
from satellite.strategy.runner import FrameRunner
from satellite.strategy.schedule import ScheduledEpoch

if TYPE_CHECKING:
    from satellite.scenario import ScenarioResult
    from satellite.sda.receiver import ReceiverSDA


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


@dataclass
class _SatEpochState:
    ctx: object
    epoch_key: tuple[str, int]


@dataclass
class _ReplayDriver:
    ctx: StrategyContext
    q_step: float
    s1_state: _SatEpochState | None = None
    s2_state: _SatEpochState | None = None
    current_key: tuple[str, int] | None = None
    global_q: float = 0.0

    def _ensure_epoch(self, epoch: ScheduledEpoch) -> None:
        key = (epoch.strategy_name, epoch.epoch_index)
        if self.current_key == key:
            return
        self.current_key = key
        self.s1_state = _SatEpochState(
            ctx=build_aim_context(
                self.ctx.s1,
                reset=isinstance(epoch.s1_action.movement, Reset),
            ),
            epoch_key=key,
        )
        self.s2_state = _SatEpochState(
            ctx=build_aim_context(
                self.ctx.s2,
                reset=isinstance(epoch.s2_action.movement, Reset),
            ),
            epoch_key=key,
        )

    def step_frame(
        self,
        epoch: ScheduledEpoch,
        local_t: float,
    ) -> tuple[bool, bool]:
        self._ensure_epoch(epoch)
        runner = FrameRunner(self.ctx)
        aim1 = runner._apply_aim(
            self.ctx.s1,
            epoch.s1_action.movement,
            self.s1_state,
            local_t,
            epoch.duration,
        )
        aim2 = runner._apply_aim(
            self.ctx.s2,
            epoch.s2_action.movement,
            self.s2_state,
            local_t,
            epoch.duration,
        )
        return runner._bidirectional_lock(aim1, aim2, self.q_step)


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
    bench._invalidate_geometry_cache()


@dataclass
class ReplayTimeline:
    q_values: np.ndarray
    s1_snapshots: list[ReceiverSnapshot]
    s2_snapshots: list[ReceiverSnapshot]
    hit_at_q: float | None
    event_steps: list[int] = field(default_factory=list)
    event_lines: list[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.q_values)

    def index_for_q(self, q: float) -> int:
        if self.step_count == 0:
            return 0
        if self.step_count == 1:
            return 0
        step = float(self.q_values[1] - self.q_values[0])
        idx = int(np.floor((q + 1e-12) / step))
        return int(np.clip(idx, 0, self.step_count - 1))

    def memory_bytes(self) -> int:
        n = self.step_count
        vec_bytes = n * 3 * 8 * 4
        snap_bytes = n * 2 * 192
        event_bytes = sum(len(s.encode("utf-8")) for s in self.event_lines)
        return int(self.q_values.nbytes + vec_bytes + snap_bytes + event_bytes)

    def memory_summary(self) -> str:
        mb = self.memory_bytes() / (1024 * 1024)
        return f"{self.step_count} steps, ~{mb:.2f} MiB"

    def restore(self, result: ScenarioResult, q_end: float) -> int:
        idx = self.index_for_q(q_end)
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
            f"  S1 bench θ={_format_sigfig(s1.bench_theta_offset * 1e3)} mrad"
            f"  φ={_format_sigfig(s1.bench_phi_offset * 1e3)} mrad"
        ),
        (
            f"  S2 bench θ={_format_sigfig(s2.bench_theta_offset * 1e3)} mrad"
            f"  φ={_format_sigfig(s2.bench_phi_offset * 1e3)} mrad"
        ),
    ]


def _append_initial_conditions(timeline: ReplayTimeline, result: ScenarioResult) -> None:
    for line in initial_conditions_lines(result):
        _append_event(timeline, 0, line)


def _fresh_context(result: ScenarioResult) -> StrategyContext:
    from satellite.sda.satellite import Satellite

    cfg = result.config
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    s2 = Satellite.build("S2", cfg.s2, cfg.s1.position, cfg)
    return StrategyContext(s1=s1, s2=s2, config=cfg)


def replay_to_q(
    result: ScenarioResult,
    q_end: float,
    *,
    event_log: list[str] | None = None,
) -> None:
    """Replay coupled simulation from q=0 through q_end."""
    if result._replay_timeline is not None:
        timeline = result._replay_timeline
        idx = timeline.restore(result, q_end)
        if event_log is not None:
            event_log.clear()
            include_final = q_end >= result.schedule.total_duration - 1e-12
            event_log.extend(
                timeline.event_log_up_to(idx, include_final=include_final)
            )
        return

    if result._stepper is None:
        ctx = _fresh_context(result)
        result._stepper = _ReplayDriver(
            ctx=ctx,
            q_step=result.config.simulation.q_step,
        )
        result.s1.receiver.reset_dish_tracking()
        result.s2.receiver.reset_dish_tracking()
        ctx.s1.receiver.reset_dish_tracking()
        ctx.s2.receiver.reset_dish_tracking()

    driver: _ReplayDriver = result._stepper
    q_step = driver.q_step
    q_end = float(np.clip(q_end, 0.0, result.schedule.total_duration))

    if abs(driver.global_q - q_end) < 1e-9:
        return

    if driver.global_q > q_end + 1e-12:
        driver.ctx = _fresh_context(result)
        driver.s1_state = None
        driver.s2_state = None
        driver.current_key = None
        driver.global_q = 0.0
        driver._prev_strategy = None
        result.s1.receiver.reset_dish_tracking()
        result.s2.receiver.reset_dish_tracking()
        driver.ctx.s1.receiver.reset_dish_tracking()
        driver.ctx.s2.receiver.reset_dish_tracking()

    prev_strategy: str | None = getattr(driver, "_prev_strategy", None)

    while True:
        epoch, local_t = result.schedule.epoch_at(driver.global_q)
        if prev_strategy is not None and epoch.strategy_name != prev_strategy:
            driver.ctx.s1.receiver.reset_dish_tracking()
            driver.ctx.s2.receiver.reset_dish_tracking()
            driver.s1_state = None
            driver.s2_state = None
            driver.current_key = None
        prev_strategy = epoch.strategy_name
        driver.step_frame(epoch, local_t)
        if driver.global_q >= q_end - 1e-12:
            break
        driver.global_q += q_step

    driver._prev_strategy = prev_strategy

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

    driver.global_q = q_end


def build_replay_timeline(result: ScenarioResult) -> ReplayTimeline:
    """Run full schedule and record receiver state at every step."""
    q_step = result.config.simulation.q_step
    total = result.schedule.total_duration

    ctx = _fresh_context(result)
    driver = _ReplayDriver(ctx=ctx, q_step=q_step)

    q_values: list[float] = []
    s1_snaps: list[ReceiverSnapshot] = []
    s2_snaps: list[ReceiverSnapshot] = []
    hit_at_q: float | None = None

    timeline = ReplayTimeline(
        q_values=np.array([], dtype=np.float64),
        s1_snapshots=[],
        s2_snapshots=[],
        hit_at_q=None,
    )
    _append_initial_conditions(timeline, result)

    step_index = 0
    q = 0.0
    prev_epoch_key: tuple[str, int] | None = None
    prev_strategy: str | None = None

    while q <= total + 1e-12:
        epoch, local_t = result.schedule.epoch_at(q)
        if prev_strategy is not None and epoch.strategy_name != prev_strategy:
            ctx.s1.receiver.reset_dish_tracking()
            ctx.s2.receiver.reset_dish_tracking()
            driver.s1_state = None
            driver.s2_state = None
            driver.current_key = None
        prev_strategy = epoch.strategy_name
        epoch_key = (epoch.strategy_name, epoch.epoch_index)
        if prev_epoch_key != epoch_key:
            _append_event(
                timeline,
                step_index,
                f"{epoch.strategy_name}: {epoch.label or 'epoch'} started",
            )
            prev_epoch_key = epoch_key

        hit_12, hit_21 = driver.step_frame(epoch, local_t)
        if hit_at_q is None and (hit_12 or hit_21):
            hit_at_q = q
            direction = "S1→S2" if hit_12 else "S2→S1"
            if hit_12 and hit_21:
                direction = "both"
            _append_event(timeline, step_index, f"Lock ({direction}) at q={q:.3f}")

        q_values.append(q)
        s1_snaps.append(_capture_receiver(ctx.s1.receiver))
        s2_snaps.append(_capture_receiver(ctx.s2.receiver))

        if q >= total - 1e-12:
            break
        step_index += 1
        q += q_step

    timeline.q_values = np.asarray(q_values, dtype=np.float64)
    timeline.s1_snapshots = s1_snaps
    timeline.s2_snapshots = s2_snaps
    timeline.hit_at_q = hit_at_q
    return timeline
