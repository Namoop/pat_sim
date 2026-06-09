"""Precomputed full-timeline replay cache for O(1) viz scrubbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from satellite.sda.receiver import ReceiverSDA

if TYPE_CHECKING:
    from satellite.scenario import ScenarioResult


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
    """Dense per-step receiver state and sparse event log."""

    q_values: np.ndarray
    s1_snapshots: list[ReceiverSnapshot]
    s2_snapshots: list[ReceiverSnapshot]
    boresight_end: np.ndarray
    phase2_handoff_index: int
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
        """Approximate heap size of the timeline."""
        n = self.step_count
        vec_bytes = n * 3 * 8 * 4  # bench + optional track targets
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

        if idx >= self.phase2_handoff_index:
            result.boresight_end = self.boresight_end.copy()
            if not result._phase2_built:
                result._ensure_phase2_transmitter()
        else:
            result.s2.phase2_transmitter = None
            result._phase2_built = False

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


def _append_event(
    timeline: ReplayTimeline,
    step_index: int,
    line: str,
) -> None:
    timeline.event_steps.append(step_index)
    timeline.event_lines.append(line)


def build_replay_timeline(result: ScenarioResult) -> ReplayTimeline:
    """Run the coupled simulation once and record state at every step."""
    from satellite.detection import beam_missed_dish_fov_at_q
    from satellite.scenario import _format_rx_angles

    q_step = result.config.simulation.q_step
    q_max = result.schedule.phase_duration
    total = result.schedule.total_duration

    q_values: list[float] = []
    s1_snaps: list[ReceiverSnapshot] = []
    s2_snaps: list[ReceiverSnapshot] = []

    timeline = ReplayTimeline(
        q_values=np.array([], dtype=np.float64),
        s1_snapshots=[],
        s2_snapshots=[],
        boresight_end=result.s2.bench.initial_beam_boresight.copy(),
        phase2_handoff_index=0,
    )

    step_index = 0

    def record(q: float) -> None:
        q_values.append(q)
        s1_snaps.append(_capture_receiver(result.s1.receiver))
        s2_snaps.append(_capture_receiver(result.s2.receiver))

    result.s2.receiver.reset_dish_tracking()
    s2 = result.s2.receiver
    _append_event(timeline, 0, "S1 Search spiral started")
    _append_event(
        timeline,
        0,
        (
            f"S2 Initial receiver offset: "
            f"{np.degrees(s2.initial_pointing_offset):.2f}° "
            f"(θ/φ magnitude "
            f"{np.degrees(s2.configured_offset_magnitude):.2f}°)"
        ),
    )

    s2_miss_logged = False
    s2_first_detect: float | None = None
    s2_slew_logged = False

    q = 0.0
    while q < q_max - 1e-12:
        rx = result.s2.receiver
        tx = result.s1.transmitter
        had_seen = rx.has_seen_beam
        dish_boresight = rx.dish_boresight.copy()
        if not had_seen and not s2_miss_logged:
            missed = beam_missed_dish_fov_at_q(
                q,
                tx.position,
                rx.dish_mount,
                dish_boresight,
                rx.dish_fov,
                tx.boresight_at,
                tx.alpha,
                tx.beam_length,
            )
            if missed is not None:
                _append_event(
                    timeline,
                    step_index,
                    _format_rx_angles(
                        q,
                        rx,
                        tx,
                        dish_boresight,
                        result.local_q(q),
                        result.config.satellite.dish_fov,
                        prefix="S2 Missed beam",
                    ),
                )
                s2_miss_logged = True

        result._step_phase1(q)
        record(q)

        if rx.has_seen_beam and not had_seen:
            _append_event(
                timeline,
                step_index,
                _format_rx_angles(
                    q,
                    rx,
                    tx,
                    dish_boresight,
                    result.local_q(q),
                    result.config.satellite.dish_fov,
                    prefix="S2 Received",
                ),
            )
            _append_event(timeline, step_index, "S2 FSM centered beam on camera")
            s2_first_detect = q
        if (
            s2_first_detect is not None
            and q > s2_first_detect + 1e-9
            and not s2_slew_logged
        ):
            _append_event(timeline, step_index, "S2 Bench slewing toward lock")
            s2_slew_logged = True

        step_index += 1
        q += q_step

    while q < q_max - 1e-12:
        result._step_phase1(q)
        record(q)
        step_index += 1
        q += q_step

    handoff_step = step_index
    timeline.phase2_handoff_index = handoff_step
    timeline.boresight_end = result.s2.bench.beam_boresight_inertial().copy()
    result.boresight_end = timeline.boresight_end.copy()
    result._ensure_phase2_transmitter()

    _append_event(timeline, handoff_step, "S1 Search spiral complete")
    if s2_first_detect is None:
        _append_event(timeline, handoff_step, "S2 No beam acquisition")
    center = "locked" if result.s2.receiver.has_seen_beam else "initial_aim"
    _append_event(
        timeline,
        handoff_step,
        f"S2 Search spiral started ({center})",
    )
    s1 = result.s1.receiver
    _append_event(
        timeline,
        handoff_step,
        f"S1 Initial receiver offset: "
        f"{np.degrees(s1.initial_pointing_offset):.2f}°",
    )

    result.s1.receiver.reset_dish_tracking()

    s1_miss_logged = False
    s1_first_detect: float | None = None
    s1_slew_logged = False
    q = q_max
    tx2 = result.s2.phase2_transmitter
    while q < total - 1e-12:
        rx = result.s1.receiver
        had_seen = rx.has_seen_beam
        dish_boresight = rx.dish_boresight.copy()
        if tx2 is not None and not had_seen and not s1_miss_logged:
            local_q = result.local_q(q)
            missed = beam_missed_dish_fov_at_q(
                local_q,
                tx2.position,
                rx.dish_mount,
                dish_boresight,
                rx.dish_fov,
                tx2.boresight_at,
                tx2.alpha,
                tx2.beam_length,
            )
            if missed is not None:
                _append_event(
                    timeline,
                    step_index,
                    _format_rx_angles(
                        q,
                        rx,
                        tx2,
                        dish_boresight,
                        local_q,
                        result.config.satellite.dish_fov,
                        prefix="S1 Missed beam",
                    ),
                )
                s1_miss_logged = True

        result._step_phase2(q)
        record(q)

        if tx2 is not None:
            if rx.has_seen_beam and not had_seen:
                _append_event(
                    timeline,
                    step_index,
                    _format_rx_angles(
                        q,
                        rx,
                        tx2,
                        dish_boresight,
                        result.local_q(q),
                        result.config.satellite.dish_fov,
                        prefix="S1 Received",
                    ),
                )
                _append_event(timeline, step_index, "S1 FSM centered beam on camera")
                s1_first_detect = q
            if (
                s1_first_detect is not None
                and q > s1_first_detect + 1e-9
                and not s1_slew_logged
            ):
                _append_event(timeline, step_index, "S1 Bench slewing toward lock")
                s1_slew_logged = True

        step_index += 1
        q += q_step

    final_step = step_index
    _append_event(timeline, final_step, "S2 Search spiral complete")
    if s1_first_detect is None:
        _append_event(timeline, final_step, "S1 No beam acquisition")

    timeline.q_values = np.asarray(q_values, dtype=np.float64)
    timeline.s1_snapshots = s1_snaps
    timeline.s2_snapshots = s2_snaps
    return timeline
