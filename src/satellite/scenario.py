"""Scenario orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from satellite.config import ScenarioConfig
from satellite.detection import (
    alignment_dot,
    beam_hits_dish,
    beam_hits_dish_at_q,
    beam_missed_dish_fov_at_q,
)
from satellite.diagnostics import SimReplayProfiler
from satellite.geometry import actual_target_direction
from satellite.math3d import Vec3, angle_between, distance
from satellite.schedule import SearchPhase, SearchSchedule
from satellite.sda.receiver import ReceiverSDA
from satellite.sda.satellite import Satellite, build_phase2_transmitter
from satellite.sda.transmitter import TransmitterSDA


@dataclass
class ScenarioResult:
    config: ScenarioConfig
    s1: Satellite
    s2: Satellite
    schedule: SearchSchedule
    phase1_hit: bool
    phase1_hit_at_q: float | None
    phase2_hit: bool
    phase2_hit_at_q: float | None
    phase2_spiral_center_source: str
    boresight_end: Vec3
    phase1_target_direction: Vec3
    phase2_target_direction: Vec3
    phase1_alignment: float
    phase2_alignment: float
    _phase2_built: bool = field(default=False, repr=False)
    last_sim_profiler: SimReplayProfiler | None = field(default=None, repr=False)

    @property
    def hit(self) -> bool:
        return self.phase1_hit or self.phase2_hit

    @property
    def hit_at_q(self) -> float | None:
        if self.phase1_hit_at_q is not None:
            return self.phase1_hit_at_q
        return self.phase2_hit_at_q

    @property
    def p1(self) -> Vec3:
        return self.s1.position

    @property
    def pt(self) -> Vec3:
        return self.s2.position

    @property
    def transmitter(self) -> TransmitterSDA:
        return self.s1.transmitter

    @property
    def receiver(self) -> ReceiverSDA:
        return self.s2.receiver

    @property
    def target_direction(self) -> Vec3:
        return self.phase1_target_direction

    @property
    def alignment_at_q_max(self) -> float:
        return self.phase1_alignment

    @property
    def believed_boresight(self) -> Vec3:
        return self.s1.believed_boresight

    @property
    def believed_distance(self) -> float:
        return distance(self.s1.position, self.s2.position)

    def local_q(self, q: float) -> float:
        _, local = self.schedule.phase_at(q)
        return local

    def active_transmitter(self, q: float) -> TransmitterSDA:
        phase, _ = self.schedule.phase_at(q)
        if phase is SearchPhase.S1_TRANSMIT:
            return self.s1.transmitter
        if self.s2.phase2_transmitter is None:
            raise RuntimeError("phase-2 transmitter not built yet")
        return self.s2.phase2_transmitter

    def active_receiver(self, q: float) -> ReceiverSDA:
        phase, _ = self.schedule.phase_at(q)
        if phase is SearchPhase.S1_TRANSMIT:
            return self.s2.receiver
        return self.s1.receiver

    def active_in_cone(self, q: float) -> bool:
        phase, local_q = self.schedule.phase_at(q)
        tx = self.active_transmitter(q)
        rx = self.active_receiver(q)
        return beam_hits_dish_at_q(
            local_q,
            tx.position,
            rx.dish_mount,
            rx.dish_boresight,
            rx.dish_fov,
            tx.boresight_at,
            tx.alpha,
            tx.beam_length,
        )

    def in_cone_at_q(self, q: float) -> bool:
        """Backward-compat alias."""
        return self.active_in_cone(q)

    def check_dish_hit(self, q: float) -> bool:
        return self.active_in_cone(q)

    def _measure(self, profiler: SimReplayProfiler | None, name: str):
        if profiler is not None and profiler.enabled:
            return profiler.measure(name)
        return nullcontext()

    def _step_phase1(
        self,
        q: float,
        profiler: SimReplayProfiler | None = None,
    ) -> bool:
        tx = self.s1.transmitter
        rx = self.s2.receiver
        local_q = q
        q_step = self.config.simulation.q_step

        if profiler is not None and profiler.enabled:
            profiler.step_count += 1

        with self._measure(profiler, "tx_boresight_at"):
            beam = tx.boresight_at(local_q)
        with self._measure(profiler, "rx_dish_geometry"):
            geom = rx.geometry_snapshot()
        with self._measure(profiler, "beam_hits_dish"):
            in_fov = beam_hits_dish(
                tx.position,
                geom.mount,
                geom.dish_boresight,
                rx.dish_fov,
                beam,
                tx.alpha,
                tx.beam_length,
            )
        with self._measure(profiler, "observe_beam"):
            rx.observe_beam(
                in_fov,
                beam,
                q_step,
                dish_at_step_start=geom.dish_boresight,
            )
        return in_fov

    def _step_phase2(
        self,
        q: float,
        profiler: SimReplayProfiler | None = None,
    ) -> bool:
        if self.s2.phase2_transmitter is None:
            raise RuntimeError("phase-2 transmitter not built yet")
        tx = self.s2.phase2_transmitter
        rx = self.s1.receiver
        local_q = q - self.schedule.phase_duration
        q_step = self.config.simulation.q_step

        if profiler is not None and profiler.enabled:
            profiler.step_count += 1

        with self._measure(profiler, "tx_boresight_at"):
            beam = tx.boresight_at(local_q)
        with self._measure(profiler, "rx_dish_geometry"):
            geom = rx.geometry_snapshot()
        with self._measure(profiler, "beam_hits_dish"):
            in_fov = beam_hits_dish(
                tx.position,
                geom.mount,
                geom.dish_boresight,
                rx.dish_fov,
                beam,
                tx.alpha,
                tx.beam_length,
            )
        with self._measure(profiler, "observe_beam"):
            rx.observe_beam(
                in_fov,
                beam,
                q_step,
                dish_at_step_start=geom.dish_boresight,
            )
        return in_fov

    def _ensure_phase2_transmitter(self) -> None:
        self.s2.phase2_transmitter = build_phase2_transmitter(
            self.s2,
            self.s1,
            self.boresight_end,
            self.config,
        )
        self._phase2_built = True

    def replay_to(
        self,
        q_end: float,
        *,
        event_log: list[str] | None = None,
    ) -> None:
        """Coupled replay from q=0 through q_end (headless + viz)."""
        q_step = self.config.simulation.q_step
        q_max = self.schedule.phase_duration
        q_end = float(np.clip(q_end, 0.0, self.schedule.total_duration))
        profiler = SimReplayProfiler.from_config(self.config.simulation.profile_replay)

        with profiler.measure("replay_total"):
            with profiler.measure("reset_s2"):
                self.s2.receiver.reset_dish_tracking()

            if event_log is not None:
                s2 = self.s2.receiver
                event_log.clear()
                event_log.extend(
                    [
                        "S1 Search spiral started",
                        (
                            f"S2 Initial receiver offset: "
                            f"{np.degrees(s2.initial_pointing_offset):.2f}° "
                            f"(θ/φ magnitude "
                            f"{np.degrees(s2.configured_offset_magnitude):.2f}°)"
                        ),
                    ]
                )

            s2_miss_logged = False
            s2_first_detect: float | None = None
            s2_slew_logged = False

            q = 0.0
            while q < q_max - 1e-12 and q <= q_end + 1e-12:
                rx = self.s2.receiver
                tx = self.s1.transmitter
                if event_log is not None:
                    had_seen = rx.has_seen_beam
                    dish_boresight = rx.dish_boresight.copy()
                    if not had_seen and not s2_miss_logged:
                        with profiler.measure("event_log_miss_check"):
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
                            with profiler.measure("event_log_format"):
                                event_log.append(
                                    _format_rx_angles(
                                        q,
                                        rx,
                                        tx,
                                        dish_boresight,
                                        self.local_q(q),
                                        self.config.satellite.dish_fov,
                                        prefix="S2 Missed beam",
                                    )
                                )
                            s2_miss_logged = True

                in_fov = self._step_phase1(q, profiler)

                if event_log is not None:
                    if rx.has_seen_beam and not had_seen:
                        with profiler.measure("event_log_format"):
                            event_log.append(
                                _format_rx_angles(
                                    q,
                                    rx,
                                    tx,
                                    dish_boresight,
                                    self.local_q(q),
                                    self.config.satellite.dish_fov,
                                    prefix="S2 Received",
                                )
                            )
                            event_log.append("S2 FSM centered beam on camera")
                        s2_first_detect = q
                    if (
                        s2_first_detect is not None
                        and q > s2_first_detect + 1e-9
                        and not s2_slew_logged
                    ):
                        event_log.append("S2 Bench slewing toward lock")
                        s2_slew_logged = True

                q += q_step

            if q_end >= q_max - 1e-12:
                while q < q_max - 1e-12:
                    self._step_phase1(q, profiler)
                    q += q_step

                with profiler.measure("phase2_handoff"):
                    self.boresight_end = (
                        self.s2.bench.beam_boresight_inertial().copy()
                    )
                    self._ensure_phase2_transmitter()

                if event_log is not None:
                    event_log.append("S1 Search spiral complete")
                    if s2_first_detect is None:
                        event_log.append("S2 No beam acquisition")
                    center = (
                        "locked"
                        if self.s2.receiver.has_seen_beam
                        else "initial_aim"
                    )
                    event_log.append(f"S2 Search spiral started ({center})")
                    s1 = self.s1.receiver
                    event_log.append(
                        f"S1 Initial receiver offset: "
                        f"{np.degrees(s1.initial_pointing_offset):.2f}°"
                    )

                with profiler.measure("reset_s1"):
                    self.s1.receiver.reset_dish_tracking()

                s1_miss_logged = False
                s1_first_detect: float | None = None
                s1_slew_logged = False
                q = q_max
                tx2 = self.s2.phase2_transmitter
                while q <= q_end + 1e-12 and q < self.schedule.total_duration - 1e-12:
                    rx = self.s1.receiver
                    if event_log is not None and tx2 is not None:
                        had_seen = rx.has_seen_beam
                        dish_boresight = rx.dish_boresight.copy()
                        local_q = self.local_q(q)
                        if not had_seen and not s1_miss_logged:
                            with profiler.measure("event_log_miss_check"):
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
                                with profiler.measure("event_log_format"):
                                    event_log.append(
                                        _format_rx_angles(
                                            q,
                                            rx,
                                            tx2,
                                            dish_boresight,
                                            local_q,
                                            self.config.satellite.dish_fov,
                                            prefix="S1 Missed beam",
                                        )
                                    )
                                s1_miss_logged = True

                    in_fov = self._step_phase2(q, profiler)

                    if event_log is not None and tx2 is not None:
                        if rx.has_seen_beam and not had_seen:
                            with profiler.measure("event_log_format"):
                                event_log.append(
                                    _format_rx_angles(
                                        q,
                                        rx,
                                        tx2,
                                        dish_boresight,
                                        self.local_q(q),
                                        self.config.satellite.dish_fov,
                                        prefix="S1 Received",
                                    )
                                )
                                event_log.append("S1 FSM centered beam on camera")
                            s1_first_detect = q
                        if (
                            s1_first_detect is not None
                            and q > s1_first_detect + 1e-9
                            and not s1_slew_logged
                        ):
                            event_log.append("S1 Bench slewing toward lock")
                            s1_slew_logged = True

                    q += q_step

                if event_log is not None and q_end >= self.schedule.total_duration - 1e-12:
                    event_log.append("S2 Search spiral complete")
                    if s1_first_detect is None:
                        event_log.append("S1 No beam acquisition")

        self.last_sim_profiler = profiler
        profiler.report(f"replay_to q<={q_end:.3f}")

    def dish_boresight_for_display(self, satellite: str, q: float) -> Vec3:
        """Return bench dish boresight for viz (bench slew visible; FSM not drawn)."""
        phase, _ = self.schedule.phase_at(q)
        if satellite == "S1":
            if phase is SearchPhase.S1_TRANSMIT:
                return self.s1.receiver.initial_dish_boresight
            return self.s1.receiver.dish_boresight
        return self.s2.receiver.dish_boresight

    def boresight_ray_length(self, satellite: str) -> float:
        sat = self.s1 if satellite == "S1" else self.s2
        return (
            distance(sat.position, sat.partner_actual)
            + self.config.simulation.boresight_extension
        )


def _format_rx_angles(
    q: float,
    receiver: ReceiverSDA,
    transmitter: TransmitterSDA,
    dish_boresight: Vec3,
    local_q: float,
    dish_fov: float,
    *,
    prefix: str,
) -> str:
    toward_partner = receiver.nominal_boresight
    beam_axis = transmitter.boresight_at(local_q)
    toward_source = -beam_axis / np.linalg.norm(beam_axis)

    receiver_offset = np.degrees(angle_between(toward_partner, dish_boresight))
    beam_offset = np.degrees(angle_between(toward_partner, beam_axis))
    incident = np.degrees(angle_between(dish_boresight, toward_source))
    fov_deg = np.degrees(dish_fov)

    return (
        f"{prefix} (q={q:.3f})\n"
        f"  Receiver offset: {receiver_offset:.2f}°\n"
        f"  Beam boresight offset: {beam_offset:.2f}°\n"
        f"  Incident angle: {incident:.2f}° (FOV {fov_deg:.2f}°)"
    )


def _scan_phase(
    result: ScenarioResult,
    phase: SearchPhase,
    q_max: float,
    q_step: float,
) -> tuple[bool, float | None]:
    hit_at_q: float | None = None

    if phase is SearchPhase.S1_TRANSMIT:
        result.s2.receiver.reset_dish_tracking()
        q = 0.0
        while q < q_max - 1e-12:
            in_fov = result._step_phase1(q)
            if in_fov and hit_at_q is None:
                hit_at_q = q
            q += q_step
    else:
        result.s1.receiver.reset_dish_tracking()
        q = q_max
        total = result.schedule.total_duration
        while q < total - 1e-12:
            in_fov = result._step_phase2(q)
            if in_fov and hit_at_q is None:
                hit_at_q = q
            q += q_step

    return hit_at_q is not None, hit_at_q


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Run headless two-phase SDA search; no mesh allocation."""
    p1 = config.s1.position
    pt = config.s2.position
    schedule = SearchSchedule(phase_duration=config.simulation.q_max)

    s1 = Satellite.build("S1", config.s1, pt, config)
    s2 = Satellite.build("S2", config.s2, p1, config)

    phase1_target = actual_target_direction(p1, pt)
    phase2_target = actual_target_direction(pt, p1)

    result = ScenarioResult(
        config=config,
        s1=s1,
        s2=s2,
        schedule=schedule,
        phase1_hit=False,
        phase1_hit_at_q=None,
        phase2_hit=False,
        phase2_hit_at_q=None,
        phase2_spiral_center_source="initial_aim",
        boresight_end=s2.bench.initial_beam_boresight.copy(),
        phase1_target_direction=phase1_target,
        phase2_target_direction=phase2_target,
        phase1_alignment=0.0,
        phase2_alignment=0.0,
    )

    phase1_hit, phase1_hit_at_q = _scan_phase(
        result,
        SearchPhase.S1_TRANSMIT,
        config.simulation.q_max,
        config.simulation.q_step,
    )
    result.phase1_hit = phase1_hit
    result.phase1_hit_at_q = phase1_hit_at_q
    result.boresight_end = s2.bench.beam_boresight_inertial().copy()
    result.phase2_spiral_center_source = (
        "locked" if s2.receiver.has_seen_beam else "initial_aim"
    )
    result.phase1_alignment = alignment_dot(
        phase1_target,
        s1.transmitter.boresight_at(config.simulation.q_max),
    )

    result._ensure_phase2_transmitter()

    phase2_hit, phase2_hit_at_q = _scan_phase(
        result,
        SearchPhase.S2_TRANSMIT,
        config.simulation.q_max,
        config.simulation.q_step,
    )
    result.phase2_hit = phase2_hit
    result.phase2_hit_at_q = phase2_hit_at_q
    result.phase2_alignment = alignment_dot(
        phase2_target,
        s2.phase2_transmitter.boresight_at(config.simulation.q_max),
    )

    return result


def format_summary(result: ScenarioResult) -> str:
    """Human-readable headless run summary."""
    tx1 = result.s1.transmitter
    q_max = result.config.simulation.q_max
    ext = result.config.simulation.boresight_extension
    lines = [
        f"Scenario: {result.config.name}",
        f"Schedule: two phases, q_max={q_max:.3f} each (total={result.schedule.total_duration:.3f})",
        f"Believed boresight (S1) = {tuple(round(float(x), 8) for x in result.believed_boresight)}",
        f"Link range |P_t - P_1| = {result.believed_distance:.6f}",
        f"Beam length (default) = range + {ext:.1f} = {tx1.beam_length:.6f}",
        f"w = {tx1.w:.10f}",
        f"Phase 1 target U_t = {tuple(round(float(x), 8) for x in result.phase1_target_direction)}",
        f"Phase 1 hit: {'yes' if result.phase1_hit else 'no'}",
    ]
    if result.phase1_hit_at_q is not None:
        lines.append(f"Phase 1 hit at q = {result.phase1_hit_at_q:.4f}")
    if result.s2.receiver.has_seen_beam and result.s2.bench.acquisition.incident_angle is not None:
        fov = result.config.satellite.dish_fov
        rx = result.s2.receiver
        lines.append(
            f"S2 bench offset magnitude = {np.degrees(rx.bench.configured_bench_offset_magnitude):.3f} deg"
        )
        lines.append(
            f"S2 pointing offset = {np.degrees(rx.initial_pointing_offset):.3f} deg"
        )
        lines.append(
            f"S2 dish incident angle at first detection = "
            f"{np.degrees(rx.bench.acquisition.incident_angle):.3f} deg "
            f"(dish FOV = {np.degrees(fov):.3f} deg)"
        )
    lines.append(
        f"Phase 1 alignment at q_max = {result.phase1_alignment:.6f} "
        f"(threshold cos(alpha) = {np.cos(result.config.satellite.alpha):.6f})"
    )
    lines.append(
        f"Phase 2 spiral center: {result.phase2_spiral_center_source} "
        f"(boresight_end = {tuple(round(float(x), 6) for x in result.boresight_end)})"
    )
    lines.append(f"Phase 2 hit: {'yes' if result.phase2_hit else 'no'}")
    if result.phase2_hit_at_q is not None:
        lines.append(f"Phase 2 hit at q = {result.phase2_hit_at_q:.4f}")
    if result.s1.receiver.has_seen_beam and result.s1.bench.acquisition.incident_angle is not None:
        lines.append(
            f"S1 dish incident angle at first detection = "
            f"{np.degrees(result.s1.bench.acquisition.incident_angle):.3f} deg"
        )
    lines.append(
        f"Phase 2 alignment at q_max = {result.phase2_alignment:.6f}"
    )
    if result.hit_at_q is not None:
        lines.append(
            f"Overall hit: {'yes' if result.hit else 'no'} (first at q = {result.hit_at_q})"
        )
    else:
        lines.append(f"Overall hit: {'yes' if result.hit else 'no'}")
    lines.append(
        f"Phase 1 offsets at q_max: theta_off = {tx1.theta_offset(q_max):.6f}, "
        f"phi_off = {tx1.phi_offset(q_max):.6f}"
    )
    return "\n".join(lines)
