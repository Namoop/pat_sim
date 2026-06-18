"""Optical bench — dish and TX beam share bench boresight; slow slew DOF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from satellite.math.geometry import direction_with_local_offset, transmitter_basis
from satellite.math.math3d import Vec3, angle_between, normalize, rotate_toward, spherical_angles_from_direction, add_scaled_vector
from satellite.physics.acquisition import AcquisitionState
from satellite.physics.fsm import FastSteeringMirror


@dataclass(frozen=True)
class BenchGeometry:
    dish_boresight: Vec3
    mount: Vec3


class OpticalBench:
    """
    Optical bench on a correctly pointed spacecraft body.
    Launch mispoint is bench rotation; dish and TX beam are co-aligned on bench.
    """

    def __init__(
        self,
        position: Vec3,
        partner_position: Vec3,
        bench_theta_offset: float,
        bench_phi_offset: float,
        max_beam_speed: float,
        max_fsm_speed: float,
        max_fsm_radius: float,
        scan_envelope_ramp: float = 2.0,
    ) -> None:
        self.position = position
        self.partner_position = partner_position
        self.bench_theta_offset = bench_theta_offset
        self.bench_phi_offset = bench_phi_offset
        self.max_beam_speed = max_beam_speed
        self.max_fsm_speed = max_fsm_speed
        self.max_fsm_radius = max_fsm_radius
        self.scan_envelope_ramp = scan_envelope_ramp


        self.toward_partner = normalize(partner_position - position)
        self._initial_bench_boresight = direction_with_local_offset(
            self.toward_partner,
            bench_theta_offset,
            bench_phi_offset,
        )
        self.bench_boresight = self._initial_bench_boresight.copy()
        self.acquisition = AcquisitionState()
        self._cache_valid = False
        self._cached_dish: Vec3 | None = None
        self._cached_mount: Vec3 | None = None

    def _invalidate_geometry_cache(self) -> None:
        self._cache_valid = False
        self._cached_dish = None
        self._cached_mount = None

    def reset_tracking(self) -> None:
        self.bench_boresight = self._initial_bench_boresight.copy()
        self.acquisition = AcquisitionState()
        self._invalidate_geometry_cache()

    def set_bench_aim(self, direction: Vec3) -> None:
        """Set shared dish/transmit boresight (collinear on bench)."""
        self.bench_boresight = normalize(direction)
        self._invalidate_geometry_cache()

    @property
    def initial_boresight(self) -> Vec3:
        return self._initial_bench_boresight

    @property
    def initial_dish_boresight(self) -> Vec3:
        return self._initial_bench_boresight

    @property
    def initial_beam_boresight(self) -> Vec3:
        return self._initial_bench_boresight

    def dish_boresight_inertial(self) -> Vec3:
        """Bench boresight = dish aim (FSM does not deflect the aperture)."""
        if self._cache_valid and self._cached_dish is not None:
            return self._cached_dish
        dish = normalize(self.bench_boresight)
        self._cached_dish = dish
        self._cache_valid = True
        return dish

    def beam_boresight_inertial(self) -> Vec3:
        """TX beam exits along bench boresight."""
        return self.dish_boresight_inertial()

    def geometry_snapshot(self, body_radius: float) -> BenchGeometry:
        dish = self.dish_boresight_inertial()
        return BenchGeometry(
            dish_boresight=dish,
            mount=self.position + dish * body_radius,
        )

    @property
    def initial_pointing_offset(self) -> float:
        return angle_between(self.toward_partner, self._initial_bench_boresight)

    @property
    def configured_bench_offset_magnitude(self) -> float:
        return float(np.hypot(self.bench_theta_offset, self.bench_phi_offset))

    def dish_mount(self, body_radius: float) -> Vec3:
        if self._cache_valid and self._cached_mount is not None:
            return self._cached_mount
        dish = self.dish_boresight_inertial()
        mount = add_scaled_vector(self.position, dish, body_radius)
        self._cached_mount = mount
        return mount

    def dish_mount_for_boresight(self, boresight: Vec3, body_radius: float) -> Vec3:
        return add_scaled_vector(self.position, boresight, body_radius)

    def observe_beam(
        self,
        fsm: FastSteeringMirror,
        in_cone: bool,
        source_position: Vec3,
        dq: float,
        *,
        dish_mount: Vec3,
        dish_at_step_start: Vec3 | None = None,
    ) -> tuple[Vec3, list[str]]:
        """
        On acquisition: FSM snaps to center beam on camera; bench slews slowly
        afterward to recenter the FSM while maintaining lock.
        """
        events: list[str] = []
        acq = self.acquisition
        dish = (
            dish_at_step_start
            if dish_at_step_start is not None
            else self.dish_boresight_inertial()
        )
        just_detected = False
        slewed = False
        was_complete = acq.slew_complete

        if in_cone and not acq.has_seen_beam:
            toward_source = normalize(source_position - dish_mount)
            acq.incident_angle = angle_between(dish, toward_source)
            acq.track_target = normalize(source_position - self.position)
            acq.has_seen_beam = True
            acq.fsm_locked = False  # Start slewing
            acq.slew_complete = False
            fsm.snap_to(self.bench_boresight, toward_source)
            acq.bench_slew_rate = self.max_beam_speed
            if self.max_beam_speed == 0.0:
                self.bench_boresight = acq.track_target.copy()
                self._invalidate_geometry_cache()
                acq.slew_complete = True
            just_detected = True

        if acq.has_seen_beam and acq.track_target is not None:
            # Update FSM slew every step
            fsm.update(self.bench_boresight, dq, self.max_fsm_speed, self.max_fsm_radius)
            acq.fsm_locked = fsm.locked


            if not just_detected and not acq.slew_complete:
                max_step = (acq.bench_slew_rate or 0.0) * dq
                remaining = angle_between(self.bench_boresight, acq.track_target)
                self.bench_boresight = rotate_toward(
                    self.bench_boresight,
                    acq.track_target,
                    max_step,
                )
                self._invalidate_geometry_cache()
                # Mirror already updated above, but we sync target
                fsm.update_with_bench(self.bench_boresight)
                if remaining <= max_step + 1e-12:
                    self.bench_boresight = acq.track_target.copy()
                    self._invalidate_geometry_cache()
                    acq.slew_complete = True
                slewed = True

        if acq.slew_complete and not was_complete:
            events.append("Slew complete")

        if slewed:
            return self.dish_boresight_inertial(), events
        return dish, events
