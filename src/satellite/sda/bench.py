"""Optical bench — dish and TX beam share bench boresight; slow slew DOF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from satellite.geometry import direction_with_local_offset, transmitter_basis
from satellite.math3d import Vec3, angle_between, normalize, rotate_toward, spherical_angles_from_direction
from satellite.sda.acquisition import AcquisitionState
from satellite.sda.fsm import FastSteeringMirror


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
        bench_slew_time: float,
    ) -> None:
        self.position = position
        self.partner_position = partner_position
        self.bench_theta_offset = bench_theta_offset
        self.bench_phi_offset = bench_phi_offset
        self.bench_slew_time = bench_slew_time

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

    def _invalidate_geometry_cache(self) -> None:
        self._cache_valid = False
        self._cached_dish = None

    def reset_tracking(self) -> None:
        self.bench_boresight = self._initial_bench_boresight.copy()
        self.acquisition = AcquisitionState()
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
        dish = self.dish_boresight_inertial()
        return self.position + dish * body_radius

    def dish_mount_for_boresight(self, boresight: Vec3, body_radius: float) -> Vec3:
        return self.position + boresight * body_radius

    def observe_beam(
        self,
        fsm: FastSteeringMirror,
        in_cone: bool,
        beam_direction: Vec3,
        dq: float,
        *,
        dish_at_step_start: Vec3 | None = None,
    ) -> Vec3:
        """
        On acquisition: FSM snaps to center beam on camera; bench slews slowly
        afterward to recenter the FSM while maintaining lock.
        """
        acq = self.acquisition
        dish = (
            dish_at_step_start
            if dish_at_step_start is not None
            else self.dish_boresight_inertial()
        )
        just_detected = False
        slewed = False

        if in_cone and not acq.has_seen_beam:
            toward_source = -normalize(beam_direction)
            acq.incident_angle = angle_between(dish, toward_source)
            acq.track_target = toward_source.copy()
            acq.has_seen_beam = True
            acq.fsm_locked = True
            fsm.snap_to(self.bench_boresight, toward_source)
            bench_incident = angle_between(self.bench_boresight, toward_source)
            if self.bench_slew_time > 0.0:
                acq.bench_slew_rate = bench_incident / self.bench_slew_time
            else:
                acq.bench_slew_rate = float("inf")
            just_detected = True

        if acq.has_seen_beam and not just_detected and acq.track_target is not None:
            max_step = (acq.bench_slew_rate or 0.0) * dq
            self.bench_boresight = rotate_toward(
                self.bench_boresight,
                acq.track_target,
                max_step,
            )
            self._invalidate_geometry_cache()
            fsm.update_with_bench(self.bench_boresight)
            slewed = True

        if slewed:
            return self.dish_boresight_inertial()
        return dish
