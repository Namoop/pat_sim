"""Spacecraft body orientation — dish and beam are fixed offsets on the body."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from satellite.geometry import (
    direction_with_local_offset,
    direction_with_tangent_offset,
    transmitter_basis,
)
from satellite.math3d import Vec3, angle_between, normalize, rotate_toward, spherical_angles_from_direction


@dataclass
class AcquisitionState:
    has_seen_beam: bool = False
    incident_angle: float | None = None
    track_target: Vec3 | None = None
    slew_rate: float | None = None


class SpacecraftBody:
    """
    Rigid spacecraft body: body boresight slews in inertial space; dish and
    transmit beam directions are fixed θ/φ offsets in the body frame.
    """

    def __init__(
        self,
        position: Vec3,
        partner_position: Vec3,
        body_theta_offset: float,
        body_phi_offset: float,
        dish_theta_offset: float,
        dish_phi_offset: float,
        beam_theta_offset: float,
        beam_phi_offset: float,
        body_slew_time: float,
    ) -> None:
        self.position = position
        self.partner_position = partner_position
        self.body_theta_offset = body_theta_offset
        self.body_phi_offset = body_phi_offset
        self.dish_theta_offset = dish_theta_offset
        self.dish_phi_offset = dish_phi_offset
        self.beam_theta_offset = beam_theta_offset
        self.beam_phi_offset = beam_phi_offset
        self.body_slew_time = body_slew_time

        self.toward_partner = normalize(partner_position - position)
        self._initial_body_boresight = direction_with_local_offset(
            self.toward_partner,
            body_theta_offset,
            body_phi_offset,
        )
        self.body_boresight = self._initial_body_boresight.copy()
        self.acquisition = AcquisitionState()

    def reset_tracking(self) -> None:
        self.body_boresight = self._initial_body_boresight.copy()
        self.acquisition = AcquisitionState()

    def _body_basis(self) -> tuple[Vec3, Vec3, Vec3]:
        theta, phi = spherical_angles_from_direction(self.body_boresight)
        return transmitter_basis(theta, phi)

    def _direction_in_body_frame(self, theta_offset: float, phi_offset: float) -> Vec3:
        u_x, u_y, u_z = self._body_basis()
        return direction_with_tangent_offset(
            u_z,
            u_x,
            u_y,
            theta_offset,
            phi_offset,
        )

    def dish_boresight_inertial(self) -> Vec3:
        """Dish aim in inertial coordinates (body frame + gimbal offset)."""
        return self._direction_in_body_frame(
            self.dish_theta_offset,
            self.dish_phi_offset,
        )

    def beam_boresight_inertial(self) -> Vec3:
        """Transmit beam nominal aim in inertial coordinates (body + beam offset)."""
        return self._direction_in_body_frame(
            self.beam_theta_offset,
            self.beam_phi_offset,
        )

    @property
    def initial_dish_boresight(self) -> Vec3:
        """Dish boresight at launch before any body slew."""
        u_x, u_y, u_z = transmitter_basis(
            *spherical_angles_from_direction(self._initial_body_boresight)
        )
        return direction_with_tangent_offset(
            u_z,
            u_x,
            u_y,
            self.dish_theta_offset,
            self.dish_phi_offset,
        )

    @property
    def initial_beam_boresight(self) -> Vec3:
        u_x, u_y, u_z = transmitter_basis(
            *spherical_angles_from_direction(self._initial_body_boresight)
        )
        return direction_with_tangent_offset(
            u_z,
            u_x,
            u_y,
            self.beam_theta_offset,
            self.beam_phi_offset,
        )

    @property
    def initial_pointing_offset(self) -> float:
        """Angular offset of the dish from true partner aim at startup (rad)."""
        return angle_between(self.toward_partner, self.initial_dish_boresight)

    @property
    def configured_dish_offset_magnitude(self) -> float:
        return float(np.hypot(self.dish_theta_offset, self.dish_phi_offset))

    @property
    def configured_body_offset_magnitude(self) -> float:
        return float(np.hypot(self.body_theta_offset, self.body_phi_offset))

    def dish_mount(self, body_radius: float) -> Vec3:
        dish = self.dish_boresight_inertial()
        return self.position + dish * body_radius

    def dish_mount_for_boresight(self, boresight: Vec3, body_radius: float) -> Vec3:
        return self.position + boresight * body_radius

    def observe_beam(self, in_cone: bool, beam_direction: Vec3, dq: float) -> Vec3:
        """
        On acquisition, slew the entire body toward the incoming beam.
        Dish and beam directions follow rigidly and may retain residual error.
        """
        acq = self.acquisition
        dish = self.dish_boresight_inertial()
        just_detected = False

        if in_cone and not acq.has_seen_beam:
            toward_source = -normalize(beam_direction)
            incident = angle_between(dish, toward_source)
            acq.incident_angle = incident
            acq.track_target = toward_source.copy()
            acq.has_seen_beam = True
            body_incident = angle_between(self.body_boresight, toward_source)
            if self.body_slew_time > 0.0:
                acq.slew_rate = body_incident / self.body_slew_time
            else:
                acq.slew_rate = float("inf")
            just_detected = True

        if acq.has_seen_beam and not just_detected and acq.track_target is not None:
            max_step = (acq.slew_rate or 0.0) * dq
            self.body_boresight = rotate_toward(
                self.body_boresight,
                acq.track_target,
                max_step,
            )

        return self.dish_boresight_inertial()
