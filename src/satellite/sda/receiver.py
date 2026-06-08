"""Receiver-side SDA stubs (wobble frame, limited aperture, dish tracking)."""

from __future__ import annotations

from dataclasses import dataclass

from satellite.geometry import (
    actual_target_plane_distance,
    cone_surface_mesh,
    direction_with_local_offset,
    dish_aperture_radius,
    dish_disc_mesh,
    receiver_global_frame,
    receiver_basis,
)
from satellite.math3d import Vec3, angle_between, distance, normalize, rotate_toward


@dataclass
class ReceiverDishState:
    boresight: Vec3
    incident_angle: float | None = None
    has_seen_beam: bool = False
    slew_rate: float | None = None
    track_target: Vec3 | None = None


class ReceiverSDA:
    """Receiver at actual position P_t — dish tracking and aperture stubs."""

    def __init__(
        self,
        p1: Vec3,
        pt: Vec3,
        gamma: float,
        beta: float,
        omega_r: float,
        l_r: float,
        dish_theta_offset: float = 0.08,
        dish_phi_offset: float = 0.06,
        body_radius: float = 0.5,
        dish_fov: float = 0.002,
        dish_slew_time: float = 0.3,
    ) -> None:
        self.p1 = p1
        self.pt = pt
        self.gamma = gamma
        self.beta = beta
        self.omega_r = omega_r
        self.l_r = l_r
        self.body_radius = body_radius
        self.dish_fov = dish_fov
        self.dish_slew_time = dish_slew_time

        self.d_circ = actual_target_plane_distance(p1, pt, l_r)

        toward_p1 = normalize(p1 - pt)
        initial_boresight = direction_with_local_offset(
            toward_p1,
            dish_theta_offset,
            dish_phi_offset,
        )
        self._initial_boresight = initial_boresight.copy()
        self.dish = ReceiverDishState(boresight=initial_boresight)

    @property
    def dish_state(self) -> ReceiverDishState:
        return self.dish

    @property
    def initial_dish_mount(self) -> Vec3:
        return self.pt + self._initial_boresight * self.body_radius

    @property
    def initial_dish_boresight(self) -> Vec3:
        return self._initial_boresight

    def reset_dish_tracking(self) -> None:
        self.dish = ReceiverDishState(boresight=self._initial_boresight.copy())

    @property
    def dish_mount(self) -> Vec3:
        """Dish mount on the receiver body surface along current boresight."""
        return self.pt + self.dish.boresight * self.body_radius

    def dish_range_from_p1(self) -> float:
        """Range from P_1 to the dish mount on the receiver surface."""
        return distance(self.p1, self.dish_mount)

    def dish_aperture_radius(self) -> float:
        return dish_aperture_radius(self.p1, self.dish_mount, self.dish_fov)

    def observe_beam(self, in_cone: bool, beam_direction: Vec3, dq: float) -> Vec3:
        """
        Update dish orientation after beam detection.

        beam_direction is the transmitter boresight (beam emission axis from P_1).
        On first collision the incident angle and a fixed track target are recorded;
        slewing begins on the next step toward that target.
        """
        just_detected = False
        if in_cone and not self.dish.has_seen_beam:
            toward_source = -normalize(beam_direction)
            incident = angle_between(self.dish.boresight, toward_source)
            self.dish.incident_angle = incident
            self.dish.track_target = toward_source.copy()
            self.dish.has_seen_beam = True
            if self.dish_slew_time > 0.0:
                self.dish.slew_rate = incident / self.dish_slew_time
            else:
                self.dish.slew_rate = float("inf")
            just_detected = True

        if (
            self.dish.has_seen_beam
            and not just_detected
            and self.dish.track_target is not None
        ):
            max_step = (self.dish.slew_rate or 0.0) * dq
            self.dish.boresight = rotate_toward(
                self.dish.boresight,
                self.dish.track_target,
                max_step,
            )
        return self.dish.boresight

    def dish_mesh_at(
        self,
        segments: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        return dish_disc_mesh(
            self.dish_mount,
            self.dish.boresight,
            self.dish_aperture_radius(),
            segments,
        )

    def frame_at(self, t: float) -> tuple[Vec3, Vec3, Vec3]:
        return receiver_global_frame(self.pt, self.p1, self.beta, self.omega_r, t)

    def receiver_cone_mesh_at(
        self,
        q: float,
        u_steps: int,
        v_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Desmos S_rec — receiver-side cone stub."""
        a_rec, b_rec, c_rec = self.frame_at(q)
        return cone_surface_mesh(
            self.pt,
            a_rec,
            b_rec,
            c_rec,
            self.gamma,
            self.l_r,
            u_steps,
            v_steps,
        )

    def receiver_cap_mesh_at(
        self,
        q: float,
        u_steps: int,
        v_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Desmos K_rec — circular cap at L_r (stub, same geometry as S_rec cap)."""
        a_rec, b_rec, c_rec = self.frame_at(q)
        return cone_surface_mesh(
            self.pt,
            a_rec,
            b_rec,
            c_rec,
            self.gamma,
            self.l_r,
            u_steps,
            v_steps,
        )

    def is_target_in_aperture(self, _direction: Vec3, _t: float) -> bool:
        """Placeholder for limited receiver area logic."""
        raise NotImplementedError("Receiver aperture detection not yet implemented")

    def basis(self) -> tuple[Vec3, Vec3, Vec3]:
        return receiver_basis(self.pt, self.p1)
