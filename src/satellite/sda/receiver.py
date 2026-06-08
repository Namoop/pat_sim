"""Receiver-side SDA stubs (wobble frame, limited aperture, dish tracking)."""

from __future__ import annotations

from dataclasses import dataclass

from satellite.geometry import (
    actual_target_plane_distance,
    cone_surface_mesh,
    direction_with_local_offset,
    dish_mesh,
    receiver_global_frame,
    receiver_basis,
)
from satellite.math3d import Vec3, angle_between, normalize


@dataclass
class ReceiverDishState:
    boresight: Vec3
    incident_angle: float | None = None
    has_seen_beam: bool = False


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
        body_radius: float = 0.1,
        dish_radius: float = 0.08,
        dish_depth: float = 0.04,
    ) -> None:
        self.p1 = p1
        self.pt = pt
        self.gamma = gamma
        self.beta = beta
        self.omega_r = omega_r
        self.l_r = l_r
        self.body_radius = body_radius
        self.dish_radius = dish_radius
        self.dish_depth = dish_depth

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

    def reset_dish_tracking(self) -> None:
        self.dish = ReceiverDishState(boresight=self._initial_boresight.copy())

    @property
    def dish_mount(self) -> Vec3:
        """Dish mount on the receiver body surface along current boresight."""
        return self.pt + self.dish.boresight * self.body_radius

    def observe_beam(self, in_cone: bool, beam_direction: Vec3) -> Vec3:
        """
        Update dish orientation when the beam is visible.

        beam_direction is the transmitter boresight (incoming beam axis).
        Detection still uses the existing in-cone test; the dish does not
        affect that yet.
        """
        incoming = normalize(beam_direction)
        if in_cone:
            if not self.dish.has_seen_beam:
                self.dish.incident_angle = angle_between(self.dish.boresight, incoming)
                self.dish.has_seen_beam = True
            self.dish.boresight = incoming
        return self.dish.boresight

    def dish_mesh_at(
        self,
        u_steps: int = 12,
        v_steps: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        return dish_mesh(
            self.dish_mount,
            self.dish.boresight,
            self.dish_radius,
            self.dish_depth,
            u_steps,
            v_steps,
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
