"""Receiver-side SDA stubs (wobble frame, limited aperture)."""

from __future__ import annotations

import numpy as np

from satellite.geometry import (
    actual_target_plane_distance,
    cone_surface_mesh,
    receiver_global_frame,
    receiver_basis,
)
from satellite.math3d import Vec3


class ReceiverSDA:
    """Receiver at actual position P_t — stubs for full aperture logic."""

    def __init__(
        self,
        p1: Vec3,
        pt: Vec3,
        gamma: float,
        beta: float,
        omega_r: float,
        l_r: float,
    ) -> None:
        self.p1 = p1
        self.pt = pt
        self.gamma = gamma
        self.beta = beta
        self.omega_r = omega_r
        self.l_r = l_r
        self.d_circ = actual_target_plane_distance(p1, pt, l_r)

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
