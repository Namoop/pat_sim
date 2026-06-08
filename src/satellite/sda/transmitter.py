"""SDA transmitter spiral cone."""

from __future__ import annotations

import math

import numpy as np

from satellite.geometry import (
    believed_direction,
    believed_distance,
    cone_surface_mesh,
    global_spiral_frame,
    spiral_path_on_target_plane,
    spiral_swept_area_mesh,
    spiral_trail_on_plane,
    transmitter_basis,
)
from satellite.math3d import Vec3, spherical_angles_from_direction


class TransmitterSDA:
    """Transmitter-side SDA spiral search around believed target P_2."""

    def __init__(
        self,
        p1: Vec3,
        p2: Vec3,
        k: float,
        alpha: float,
        beam_length: float | None = None,
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.alpha = alpha
        self.w = k * alpha / math.pi

        direction = believed_direction(p1, p2)
        self.theta_0, self.phi_0 = spherical_angles_from_direction(direction)
        self.u_x, self.u_y, self.u_z = transmitter_basis(self.theta_0, self.phi_0)
        self.nominal_boresight = self.u_z

        self.beam_length = (
            beam_length if beam_length is not None else believed_distance(p1, p2)
        )
        self.plane_distance = believed_distance(p1, p2)

    def frame_at(self, q: float) -> tuple[Vec3, Vec3, Vec3]:
        return global_spiral_frame(
            q,
            self.theta_0,
            self.phi_0,
            self.w,
            self.k,
            self.u_x,
            self.u_y,
            self.u_z,
        )

    def boresight_at(self, q: float) -> Vec3:
        a_s, _, _ = self.frame_at(q)
        return a_s

    def cone_mesh_at(
        self,
        q: float,
        u_steps: int,
        v_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_s, b_s, c_s = self.frame_at(q)
        return cone_surface_mesh(
            self.p1,
            a_s,
            b_s,
            c_s,
            self.alpha,
            self.beam_length,
            u_steps,
            v_steps,
        )

    def spiral_path_up_to(self, q: float, num_steps: int) -> np.ndarray:
        return spiral_path_on_target_plane(
            self.p1,
            q,
            self.boresight_at,
            self.nominal_boresight,
            self.plane_distance,
            num_steps,
        )

    def swept_area_mesh_up_to(
        self,
        q: float,
        num_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return spiral_swept_area_mesh(
            self.p1,
            q,
            self.boresight_at,
            self.nominal_boresight,
            self.plane_distance,
            self.alpha,
            num_steps,
        )

    def spiral_trail(
        self,
        q_max: float,
        num_steps: int,
    ) -> np.ndarray:
        return spiral_trail_on_plane(
            self.p1,
            q_max,
            self.boresight_at,
            self.plane_distance,
            num_steps,
            plane_normal=self.nominal_boresight,
        )

    def theta_offset(self, q: float) -> float:
        return self.w * q

    def phi_offset(self, q: float) -> float:
        return self.k * q
