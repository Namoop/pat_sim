"""SDA transmitter spiral cone."""

from __future__ import annotations

import math

import numpy as np

from satellite.geometry import (
    believed_direction,
    believed_distance,
    cone_surface_mesh,
    global_spiral_frame,
    ribbon_swept_mesh,
    spiral_path_on_target_plane,
    spiral_trail_on_plane,
    transmitter_basis,
)
from satellite.math3d import Vec3, distance, spherical_angles_from_direction


class TransmitterSDA:
    """Transmitter-side SDA spiral search around believed target P_2."""

    def __init__(
        self,
        p1: Vec3,
        p2: Vec3,
        pt: Vec3,
        k: float,
        alpha: float,
        beam_length: float | None = None,
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.pt = pt
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
        self.believed_range = believed_distance(p1, p2)
        self.actual_target_range = distance(p1, pt)

    def frame_at(self, q: float) -> tuple[Vec3, Vec3, Vec3]:
        return global_spiral_frame(
            q,
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
            self.believed_range,
            num_steps,
        )

    def swept_area_mesh_up_to(
        self,
        q: float,
        u_steps: int,
        v_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """3D ribbon R(u,v) at actual-target range |P_t - P_1|."""
        return ribbon_swept_mesh(
            self.p1,
            q,
            self.frame_at,
            self.alpha,
            self.actual_target_range,
            u_steps,
            v_steps,
        )

    def believed_target_at(self, q: float) -> Vec3:
        """Beam-axis point at believed range — equals P_2 at q=0."""
        return self.p1 + self.believed_range * self.boresight_at(q)

    def spiral_point_at_actual_range(self, q: float) -> Vec3:
        """K cap center at actual-target height along boresight at q."""
        return self.p1 + self.actual_target_range * self.boresight_at(q)

    def spiral_trail(
        self,
        q_max: float,
        num_steps: int,
    ) -> np.ndarray:
        return spiral_trail_on_plane(
            self.p1,
            q_max,
            self.boresight_at,
            self.believed_range,
            num_steps,
            plane_normal=self.nominal_boresight,
        )

    def theta_offset(self, q: float) -> float:
        return self.w * q

    def phi_offset(self, q: float) -> float:
        return self.k * q
