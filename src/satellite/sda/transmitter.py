"""SDA transmitter spiral cone."""

from __future__ import annotations

import math

import numpy as np

from satellite.geometry import (
    cone_surface_mesh,
    global_spiral_frame,
    ribbon_swept_mesh,
    spiral_path_on_target_plane,
    spiral_trail_on_plane,
    transmitter_basis,
)
from satellite.math3d import Vec3, distance, normalize, spherical_angles_from_direction
from satellite.sda.boresight_table import BoresightTable


class TransmitterSDA:
    """Transmitter-side SDA spiral search around a believed boresight."""

    def __init__(
        self,
        position: Vec3,
        believed_boresight: Vec3,
        partner_actual: Vec3,
        k: float,
        alpha: float,
        beam_length: float,
        *,
        boresight_table: BoresightTable | None = None,
    ) -> None:
        self.position = position
        self.believed_boresight = normalize(believed_boresight)
        self.partner_actual = partner_actual
        self.k = k
        self.alpha = alpha
        self.w = k * alpha / math.pi

        direction = self.believed_boresight
        self.theta_0, self.phi_0 = spherical_angles_from_direction(direction)
        self.u_x, self.u_y, self.u_z = transmitter_basis(self.theta_0, self.phi_0)
        self.nominal_boresight = self.u_z

        self.believed_range = distance(position, partner_actual)
        self.actual_target_range = self.believed_range
        self.beam_length = beam_length
        self.boresight_table = boresight_table

    @classmethod
    def from_boresight(
        cls,
        position: Vec3,
        believed_boresight: Vec3,
        partner_actual: Vec3,
        k: float,
        alpha: float,
        beam_length: float | None = None,
        *,
        boresight_extension: float = 5.0,
        q_max: float | None = None,
        q_step: float | None = None,
    ) -> TransmitterSDA:
        link_range = distance(position, partner_actual)
        length = (
            beam_length
            if beam_length is not None
            else link_range + boresight_extension
        )
        tx = cls(position, believed_boresight, partner_actual, k, alpha, length)
        if q_max is not None and q_step is not None:
            tx.boresight_table = BoresightTable.build(tx.boresight_at, q_max, q_step)
        return tx

    # Backward-compat aliases
    @property
    def p1(self) -> Vec3:
        return self.position

    @property
    def pt(self) -> Vec3:
        return self.partner_actual

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
        if self.boresight_table is not None:
            return self.boresight_table.at(q)
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
            self.position,
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
            self.position,
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
        target_range: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """3D ribbon R(u,v) at target range along each boresight."""
        return ribbon_swept_mesh(
            self.position,
            q,
            self.frame_at,
            self.alpha,
            target_range if target_range is not None else self.actual_target_range,
            u_steps,
            v_steps,
        )

    def believed_target_at(self, q: float) -> Vec3:
        """Beam-axis point at believed range along boresight at q."""
        return self.position + self.believed_range * self.boresight_at(q)

    def spiral_point_at_actual_range(self, q: float) -> Vec3:
        """K cap center at actual-target height along boresight at q."""
        return self.position + self.actual_target_range * self.boresight_at(q)

    def spiral_trail(
        self,
        q_max: float,
        num_steps: int,
    ) -> np.ndarray:
        return spiral_trail_on_plane(
            self.position,
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
