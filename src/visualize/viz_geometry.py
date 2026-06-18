"""Cone mesh from an arbitrary bench aim direction."""

from __future__ import annotations

import numpy as np

from satellite.math.geometry import axis_perpendicular_basis, cone_surface_mesh
from satellite.math.math3d import Vec3, normalize


def cone_mesh_for_aim(
    apex: Vec3,
    aim: Vec3,
    alpha: float,
    beam_length: float,
    u_steps: int,
    v_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(aim)
    b_axis, c_axis = axis_perpendicular_basis(axis)
    return cone_surface_mesh(
        apex,
        axis,
        b_axis,
        c_axis,
        alpha,
        beam_length,
        u_steps,
        v_steps,
    )
