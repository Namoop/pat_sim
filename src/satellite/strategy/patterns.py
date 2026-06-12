"""Tangent-plane aim path math for movement patterns."""

from __future__ import annotations

import math

import numpy as np

from satellite.geometry import (
    local_spiral_angles,
    spherical_angles_from_direction,
    spherical_to_cartesian,
    transform_local_to_global,
    transmitter_basis,
)
from satellite.math3d import Vec3, normalize


def basis_at_direction(center: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    theta_0, phi_0 = spherical_angles_from_direction(center)
    return transmitter_basis(theta_0, phi_0)


def spiral_aim_at(
    local_t: float,
    *,
    w: float,
    k: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
    max_radius: float,
    speed: float = 1.0,
) -> Vec3:
    """Archimedean spiral aim; angular radius grows as w·u, capped at max_radius."""
    if w <= 0.0:
        # Use a copy to avoid mutating the basis vector
        return u_z.copy()
    u = speed * local_t
    if w * u > max_radius:
        u = max_radius / w
    
    # Optimization: Calculate only the aim vector (A_l) as pure floats
    theta_l, phi_l = w * u, k * u
    
    # spherical_to_cartesian unrolled
    sin_theta = math.sin(theta_l)
    cos_theta = math.cos(theta_l)
    sin_phi = math.sin(phi_l)
    cos_phi = math.cos(phi_l)
    
    al0 = sin_theta * cos_phi
    al1 = sin_theta * sin_phi
    al2 = cos_theta
    
    # transform_local_to_global unrolled
    # result = al0 * u_x + al1 * u_y + al2 * u_z
    asx = al0 * u_x[0] + al1 * u_y[0] + al2 * u_z[0]
    asy = al0 * u_x[1] + al1 * u_y[1] + al2 * u_z[1]
    asz = al0 * u_x[2] + al1 * u_y[2] + al2 * u_z[2]
    
    # normalize unrolled
    norm = math.sqrt(asx * asx + asy * asy + asz * asz)
    return np.array([asx / norm, asy / norm, asz / norm], dtype=np.float64)


def circle_aim_at(
    local_t: float,
    *,
    duration: float,
    radius: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    if duration <= 0.0:
        return normalize(u_z)
    angle = 2.0 * math.pi * local_t / duration
    offset = radius * (math.cos(angle) * u_x + math.sin(angle) * u_y)
    return normalize(u_z + offset)


def line_aim_at(
    local_t: float,
    *,
    duration: float,
    extent: float,
    axis_angle: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    if duration <= 0.0:
        return normalize(u_z)
    t = local_t / duration
    axis = math.cos(axis_angle) * u_x + math.sin(axis_angle) * u_y
    offset = extent * (2.0 * t - 1.0) * axis
    return normalize(u_z + offset)


def grid_aim_at(
    local_t: float,
    *,
    duration: float,
    spacing: float,
    extent: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    if duration <= 0.0 or spacing <= 0.0:
        return normalize(u_z)
    n = max(1, int(math.ceil(extent / spacing)))
    total_cells = n * n
    progress = min(local_t / duration, 1.0 - 1e-12)
    idx = int(progress * total_cells)
    idx = min(idx, total_cells - 1)
    row = idx // n
    col = idx % n
    if row % 2 == 1:
        col = n - 1 - col
    u_off = (col - (n - 1) / 2.0) * spacing
    v_off = (row - (n - 1) / 2.0) * spacing
    return normalize(u_z + u_off * u_x + v_off * u_y)
