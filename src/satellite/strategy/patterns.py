"""Tangent-plane aim path math for movement patterns optimized with Numba."""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from satellite.geometry import (
    spherical_angles_from_direction,
    transmitter_basis,
)
from satellite.math3d import Vec3, normalize


def basis_at_direction(center: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    theta_0, phi_0 = spherical_angles_from_direction(center)
    return transmitter_basis(theta_0, phi_0)


@njit(cache=True)
def spiral_aim_at(
    local_t: float,
    *,
    w: float,
    k: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
    max_radius: float,
    max_beam_speed: float,
) -> Vec3:
    """Archimedean spiral aim with constant linear speed along the path, preserving duration."""
    if w <= 0.0:
        return u_z.copy()
    
    effective_speed = max_beam_speed
    
    if k <= 0.0:
        u = effective_speed * local_t
    else:
        Y = (k * effective_speed * local_t) / w
        if Y <= 0.0:
            u = 0.0
        else:
            x = math.sqrt(2.0 * Y) if Y > 2.0 else Y
            for _ in range(3):
                sqrt_term = math.sqrt(1.0 + x * x)
                h_x = 0.5 * (x * sqrt_term + math.log(x + sqrt_term))
                diff = h_x - Y
                x = x - diff / sqrt_term
            u = x / k

    if w * u > max_radius:
        u = max_radius / w
    
    theta_l, phi_l = w * u, k * u
    
    sin_theta = math.sin(theta_l)
    cos_theta = math.cos(theta_l)
    sin_phi = math.sin(phi_l)
    cos_phi = math.cos(phi_l)
    
    al0 = sin_theta * cos_phi
    al1 = sin_theta * sin_phi
    al2 = cos_theta
    
    asx = al0 * u_x[0] + al1 * u_y[0] + al2 * u_z[0]
    asy = al0 * u_x[1] + al1 * u_y[1] + al2 * u_z[1]
    asz = al0 * u_x[2] + al1 * u_y[2] + al2 * u_z[2]
    
    norm = (asx * asx + asy * asy + asz * asz) ** 0.5
    return np.array([asx / norm, asy / norm, asz / norm], dtype=np.float64)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def rosette_aim_at(
    local_t: float,
    *,
    A: float,
    w1: float,
    w2: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    r = A * math.cos(w2 * local_t)
    u_off = r * math.cos(w1 * local_t)
    v_off = r * math.sin(w1 * local_t)
    return normalize(u_z + u_off * u_x + v_off * u_y)


@njit(cache=True)
def lissajous_aim_at(
    local_t: float,
    *,
    A: float,
    wx: float,
    wy: float,
    delta: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    u_off = A * math.sin(wx * local_t + delta)
    v_off = A * math.sin(wy * local_t)
    return normalize(u_z + u_off * u_x + v_off * u_y)


@njit(cache=True)
def raster_aim_at(
    local_t: float,
    *,
    duration: float,
    radius: float,
    steps: int,
    horizontal: bool = True,
    serpentine: bool = True,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    if duration <= 0.0 or steps <= 1:
        return normalize(u_z)

    p = local_t / duration
    line_progress = p * steps
    line_idx = min(int(line_progress), steps - 1)
    t_line = line_progress - line_idx
    
    line_offset = -radius + (line_idx / (steps - 1)) * 2.0 * radius
    chord_half_length = math.sqrt(max(0.0, radius**2 - line_offset**2))
    
    if serpentine and line_idx % 2 == 1:
        scan_offset = chord_half_length * (1.0 - 2.0 * t_line)
    else:
        scan_offset = chord_half_length * (2.0 * t_line - 1.0)
        
    if horizontal:
        u_off = scan_offset
        v_off = line_offset
    else:
        u_off = line_offset
        v_off = scan_offset
        
    return normalize(u_z + u_off * u_x + v_off * u_y)
