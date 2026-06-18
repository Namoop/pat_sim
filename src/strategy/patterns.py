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

from satellite.envelope import ENVELOPE_COSINE, ENVELOPE_LINEAR, ENVELOPE_SMOOTH
from satellite.math.geometry import (
    spherical_angles_from_direction,
    transmitter_basis,
)
from satellite.math.math3d import Vec3, normalize


def basis_at_direction(center: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    theta_0, phi_0 = spherical_angles_from_direction(center)
    return transmitter_basis(theta_0, phi_0)


@njit(cache=True)
def scan_envelope_scale(
    local_t: float,
    ramp_duration: float,
    profile_id: int = ENVELOPE_SMOOTH,
) -> float:
    """Amplitude scale: 0 at t=0, 1 after ramp_duration."""
    if ramp_duration <= 0.0:
        return 1.0
    if local_t <= 0.0:
        return 0.0
    if local_t >= ramp_duration:
        return 1.0
    t = local_t / ramp_duration
    if profile_id == ENVELOPE_LINEAR:
        return t
    if profile_id == ENVELOPE_COSINE:
        return 0.5 * (1.0 - math.cos(math.pi * t))
    # smoothstep (default)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def aim_from_offsets(
    u_off: float,
    v_off: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    v_x = u_z[0] + u_off * u_x[0] + v_off * u_y[0]
    v_y = u_z[1] + u_off * u_x[1] + v_off * u_y[1]
    v_z = u_z[2] + u_off * u_x[2] + v_off * u_y[2]
    return normalize(np.array([v_x, v_y, v_z], dtype=np.float64))


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
    phase_offset: float = 0.0,
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
    
    theta_l, phi_l = w * u, k * u + phase_offset
    
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
    envelope_ramp: float = 0.0,
    envelope_profile: int = ENVELOPE_SMOOTH,
) -> Vec3:
    if duration <= 0.0:
        return normalize(u_z)
    angle = 2.0 * math.pi * local_t / duration
    scale = scan_envelope_scale(local_t, envelope_ramp, envelope_profile)
    offset = scale * radius * (math.cos(angle) * u_x + math.sin(angle) * u_y)
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
    envelope_ramp: float = 0.0,
    envelope_profile: int = ENVELOPE_SMOOTH,
) -> Vec3:
    if duration <= 0.0:
        return normalize(u_z)
    t = local_t / duration
    axis = math.cos(axis_angle) * u_x + math.sin(axis_angle) * u_y
    scale = scan_envelope_scale(local_t, envelope_ramp, envelope_profile)
    offset = scale * extent * (2.0 * t - 1.0) * axis
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
    envelope_ramp: float = 0.0,
    envelope_profile: int = ENVELOPE_SMOOTH,
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
    scale = scan_envelope_scale(local_t, envelope_ramp, envelope_profile)
    u_off = scale * (col - (n - 1) / 2.0) * spacing
    v_off = scale * (row - (n - 1) / 2.0) * spacing
    return aim_from_offsets(u_off, v_off, u_x, u_y, u_z)


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
    envelope_ramp: float = 0.0,
    envelope_profile: int = ENVELOPE_SMOOTH,
) -> Vec3:
    scale = scan_envelope_scale(local_t, envelope_ramp, envelope_profile)
    r = scale * A * math.cos(w2 * local_t)
    u_off = r * math.cos(w1 * local_t)
    v_off = r * math.sin(w1 * local_t)
    return aim_from_offsets(u_off, v_off, u_x, u_y, u_z)


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
    envelope_ramp: float = 0.0,
    envelope_profile: int = ENVELOPE_SMOOTH,
) -> Vec3:
    scale = scan_envelope_scale(local_t, envelope_ramp, envelope_profile)
    u_off = scale * A * math.sin(wx * local_t + delta)
    v_off = scale * A * math.sin(wy * local_t)
    return aim_from_offsets(u_off, v_off, u_x, u_y, u_z)


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
    envelope_ramp: float = 0.0,
    envelope_profile: int = ENVELOPE_SMOOTH,
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

    scale = scan_envelope_scale(local_t, envelope_ramp, envelope_profile)
    u_off *= scale
    v_off *= scale
        
    return aim_from_offsets(u_off, v_off, u_x, u_y, u_z)
