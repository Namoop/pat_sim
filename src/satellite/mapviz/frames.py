"""Tangent-plane projection for angular map visualization."""

from __future__ import annotations

import numpy as np

from satellite.geometry import transmitter_basis
from satellite.math3d import Vec3, cross, dot, norm, normalize, spherical_angles_from_direction
from satellite.sda.transmitter import TransmitterSDA


def direction_to_tangent_angles(origin: Vec3, direction: Vec3) -> tuple[float, float]:
    """
    Return (theta, phi) of direction in the tangent plane at origin.

    Inverse of direction_with_tangent_offset: w = theta*u_x + phi*u_y rotates
    origin toward direction by |w| around cross(origin, w).
    """
    base = normalize(origin)
    aim = normalize(direction)
    tilt = float(np.arccos(np.clip(dot(base, aim), -1.0, 1.0)))
    if tilt < 1e-15:
        return 0.0, 0.0
    axis_perp = cross(base, aim)
    if norm(axis_perp) < 1e-15:
        return 0.0, 0.0
    n = normalize(axis_perp)
    w = tilt * cross(n, base)
    u_x, u_y, _ = transmitter_basis(*spherical_angles_from_direction(base))
    return float(dot(w, u_x)), float(dot(w, u_y))


def tangent_disc(
    center_theta: float,
    center_phi: float,
    radius: float,
    segments: int,
) -> np.ndarray:
    """Circle boundary in (theta, phi) for beam / FOV rendering."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float64)
    return np.column_stack(
        (
            center_theta + radius * np.cos(angles),
            center_phi + radius * np.sin(angles),
        )
    )


def point_in_disc(
    point: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> bool:
    """Small-angle disc membership in tangent plane."""
    d_theta = point[0] - center[0]
    d_phi = point[1] - center[1]
    return float(np.hypot(d_theta, d_phi)) <= radius + 1e-12


def spiral_trail_in_map(
    origin: Vec3,
    tx: TransmitterSDA,
    local_t_end: float,
    num_steps: int,
) -> np.ndarray:
    """Sample transmitter boresight from local t=0 through local_t_end."""
    if local_t_end <= 0.0 or num_steps < 2:
        beam = tx.boresight_at(0.0)
        theta, phi = direction_to_tangent_angles(origin, beam)
        return np.array([[theta, phi]], dtype=np.float64)

    ts = np.linspace(0.0, local_t_end, num_steps, dtype=np.float64)
    points = np.empty((num_steps, 2), dtype=np.float64)
    for i, u in enumerate(qs):
        beam = tx.boresight_at(float(u))
        points[i] = direction_to_tangent_angles(origin, beam)
    return points
