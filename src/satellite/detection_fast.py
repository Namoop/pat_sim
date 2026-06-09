"""Optional Numba-accelerated detection helpers."""

from __future__ import annotations

import numpy as np

from satellite.math3d import Vec3, angle_between, dot, norm, normalize

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _beam_hits_dish_numpy(
    apex: np.ndarray,
    dish_mount: np.ndarray,
    dish_boresight: np.ndarray,
    cos_fov: float,
    beam_axis: np.ndarray,
    cos_alpha: float,
    beam_length: float,
) -> bool:
    toward = -beam_axis / norm(beam_axis)
    dish_u = dish_boresight / norm(dish_boresight)
    if float(np.arccos(np.clip(np.dot(dish_u, toward), -1.0, 1.0))) > np.arccos(cos_fov):
        return False
    to_point = dish_mount - apex
    dist = norm(to_point)
    if dist <= 0.0 or dist > beam_length:
        return False
    direction = to_point / dist
    beam_u = beam_axis / norm(beam_axis)
    return float(np.dot(beam_u, direction)) >= cos_alpha


if _HAS_NUMBA:

    @njit(cache=True)
    def beam_hits_dish_fast(
        apex: np.ndarray,
        dish_mount: np.ndarray,
        dish_boresight: np.ndarray,
        cos_fov: float,
        beam_axis: np.ndarray,
        cos_alpha: float,
        beam_length: float,
    ) -> bool:
        bx, by, bz = beam_axis[0], beam_axis[1], beam_axis[2]
        bnorm = (bx * bx + by * by + bz * bz) ** 0.5
        tx, ty, tz = -bx / bnorm, -by / bnorm, -bz / bnorm
        dx, dy, dz = dish_boresight[0], dish_boresight[1], dish_boresight[2]
        dnorm = (dx * dx + dy * dy + dz * dz) ** 0.5
        dx, dy, dz = dx / dnorm, dy / dnorm, dz / dnorm
        dot_dt = dx * tx + dy * ty + dz * tz
        if dot_dt < -1.0:
            dot_dt = -1.0
        elif dot_dt > 1.0:
            dot_dt = 1.0
        incident = np.arccos(dot_dt)
        if incident > np.arccos(cos_fov):
            return False
        px, py, pz = dish_mount[0] - apex[0], dish_mount[1] - apex[1], dish_mount[2] - apex[2]
        dist = (px * px + py * py + pz * pz) ** 0.5
        if dist <= 0.0 or dist > beam_length:
            return False
        px, py, pz = px / dist, py / dist, pz / dist
        dot_bp = (bx / bnorm) * px + (by / bnorm) * py + (bz / bnorm) * pz
        return dot_bp >= cos_alpha

else:
    beam_hits_dish_fast = _beam_hits_dish_numpy


def beam_hits_dish_optimized(
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    dish_fov: float,
    beam_axis: Vec3,
    alpha: float,
    beam_length: float,
) -> bool:
    """Detection using Numba when compiled, else NumPy array kernel."""
    cos_fov = float(np.cos(dish_fov))
    cos_alpha = float(np.cos(alpha))
    return bool(
        beam_hits_dish_fast(
            np.asarray(apex, dtype=np.float64),
            np.asarray(dish_mount, dtype=np.float64),
            np.asarray(dish_boresight, dtype=np.float64),
            cos_fov,
            np.asarray(beam_axis, dtype=np.float64),
            cos_alpha,
            float(beam_length),
        )
    )


# Only export optimized path when Numba is installed.
if not _HAS_NUMBA:
    beam_hits_dish_optimized = None  # type: ignore[assignment,misc]
