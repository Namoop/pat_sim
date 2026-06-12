"""Optional Numba-accelerated detection helpers."""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _beam_hits_dish_pure_python(
    apex: np.ndarray,
    dish_mount: np.ndarray,
    dish_boresight: np.ndarray,
    cos_fov: float,
    beam_axis: np.ndarray,
    cos_alpha: float,
    beam_length: float,
) -> bool:
    # Vector from transmitter apex to dish mount
    px = dish_mount[0] - apex[0]
    py = dish_mount[1] - apex[1]
    pz = dish_mount[2] - apex[2]
    dist_sq = px * px + py * py + pz * pz
    if dist_sq <= 0.0:
        return False
    dist = math.sqrt(dist_sq)
    if dist > beam_length:
        return False

    # tx, ty, tz is direction FROM transmitter TO receiver (for transmitter cone)
    tx, ty, tz = px / dist, py / dist, pz / dist

    # For receiver FOV, we need direction FROM receiver TO transmitter (incoming)
    ix, iy, iz = -tx, -ty, -tz

    dx, dy, dz = dish_boresight[0], dish_boresight[1], dish_boresight[2]
    dnorm = math.sqrt(dx * dx + dy * dy + dz * dz)
    dot_dt = (dx / dnorm) * ix + (dy / dnorm) * iy + (dz / dnorm) * iz

    # incident angle <= dish_fov means cos(incident) >= cos(dish_fov)
    if dot_dt < cos_fov:
        return False

    bx, by, bz = beam_axis[0], beam_axis[1], beam_axis[2]
    bnorm = math.sqrt(bx * bx + by * by + bz * bz)
    dot_bp = (bx / bnorm) * tx + (by / bnorm) * ty + (bz / bnorm) * tz

    return dot_bp >= cos_alpha


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
        px, py, pz = dish_mount[0] - apex[0], dish_mount[1] - apex[1], dish_mount[2] - apex[2]
        dist_sq = px * px + py * py + pz * pz
        if dist_sq <= 0.0:
            return False
        dist = dist_sq**0.5
        if dist > beam_length:
            return False
        
        tx, ty, tz = px / dist, py / dist, pz / dist
        ix, iy, iz = -tx, -ty, -tz

        dx, dy, dz = dish_boresight[0], dish_boresight[1], dish_boresight[2]
        dnorm = (dx * dx + dy * dy + dz * dz) ** 0.5
        dot_dt = (dx / dnorm) * ix + (dy / dnorm) * iy + (dz / dnorm) * iz

        if dot_dt < cos_fov:
            return False
            
        bx, by, bz = beam_axis[0], beam_axis[1], beam_axis[2]
        bnorm = (bx * bx + by * by + bz * bz) ** 0.5
        dot_bp = (bx / bnorm) * tx + (by / bnorm) * ty + (bz / bnorm) * tz
        return dot_bp >= cos_alpha

else:
    beam_hits_dish_fast = _beam_hits_dish_pure_python


def beam_hits_dish_optimized(
    apex: np.ndarray,
    dish_mount: np.ndarray,
    dish_boresight: np.ndarray,
    dish_fov: float,
    beam_axis: np.ndarray,
    alpha: float,
    beam_length: float,
) -> bool:
    """Detection using Numba when compiled, else pure Python fallback."""
    cos_fov = math.cos(dish_fov)
    cos_alpha = math.cos(alpha)
    return bool(
        beam_hits_dish_fast(
            apex,
            dish_mount,
            dish_boresight,
            cos_fov,
            beam_axis,
            cos_alpha,
            float(beam_length),
        )
    )
