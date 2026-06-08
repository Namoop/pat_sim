"""Hit detection helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.geometry import dish_aperture_radius
from satellite.math3d import Vec3, dot, normalize


def alignment_dot(target_direction: Vec3, boresight: Vec3) -> float:
    """Desmos f_dot — cosine of angle between target and boresight."""
    return dot(target_direction, boresight)


def is_in_cone(alignment: float, alpha: float) -> bool:
    """True when target falls within cone half-angle alpha."""
    return alignment >= np.cos(alpha)


def cone_intersects_disc(
    apex: Vec3,
    axis: Vec3,
    alpha: float,
    length: float,
    center: Vec3,
    normal: Vec3,
    radius: float,
) -> bool:
    """True when a finite cone overlaps a flat disc."""
    a = normalize(axis)
    n = normalize(normal)
    tan_a = np.tan(alpha)

    oc = center - apex
    t = dot(oc, a)
    oc_sq = dot(oc, oc)
    h_sq = max(0.0, oc_sq - t * t)
    h = float(np.sqrt(h_sq))

    if t + radius < 0.0 or t - radius > length:
        return False

    t_clamped = min(max(t, 0.0), length)
    h_clamped_sq = max(0.0, oc_sq - t_clamped * t_clamped)
    h_clamped = float(np.sqrt(h_clamped_sq))
    if h_clamped <= t_clamped * tan_a + radius:
        return True

    denom = dot(a, n)
    if abs(denom) < 1e-12:
        return False

    t_plane = dot(center - apex, n) / denom
    if t_plane < 0.0 or t_plane > length:
        return False

    axis_on_plane = apex + t_plane * a
    to_center = center - axis_on_plane
    d_plane = float(
        np.sqrt(max(0.0, dot(to_center, to_center) - dot(to_center, n) ** 2))
    )
    if d_plane <= t_plane * tan_a + radius:
        return True

    if h <= radius and 0.0 <= t <= length:
        return True

    return False


def beam_hits_dish_at_q(
    q: float,
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    body_radius: float,
    dish_fov: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> bool:
    """True when the transmitter cone at q intersects the receiver dish disc."""
    radius = dish_aperture_radius(body_radius, dish_fov)
    return cone_intersects_disc(
        apex,
        boresight_fn(q),
        alpha,
        beam_length,
        dish_mount,
        dish_boresight,
        radius,
    )


def scan_dish_hits_up_to(
    q_max: float,
    q_step: float,
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    body_radius: float,
    dish_fov: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> tuple[bool, float | None]:
    """
    Scan from 0 to q_max for cone-dish intersection.

    Returns (hit, hit_at_q). hit_at_q is the first q where the beam cone
    collides with the receiver dish disc.
    """
    hit_at_q: float | None = None
    q = 0.0
    while q <= q_max + 1e-12:
        if beam_hits_dish_at_q(
            q,
            apex,
            dish_mount,
            dish_boresight,
            body_radius,
            dish_fov,
            boresight_fn,
            alpha,
            beam_length,
        ):
            hit_at_q = q
            break
        q += q_step

    return hit_at_q is not None, hit_at_q
