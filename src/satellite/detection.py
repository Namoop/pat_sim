"""Hit detection helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.math3d import Vec3, dot, normalize, norm


def alignment_dot(target_direction: Vec3, boresight: Vec3) -> float:
    """Desmos f_dot — cosine of angle between target and boresight."""
    return dot(target_direction, boresight)


def is_in_cone(alignment: float, alpha: float) -> bool:
    """True when target falls within cone half-angle alpha."""
    return alignment >= np.cos(alpha)


def cone_intersects_sphere(
    apex: Vec3,
    axis: Vec3,
    alpha: float,
    length: float,
    center: Vec3,
    radius: float,
) -> bool:
    """True when a finite cone and sphere overlap."""
    a = normalize(axis)
    oc = center - apex
    t = dot(oc, a)
    oc_sq = dot(oc, oc)
    r = radius

    if oc_sq <= r * r:
        return True

    if t + r < 0.0 or t - r > length:
        return False

    tan_a = np.tan(alpha)

    t_clamped = min(max(t, 0.0), length)
    h_sq = max(0.0, oc_sq - t_clamped * t_clamped)
    h = float(np.sqrt(h_sq))
    if h <= t_clamped * tan_a + r:
        return True

    if t >= length - r:
        cap_center = apex + length * a
        vc = center - cap_center
        h_cap_sq = max(0.0, dot(vc, vc) - dot(vc, a) ** 2)
        if float(np.sqrt(h_cap_sq)) <= length * tan_a + r:
            return True

    if t < r:
        h_apex_sq = max(0.0, oc_sq - t * t)
        if float(np.sqrt(h_apex_sq)) <= r:
            return True

    return False


def beam_hits_receiver_at_q(
    q: float,
    apex: Vec3,
    receiver_center: Vec3,
    receiver_radius: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> bool:
    """True when the transmitter cone at q intersects the receiver body."""
    return cone_intersects_sphere(
        apex,
        boresight_fn(q),
        alpha,
        beam_length,
        receiver_center,
        receiver_radius,
    )


def scan_beam_hits_up_to(
    q_max: float,
    q_step: float,
    apex: Vec3,
    receiver_center: Vec3,
    receiver_radius: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> tuple[bool, float | None]:
    """
    Scan from 0 to q_max for cone-receiver intersection.

    Returns (hit, hit_at_q). hit_at_q is the first q where the beam cone
    collides with the receiver sphere.
    """
    hit_at_q: float | None = None
    q = 0.0
    while q <= q_max + 1e-12:
        if beam_hits_receiver_at_q(
            q,
            apex,
            receiver_center,
            receiver_radius,
            boresight_fn,
            alpha,
            beam_length,
        ):
            hit_at_q = q
            break
        q += q_step

    return hit_at_q is not None, hit_at_q


def in_cone_at_q(
    q: float,
    target_direction: Vec3,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
) -> bool:
    """Legacy point-direction test — target center within cone half-angle."""
    boresight = boresight_fn(q)
    return is_in_cone(alignment_dot(target_direction, boresight), alpha)


def scan_hits_up_to(
    q_max: float,
    q_step: float,
    target_direction: Vec3,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
) -> tuple[bool, float | None]:
    """
    Legacy scan — first q where target direction enters cone aperture.
    """
    hit_at_q: float | None = None
    q = 0.0
    while q <= q_max + 1e-12:
        if in_cone_at_q(q, target_direction, boresight_fn, alpha):
            hit_at_q = q
            break
        q += q_step

    return hit_at_q is not None, hit_at_q
