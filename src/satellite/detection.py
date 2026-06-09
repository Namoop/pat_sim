"""Hit detection helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.math3d import Vec3, angle_between, dot, norm, normalize


def alignment_dot(target_direction: Vec3, boresight: Vec3) -> float:
    """Desmos f_dot — cosine of angle between target and boresight."""
    return dot(target_direction, boresight)


def is_in_cone(alignment: float, alpha: float) -> bool:
    """True when target falls within cone half-angle alpha."""
    return alignment >= np.cos(alpha)


def is_within_dish_fov(
    dish_boresight: Vec3,
    incoming_direction: Vec3,
    dish_fov: float,
) -> bool:
    """True when incoming beam lies within the dish full FOV (radians)."""
    return angle_between(dish_boresight, incoming_direction) <= dish_fov


def point_in_transmitter_cone(
    apex: Vec3,
    axis: Vec3,
    alpha: float,
    length: float,
    point: Vec3,
) -> bool:
    """True when a point lies inside the finite transmitter cone."""
    to_point = point - apex
    dist = norm(to_point)
    if dist <= 0.0 or dist > length:
        return False
    direction = to_point / dist
    return dot(normalize(axis), direction) >= np.cos(alpha)


def beam_incident_angle_at_q(
    q: float,
    dish_boresight: Vec3,
    boresight_fn: Callable[[float], Vec3],
) -> float:
    """Angle between dish boresight and incoming beam at q."""
    toward_source = -normalize(boresight_fn(q))
    return angle_between(dish_boresight, toward_source)


def beam_illuminates_dish_mount_at_q(
    q: float,
    apex: Vec3,
    dish_mount: Vec3,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> bool:
    """True when the transmitter cone covers the dish mount point."""
    return point_in_transmitter_cone(
        apex,
        boresight_fn(q),
        alpha,
        beam_length,
        dish_mount,
    )


def beam_missed_dish_fov_at_q(
    q: float,
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    dish_fov: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> float | None:
    """
    Return incident angle (rad) when the beam hits the mount but is outside
    dish FOV, else None.
    """
    if not beam_illuminates_dish_mount_at_q(
        q, apex, dish_mount, boresight_fn, alpha, beam_length
    ):
        return None
    incident = beam_incident_angle_at_q(q, dish_boresight, boresight_fn)
    if incident <= dish_fov:
        return None
    return incident


def beam_hits_dish_at_q(
    q: float,
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    dish_fov: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> bool:
    """
    True when the transmitter beam illuminates the dish mount and the
  incoming direction falls within the dish angular FOV.
    """
    beam_axis = boresight_fn(q)
    toward_source = -normalize(beam_axis)
    if not is_within_dish_fov(dish_boresight, toward_source, dish_fov):
        return False
    return point_in_transmitter_cone(apex, beam_axis, alpha, beam_length, dish_mount)


def scan_dish_hits_up_to(
    q_max: float,
    q_step: float,
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    dish_fov: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> tuple[bool, float | None]:
    """
    Scan from 0 to q_max for dish detection.

    Returns (hit, hit_at_q). hit_at_q is the first q where the beam is
    visible to the dish (transmitter cone on mount and within dish FOV).
    """
    hit_at_q: float | None = None
    q = 0.0
    while q <= q_max + 1e-12:
        if beam_hits_dish_at_q(
            q,
            apex,
            dish_mount,
            dish_boresight,
            dish_fov,
            boresight_fn,
            alpha,
            beam_length,
        ):
            hit_at_q = q
            break
        q += q_step

    return hit_at_q is not None, hit_at_q
