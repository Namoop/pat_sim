"""Hit detection helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.math3d import Vec3, angle_between, dot, norm, normalize

from satellite import detection_fast as _detection_fast

_beam_hits_dish_fast = _detection_fast.beam_hits_dish_optimized


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


def incoming_from_source(apex: Vec3, dish_mount: Vec3) -> Vec3:
    """Unit direction from dish mount toward the transmitter apex."""
    return normalize(apex - dish_mount)


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


def beam_incident_angle_at_t(
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
) -> float:
    """Angle between dish boresight and incoming radiation from apex."""
    toward_source = incoming_from_source(apex, dish_mount)
    return angle_between(dish_boresight, toward_source)


def beam_illuminates_dish_mount_at_t(
    t: float,
    apex: Vec3,
    dish_mount: Vec3,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> bool:
    """True when the transmitter cone covers the dish mount point."""
    return point_in_transmitter_cone(
        apex,
        boresight_fn(t),
        alpha,
        beam_length,
        dish_mount,
    )


def beam_missed_dish_fov_at_t(
    t: float,
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
    if not beam_illuminates_dish_mount_at_t(
        t, apex, dish_mount, boresight_fn, alpha, beam_length
    ):
        return None
    incident = beam_incident_angle_at_t(apex, dish_mount, dish_boresight)
    if incident <= dish_fov:
        return None
    return incident


def beam_hits_dish(
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    dish_fov: float,
    beam_axis: Vec3,
    alpha: float,
    beam_length: float,
) -> bool:
    """
    True when the transmitter beam illuminates the dish mount and the
    incoming direction falls within the dish angular FOV.
    """
    if _beam_hits_dish_fast is not None:
        return _beam_hits_dish_fast(
            apex,
            dish_mount,
            dish_boresight,
            dish_fov,
            beam_axis,
            alpha,
            beam_length,
        )
    toward_source = incoming_from_source(apex, dish_mount)
    if not is_within_dish_fov(dish_boresight, toward_source, dish_fov):
        return False
    return point_in_transmitter_cone(apex, beam_axis, alpha, beam_length, dish_mount)


def beam_hits_dish_at_t(
    t: float,
    apex: Vec3,
    dish_mount: Vec3,
    dish_boresight: Vec3,
    dish_fov: float,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
    beam_length: float,
) -> bool:
    """Like beam_hits_dish but evaluates boresight_fn at t."""
    return beam_hits_dish(
        apex,
        dish_mount,
        dish_boresight,
        dish_fov,
        boresight_fn(t),
        alpha,
        beam_length,
    )
