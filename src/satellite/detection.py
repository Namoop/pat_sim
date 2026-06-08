"""Hit detection helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.math3d import Vec3, dot


def alignment_dot(target_direction: Vec3, boresight: Vec3) -> float:
    """Desmos f_dot — cosine of angle between target and boresight."""
    return dot(target_direction, boresight)


def is_in_cone(alignment: float, alpha: float) -> bool:
    """True when target falls within cone half-angle alpha."""
    return alignment >= np.cos(alpha)


def in_cone_at_q(
    q: float,
    target_direction: Vec3,
    boresight_fn: Callable[[float], Vec3],
    alpha: float,
) -> bool:
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
    Desmos S_canned — scan from 0 to q_max.

    Returns (hit, hit_at_q). hit_at_q is the first q where target enters cone.
    """
    hit_at_q: float | None = None
    q = 0.0
    while q <= q_max + 1e-12:
        if in_cone_at_q(q, target_direction, boresight_fn, alpha):
            hit_at_q = q
            break
        q += q_step

    return hit_at_q is not None, hit_at_q
