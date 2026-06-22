"""3D visualization squish: compress link-axis distance, scale angles to match."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from satellite.math.math3d import Vec3, normalize


def link_axis(p1: Vec3, p2: Vec3) -> np.ndarray:
    sep = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    sep_len = float(np.linalg.norm(sep))
    if sep_len < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return sep / sep_len


def squish_position(
    point: Vec3,
    origin: Vec3,
    axis: np.ndarray,
    squish: float,
) -> np.ndarray:
    if squish == 1.0:
        return np.asarray(point, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    rel = p - o
    t = float(np.dot(rel, a))
    return p + a * t * (1.0 / squish - 1.0)


def squish_direction(
    direction: Vec3,
    ref_axis: Vec3,
    squish: float,
) -> np.ndarray:
    """Scale angular deviation from ref_axis (unit vector toward partner)."""
    if squish == 1.0:
        return normalize(np.asarray(direction, dtype=np.float64))
    d = normalize(np.asarray(direction, dtype=np.float64))
    r = normalize(np.asarray(ref_axis, dtype=np.float64))
    parallel = float(np.dot(d, r))
    perp = d - parallel * r
    perp_len = float(np.linalg.norm(perp))
    if perp_len < 1e-12:
        return d.copy()
    theta = float(np.arctan2(perp_len, parallel))
    theta_new = squish * theta
    perp_unit = perp / perp_len
    return normalize(np.cos(theta_new) * r + np.sin(theta_new) * perp_unit)


def squish_length(length: float, squish: float) -> float:
    if squish == 1.0:
        return length
    return length / squish


def squish_angle(angle: float, squish: float) -> float:
    if squish == 1.0:
        return angle
    return angle * squish


@dataclass(frozen=True)
class SquishContext:
    squish: float
    origin: np.ndarray
    axis: np.ndarray

    @classmethod
    def from_satellites(cls, p1: Vec3, p2: Vec3, squish: float) -> SquishContext:
        return cls(
            squish=float(squish),
            origin=np.asarray(p1, dtype=np.float64),
            axis=link_axis(p1, p2),
        )

    def position(self, point: Vec3) -> np.ndarray:
        return squish_position(point, self.origin, self.axis, self.squish)

    def direction(self, direction: Vec3, *, toward_partner: Vec3) -> np.ndarray:
        return squish_direction(direction, toward_partner, self.squish)

    def length(self, length: float) -> float:
        return squish_length(length, self.squish)

    def angle(self, angle: float) -> float:
        return squish_angle(angle, self.squish)

    def positions(self, points: np.ndarray) -> np.ndarray:
        if self.squish == 1.0 or len(points) == 0:
            return np.asarray(points, dtype=np.float64)
        out = np.asarray(points, dtype=np.float64).copy()
        for i in range(out.shape[0]):
            out[i] = self.position(out[i])
        return out
