"""3D vector math helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.floating]
Mat3 = NDArray[np.floating]


def as_vec3(values: list[float] | tuple[float, float, float] | Vec3) -> Vec3:
    return np.asarray(values, dtype=np.float64).reshape(3)


def dot(a: Vec3, b: Vec3) -> float:
    return float(np.dot(a, b))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return np.cross(a, b)


def norm(v: Vec3) -> float:
    return float(np.linalg.norm(v))


def normalize(v: Vec3) -> Vec3:
    n = norm(v)
    if n == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return v / n


def angle_between(a: Vec3, b: Vec3) -> float:
    """Angle in radians between two directions."""
    return float(np.arccos(np.clip(dot(normalize(a), normalize(b)), -1.0, 1.0)))


def slerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Spherical linear interpolation between unit directions."""
    a_u = normalize(a)
    b_u = normalize(b)
    cos_theta = dot(a_u, b_u)
    if cos_theta > 0.9999:
        return normalize(a_u + t * (b_u - a_u))
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return w1 * a_u + w2 * b_u


def rotate_vector(v: Vec3, axis: Vec3, angle: float) -> Vec3:
    """Rotate vector v about unit axis by angle radians (Rodrigues)."""
    axis = normalize(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return v * cos_a + cross(axis, v) * sin_a + axis * dot(axis, v) * (1.0 - cos_a)


def rotate_toward(from_dir: Vec3, to_dir: Vec3, max_angle: float) -> Vec3:
    """Rotate from_dir toward to_dir by at most max_angle radians."""
    angle = angle_between(from_dir, to_dir)
    if angle <= max_angle:
        return normalize(to_dir)
    return slerp(from_dir, to_dir, max_angle / angle)


def distance(a: Vec3, b: Vec3) -> float:
    return norm(b - a)


def spherical_to_cartesian(theta: float, phi: float) -> Vec3:
    """Unit vector from polar angle theta and azimuth phi (Desmos convention)."""
    sin_theta = np.sin(theta)
    return np.array(
        [
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


def spherical_angles_from_direction(direction: Vec3) -> tuple[float, float]:
    """Return (theta, phi) for a direction vector."""
    unit = normalize(direction)
    phi = float(np.arctan2(unit[1], unit[0]))
    theta = float(np.arctan2(np.sqrt(unit[0] ** 2 + unit[1] ** 2), unit[2]))
    return theta, phi


def transform_local_to_global(
    local: Vec3,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    """Map local (x, y, z) components onto global basis vectors."""
    return local[0] * u_x + local[1] * u_y + local[2] * u_z


def linspace(start: float, stop: float, num: int) -> NDArray[np.floating]:
    return np.linspace(start, stop, num, dtype=np.float64)
