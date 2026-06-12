"""3D vector math helpers optimized with Numba and standard library math."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

Vec3 = NDArray[np.floating]
Mat3 = NDArray[np.floating]


def as_vec3(values: list[float] | tuple[float, float, float] | Vec3) -> Vec3:
    return np.asarray(values, dtype=np.float64).reshape(3)


@njit(cache=True)
def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def cross(a: Vec3, b: Vec3) -> Vec3:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=np.float64)


@njit(cache=True)
def norm(v: Vec3) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


@njit(cache=True)
def normalize(v: Vec3) -> Vec3:
    n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
    if n == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return np.array([v[0] / n, v[1] / n, v[2] / n], dtype=np.float64)


@njit(cache=True)
def angle_between(a: Vec3, b: Vec3) -> float:
    """Angle in radians between two directions."""
    a_u = normalize(a)
    b_u = normalize(b)
    cos_theta = a_u[0] * b_u[0] + a_u[1] * b_u[1] + a_u[2] * b_u[2]
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    return math.acos(cos_theta)


@njit(cache=True)
def slerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Spherical linear interpolation between unit directions."""
    a_u = normalize(a)
    b_u = normalize(b)
    cos_theta = a_u[0] * b_u[0] + a_u[1] * b_u[1] + a_u[2] * b_u[2]
    if cos_theta > 0.9999:
        diff = b_u - a_u
        return normalize(a_u + t * diff)
    if cos_theta < -0.9999:
        cos_theta = -0.9999
    theta = math.acos(cos_theta)
    sin_theta = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    return np.array([
        w1 * a_u[0] + w2 * b_u[0],
        w1 * a_u[1] + w2 * b_u[1],
        w1 * a_u[2] + w2 * b_u[2]
    ], dtype=np.float64)


@njit(cache=True)
def rotate_vector(v: Vec3, axis: Vec3, angle: float) -> Vec3:
    """Rotate vector v about unit axis by angle radians (Rodrigues)."""
    ax = normalize(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    c = cross(ax, v)
    d = ax[0] * v[0] + ax[1] * v[1] + ax[2] * v[2]
    factor = d * (1.0 - cos_a)
    return np.array([
        v[0] * cos_a + c[0] * sin_a + ax[0] * factor,
        v[1] * cos_a + c[1] * sin_a + ax[1] * factor,
        v[2] * cos_a + c[2] * sin_a + ax[2] * factor
    ], dtype=np.float64)


@njit(cache=True)
def rotate_toward(from_dir: Vec3, to_dir: Vec3, max_angle: float) -> Vec3:
    """Rotate from_dir toward to_dir by at most max_angle radians."""
    angle = angle_between(from_dir, to_dir)
    if angle <= max_angle:
        return normalize(to_dir)
    return slerp(from_dir, to_dir, max_angle / angle)


@njit(cache=True)
def distance(a: Vec3, b: Vec3) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


@njit(cache=True)
def spherical_to_cartesian(theta: float, phi: float) -> Vec3:
    """Unit vector from polar angle theta and azimuth phi (Desmos convention)."""
    sin_theta = math.sin(theta)
    return np.array([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        math.cos(theta)
    ], dtype=np.float64)


@njit(cache=True)
def spherical_angles_from_direction(direction: Vec3) -> tuple[float, float]:
    """Return (theta, phi) for a direction vector."""
    unit = normalize(direction)
    phi = math.atan2(unit[1], unit[0])
    theta = math.atan2((unit[0] ** 2 + unit[1] ** 2) ** 0.5, unit[2])
    return theta, phi


@njit(cache=True)
def transform_local_to_global(
    local: Vec3,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> Vec3:
    """Map local (x, y, z) components onto global basis vectors."""
    return np.array([
        local[0] * u_x[0] + local[1] * u_y[0] + local[2] * u_z[0],
        local[0] * u_x[1] + local[1] * u_y[1] + local[2] * u_z[1],
        local[0] * u_x[2] + local[1] * u_y[2] + local[2] * u_z[2]
    ], dtype=np.float64)


def linspace(start: float, stop: float, num: int) -> NDArray[np.floating]:
    return np.linspace(start, stop, num, dtype=np.float64)
