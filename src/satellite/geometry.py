"""Coordinate frames and parametric surfaces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.math3d import (
    Vec3,
    cross,
    distance,
    dot,
    linspace,
    normalize,
    norm,
    spherical_angles_from_direction,
    spherical_to_cartesian,
    transform_local_to_global,
)


def believed_direction(p1: Vec3, p2: Vec3) -> Vec3:
    return normalize(p2 - p1)


def believed_distance(p1: Vec3, p2: Vec3) -> float:
    return distance(p1, p2)


def actual_target_direction(p1: Vec3, pt: Vec3) -> Vec3:
    return normalize(pt - p1)


def actual_target_plane_distance(p1: Vec3, pt: Vec3, l_r: float) -> float:
    return distance(p1, pt) - l_r


def transmitter_basis(theta_0: float, phi_0: float) -> tuple[Vec3, Vec3, Vec3]:
    """Desmos U_x, U_y, U_z from nominal spherical angles."""
    u_z = spherical_to_cartesian(theta_0, phi_0)
    u_x = np.array(
        [
            np.cos(theta_0) * np.cos(phi_0),
            np.cos(theta_0) * np.sin(phi_0),
            -np.sin(theta_0),
        ],
        dtype=np.float64,
    )
    u_y = np.array(
        [-np.sin(phi_0), np.cos(phi_0), 0.0],
        dtype=np.float64,
    )
    return u_x, u_y, u_z


def receiver_basis(pt: Vec3, p1: Vec3, u_up: Vec3 | None = None) -> tuple[Vec3, Vec3, Vec3]:
    """Desmos U_look, U_T, U_B at actual receiver position."""
    if u_up is None:
        u_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    u_look = normalize(p1 - pt)
    u_t = normalize(cross(u_look, u_up))
    u_b = cross(u_look, u_t)
    return u_look, u_t, u_b


def spiral_angles(
    u: float,
    theta_0: float,
    phi_0: float,
    w: float,
    k: float,
) -> tuple[float, float]:
    return theta_0 + w * u, phi_0 + k * u


def local_spiral_frame(theta: float, phi: float) -> tuple[Vec3, Vec3, Vec3]:
    """Desmos A_L, B_L, C_L from spherical angles."""
    a_l = spherical_to_cartesian(theta, phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    b_l = np.array(
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
        dtype=np.float64,
    )
    c_l = np.array([-sin_phi, cos_phi, 0.0], dtype=np.float64)
    return a_l, b_l, c_l


def global_spiral_frame(
    u: float,
    theta_0: float,
    phi_0: float,
    w: float,
    k: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> tuple[Vec3, Vec3, Vec3]:
    """Desmos A_s, B_s, C_s at parameter u."""
    theta, phi = spiral_angles(u, theta_0, phi_0, w, k)
    a_l, b_l, c_l = local_spiral_frame(theta, phi)
    return (
        transform_local_to_global(a_l, u_x, u_y, u_z),
        transform_local_to_global(b_l, u_x, u_y, u_z),
        transform_local_to_global(c_l, u_x, u_y, u_z),
    )


def cone_surface_point(
    apex: Vec3,
    a_axis: Vec3,
    b_axis: Vec3,
    c_axis: Vec3,
    alpha: float,
    u_dist: float,
    v_angle: float,
) -> Vec3:
    """Single point on Desmos S(u, v) cone surface."""
    spread = u_dist * np.tan(alpha)
    return (
        apex
        + u_dist * a_axis
        + spread * np.cos(v_angle) * b_axis
        + spread * np.sin(v_angle) * c_axis
    )


def cone_surface_mesh(
    apex: Vec3,
    a_axis: Vec3,
    b_axis: Vec3,
    c_axis: Vec3,
    alpha: float,
    length: float,
    u_steps: int,
    v_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for a cone mesh with explicit apex at apex."""
    if u_steps < 2:
        raise ValueError("u_steps must be >= 2")

    v_vals = np.linspace(0.0, 2.0 * np.pi, v_steps, endpoint=False, dtype=np.float64)

    verts: list[Vec3] = [apex]
    for ring in range(1, u_steps):
        u_dist = length * ring / (u_steps - 1)
        for v_angle in v_vals:
            verts.append(
                cone_surface_point(apex, a_axis, b_axis, c_axis, alpha, u_dist, v_angle)
            )

    vertices = np.array(verts, dtype=np.float64)
    faces: list[list[int]] = []

    # Fan from apex to first ring
    for j in range(v_steps):
        j_next = (j + 1) % v_steps
        faces.append([0, 1 + j, 1 + j_next])

    # Quads between successive rings
    for ring in range(1, u_steps - 1):
        base = 1 + (ring - 1) * v_steps
        next_base = 1 + ring * v_steps
        for j in range(v_steps):
            j_next = (j + 1) % v_steps
            faces.append([base + j, base + j_next, next_base + j_next])
            faces.append([base + j, next_base + j_next, next_base + j])

    return vertices, np.array(faces, dtype=np.int64)


def ray_plane_intersection(
    p1: Vec3,
    boresight: Vec3,
    plane_normal: Vec3,
    plane_distance: float,
) -> Vec3:
    """
    Intersect ray p1 + t*boresight with the plane perpendicular to plane_normal
    passing through p1 + plane_distance * plane_normal.
    """
    n = normalize(plane_normal)
    denom = dot(boresight, n)
    if abs(denom) < 1e-12:
        return p1 + plane_distance * n
    t = plane_distance / denom
    return p1 + t * boresight


def spiral_path_on_target_plane(
    p1: Vec3,
    q_end: float,
    boresight_fn: Callable[[float], Vec3],
    plane_normal: Vec3,
    plane_distance: float,
    num_steps: int,
) -> np.ndarray:
    """Spiral polyline on the flat target plane from q=0 to q=q_end."""
    if q_end <= 0.0 or num_steps < 2:
        n = normalize(plane_normal)
        return np.array([p1 + plane_distance * n], dtype=np.float64)

    q_vals = linspace(0.0, q_end, num_steps)
    points: list[Vec3] = []
    for q in q_vals:
        axis = normalize(boresight_fn(q))
        points.append(
            ray_plane_intersection(p1, axis, plane_normal, plane_distance)
        )
    return np.array(points, dtype=np.float64)


def spiral_swept_area_mesh(
    p1: Vec3,
    q_end: float,
    boresight_fn: Callable[[float], Vec3],
    plane_normal: Vec3,
    plane_distance: float,
    alpha: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filled orange swept area on the target plane, growing with q_end.

    Triangulates from the nominal center (believed aim point on the plane) out
    to the spiral path, plus small footprint disks along the path.
    """
    n = normalize(plane_normal)
    center = p1 + plane_distance * n
    path = spiral_path_on_target_plane(
        p1, q_end, boresight_fn, plane_normal, plane_distance, num_steps
    )

    if q_end <= 0.0 or len(path) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    footprint_r = plane_distance * np.tan(alpha)
    disk_steps = 12

    verts: list[Vec3] = [center, *path]
    faces: list[list[int]] = []

    # Fan from nominal center along the spiral (filled region grows with q)
    for i in range(len(path) - 1):
        faces.append([0, i + 1, i + 2])

    # Beam footprint disks along the path for visible width (Desmos orange disk)
    sample_indices = list(range(0, len(path), max(1, len(path) // 30)))
    if sample_indices[-1] != len(path) - 1:
        sample_indices.append(len(path) - 1)

    for idx in sample_indices:
        point = path[idx]
        if idx < len(path) - 1:
            tangent = path[idx + 1] - path[idx]
        elif idx > 0:
            tangent = path[idx] - path[idx - 1]
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        tangent = tangent - dot(tangent, n) * n
        if norm(tangent) < 1e-12:
            fallback = np.array([1.0, 0.0, 0.0]) - dot(np.array([1.0, 0.0, 0.0]), n) * n
            tangent = fallback if norm(fallback) > 1e-12 else np.array([0.0, 1.0, 0.0])
        tangent = normalize(tangent)
        bitangent = normalize(cross(n, tangent))

        disk_center_idx = len(verts)
        verts.append(point)
        for k in range(disk_steps):
            angle = 2.0 * np.pi * k / disk_steps
            verts.append(
                point
                + footprint_r * np.cos(angle) * tangent
                + footprint_r * np.sin(angle) * bitangent
            )
        for k in range(disk_steps):
            k_next = (k + 1) % disk_steps
            faces.append([disk_center_idx, disk_center_idx + 1 + k, disk_center_idx + 1 + k_next])

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def spiral_trail_on_plane(
    p1: Vec3,
    q_max: float,
    boresight_fn: Callable[[float], Vec3],
    plane_distance: float,
    num_steps: int,
    plane_normal: Vec3 | None = None,
) -> np.ndarray:
    """Backward-compatible spiral polyline on the target plane."""
    if plane_normal is None:
        plane_normal = normalize(boresight_fn(0.0))
    return spiral_path_on_target_plane(
        p1,
        q_max,
        boresight_fn,
        plane_normal,
        plane_distance,
        num_steps,
    )


def actual_position_from_jumble(
    p1: Vec3,
    p2: Vec3,
    theta_jumble: float,
    phi_jumble: float,
) -> Vec3:
    """
    Compute actual receiver position by applying launch jumble offsets to the
    believed direction at believed range.
    """
    d = believed_distance(p1, p2)
    theta_0, phi_0 = spherical_angles_from_direction(believed_direction(p1, p2))
    theta_actual = theta_0 + theta_jumble
    phi_actual = phi_0 + phi_jumble
    direction = spherical_to_cartesian(theta_actual, phi_actual)
    return p1 + d * direction


def wobble_normal(
    u_look: Vec3,
    u_t: Vec3,
    u_b: Vec3,
    beta: float,
    omega_r: float,
    t: float,
) -> Vec3:
    """Desmos N_wobble — boresight deviated from look axis by beta, rotating."""
    angle = omega_r * t
    return (
        np.cos(beta) * u_look
        + np.sin(beta) * np.cos(angle) * u_t
        + np.sin(beta) * np.sin(angle) * u_b
    )


def receiver_local_frame(beta: float, omega_r: float, t: float) -> tuple[Vec3, Vec3, Vec3]:
    """Desmos A_rLocal, B_rLocal, C_rLocal."""
    angle = omega_r * t
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)

    a_r = np.array([sin_b * cos_a, sin_b * sin_a, cos_b], dtype=np.float64)
    b_r = np.array([cos_b * cos_a, cos_b * sin_a, -sin_b], dtype=np.float64)
    c_r = np.array([-sin_a, cos_a, 0.0], dtype=np.float64)
    return a_r, b_r, c_r


def receiver_global_frame(
    pt: Vec3,
    p1: Vec3,
    beta: float,
    omega_r: float,
    t: float,
) -> tuple[Vec3, Vec3, Vec3]:
    """Desmos A_rec, B_rec, C_rec."""
    u_look, u_t, u_b = receiver_basis(pt, p1)
    a_l, b_l, c_l = receiver_local_frame(beta, omega_r, t)
    return (
        transform_local_to_global(a_l, u_t, u_b, u_look),
        transform_local_to_global(b_l, u_t, u_b, u_look),
        transform_local_to_global(c_l, u_t, u_b, u_look),
    )
