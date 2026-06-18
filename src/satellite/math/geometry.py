"""Coordinate frames and parametric surfaces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from satellite.math.math3d import (
    Vec3,
    cross,
    distance,
    dot,
    linspace,
    normalize,
    norm,
    rotate_vector,
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


def direction_with_tangent_offset(
    base: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    offset_u: float,
    offset_v: float,
) -> Vec3:
    """
    Apply combined tangent-plane offsets to a unit direction.

    offset_u and offset_v are small angles (rad) along tangent_u and tangent_v;
    the total tilt magnitude is hypot(offset_u, offset_v).
    """
    axis = normalize(base)
    w = offset_u * tangent_u + offset_v * tangent_v
    w_mag = norm(w)
    if w_mag < 1e-15:
        return axis
    rot_axis = normalize(cross(axis, w))
    return normalize(rotate_vector(axis, rot_axis, w_mag))


def direction_with_local_offset(
    base: Vec3,
    theta_offset: float,
    phi_offset: float,
) -> Vec3:
    """Apply tangent offsets using the transmitter-style basis at base."""
    theta_0, phi_0 = spherical_angles_from_direction(base)
    u_x, u_y, u_z = transmitter_basis(theta_0, phi_0)
    return direction_with_tangent_offset(
        u_z,
        u_x,
        u_y,
        theta_offset,
        phi_offset,
    )


def axis_perpendicular_basis(axis: Vec3) -> tuple[Vec3, Vec3]:
    axis = normalize(axis)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(dot(axis, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = normalize(cross(axis, ref))
    v = cross(axis, u)
    return u, v


def dish_aperture_radius(body_radius: float, dish_fov: float) -> float:
    """Physical disc radius from full FOV (radians) relative to the body."""
    return body_radius * np.tan(dish_fov / 2.0)


def dish_disc_mesh(
    mount: Vec3,
    boresight: Vec3,
    radius: float,
    segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat circular dish aperture in the plane normal to boresight."""
    axis = normalize(boresight)
    u, v = axis_perpendicular_basis(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float64)

    verts: list[Vec3] = [mount]
    for theta in angles:
        verts.append(mount + radius * (np.cos(theta) * u + np.sin(theta) * v))

    faces: list[list[int]] = []
    for i in range(segments):
        j = i + 1
        k = 1 if i == segments - 1 else i + 2
        faces.append([0, j, k])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def local_spiral_angles(u: float, w: float, k: float) -> tuple[float, float]:
    """Desmos Theta_L(u) = w*u, Phi_L(u) = k*u in the transmitter local frame."""
    return w * u, k * u


def spiral_angles(
    u: float,
    theta_0: float,
    phi_0: float,
    w: float,
    k: float,
) -> tuple[float, float]:
    """Global spherical angles Theta(u), Phi(u) — reference only."""
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
    w: float,
    k: float,
    u_x: Vec3,
    u_y: Vec3,
    u_z: Vec3,
) -> tuple[Vec3, Vec3, Vec3]:
    """
    Desmos A_s, B_s, C_s at spiral parameter u.

    Uses local angles Theta_L = w*u, Phi_L = k*u, then maps through (U_x, U_y, U_z).
    At u=0 the boresight is U_z (believed direction toward P_2).
    """
    theta_l, phi_l = local_spiral_angles(u, w, k)
    a_l, b_l, c_l = local_spiral_frame(theta_l, phi_l)
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
    t_end: float,
    boresight_fn: Callable[[float], Vec3],
    plane_normal: Vec3,
    plane_distance: float,
    num_steps: int,
) -> np.ndarray:
    """Spiral polyline on the flat target plane from t=0 to t=t_end."""
    if t_end <= 0.0 or num_steps < 2:
        n = normalize(plane_normal)
        return np.array([p1 + plane_distance * n], dtype=np.float64)

    t_vals = linspace(0.0, t_end, num_steps)
    points: list[Vec3] = []
    for t in t_vals:
        axis = normalize(boresight_fn(t))
        points.append(
            ray_plane_intersection(p1, axis, plane_normal, plane_distance)
        )
    return np.array(points, dtype=np.float64)


def ribbon_surface_point(
    p1: Vec3,
    a_axis: Vec3,
    b_axis: Vec3,
    distance: float,
    alpha: float,
    v_width: float,
) -> Vec3:
    """R_ribbon(u,v) = P_1 + d*A_s(u) + d*tan(alpha)*v*B_s(u) at fixed u."""
    return p1 + distance * a_axis + distance * np.tan(alpha) * v_width * b_axis


def ribbon_swept_mesh(
    p1: Vec3,
    t_end: float,
    frame_fn: Callable[[float], tuple[Vec3, Vec3, Vec3]],
    alpha: float,
    distance: float,
    u_steps: int,
    v_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous 3D ribbon swept along the spiral up to t_end:

        R_ribbon(u,v) = P_1 + d*A_s(u) + d*tan(alpha)*v*B_s(u)

    u in [0, t_end] follows the search spiral; v in [-1, 1] spans beam width.
    """
    if t_end <= 0.0 or u_steps < 2 or v_steps < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    u_vals = linspace(0.0, t_end, u_steps)
    v_vals = linspace(-1.0, 1.0, v_steps)

    verts: list[Vec3] = []
    for u in u_vals:
        a_s, b_s, _ = frame_fn(u)
        for v_width in v_vals:
            verts.append(
                ribbon_surface_point(p1, a_s, b_s, distance, alpha, float(v_width))
            )

    vertices = np.array(verts, dtype=np.float64)
    faces: list[list[int]] = []

    for i in range(u_steps - 1):
        for j in range(v_steps - 1):
            a = i * v_steps + j
            b_idx = a + 1
            c_idx = a + v_steps
            d_idx = c_idx + 1
            faces.append([a, b_idx, d_idx])
            faces.append([a, d_idx, c_idx])

    return vertices, np.array(faces, dtype=np.int64)


def plane_spiral_swept_mesh(
    p1: Vec3,
    t_end: float,
    boresight_fn: Callable[[float], Vec3],
    plane_normal: Vec3,
    plane_distance: float,
    alpha: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single flat swept area on the target plane at actual-target height.

    Projects the spiral onto the plane perpendicular to the believed boresight,
    then builds one ribbon mesh (path ± beam footprint radius) growing with t.
    """
    if t_end <= 0.0 or num_steps < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    n = normalize(plane_normal)
    footprint_r = plane_distance * np.tan(alpha)
    if footprint_r <= 0.0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    path = spiral_path_on_target_plane(
        p1, t_end, boresight_fn, plane_normal, plane_distance, num_steps
    )
    if len(path) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    left_pts: list[Vec3] = []
    right_pts: list[Vec3] = []

    for i in range(len(path)):
        if i < len(path) - 1:
            tangent = path[i + 1] - path[i]
        else:
            tangent = path[i] - path[i - 1]

        tangent = tangent - dot(tangent, n) * n
        if norm(tangent) < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0]) - dot(np.array([1.0, 0.0, 0.0]), n) * n
        tangent = normalize(tangent)
        lateral = normalize(cross(n, tangent))

        left_pts.append(path[i] + footprint_r * lateral)
        right_pts.append(path[i] - footprint_r * lateral)

    left = np.array(left_pts, dtype=np.float64)
    right = np.array(right_pts, dtype=np.float64)

    verts = np.vstack([left, right])
    faces: list[list[int]] = []
    count = len(path)

    for i in range(count - 1):
        li = i
        ri = count + i
        faces.append([li, ri, ri + 1])
        faces.append([li, ri + 1, li + 1])

    # Close the start with a fan from the spiral origin on the plane
    origin_idx = len(verts)
    verts = np.vstack([verts, path[0:1]])
    faces.append([origin_idx, 0, count])
    faces.append([origin_idx, count, 1])

    return verts, np.array(faces, dtype=np.int64)


def desmos_k_surface_mesh(
    p1: Vec3,
    t_end: float,
    frame_fn: Callable[[float], tuple[Vec3, Vec3, Vec3]],
    alpha: float,
    distance: float,
    u_steps: int,
    v_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Desmos K(u,v) swept search caps from u=0 to u=t_end:

        K(u,v) = P_1 + d*A_s(u) + d*tan(alpha)*cos(v)*B_s(u) + d*tan(alpha)*sin(v)*C_s(u)

    At u=0 the first cap lies along the believed boresight at the given range.
    For visualization, pass |P_t - P_1| so caps sit at actual-target height.
    """
    if t_end <= 0.0 or u_steps < 1 or v_steps < 3:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    u_vals = linspace(0.0, t_end, max(2, u_steps))
    v_vals = np.linspace(0.0, 2.0 * np.pi, v_steps, endpoint=False, dtype=np.float64)

    verts: list[Vec3] = []
    for u in u_vals:
        a_s, b_s, c_s = frame_fn(u)
        for v_angle in v_vals:
            verts.append(
                cone_surface_point(p1, a_s, b_s, c_s, alpha, distance, v_angle)
            )

    vertices = np.array(verts, dtype=np.float64)
    u_count = len(u_vals)
    faces: list[list[int]] = []

    for i in range(u_count - 1):
        for j in range(v_steps):
            j_next = (j + 1) % v_steps
            a = i * v_steps + j
            b_idx = i * v_steps + j_next
            c_idx = (i + 1) * v_steps + j
            d_idx = (i + 1) * v_steps + j_next
            faces.append([a, b_idx, d_idx])
            faces.append([a, d_idx, c_idx])

    return vertices, np.array(faces, dtype=np.int64)


def actual_position_from_jumble(
    p1: Vec3,
    believed_boresight: Vec3,
    link_range: float,
    theta_jumble: float,
    phi_jumble: float,
) -> Vec3:
    """
    Compute actual receiver position by applying launch jumble offsets to the
    believed direction at link_range.
    """
    theta_0, phi_0 = spherical_angles_from_direction(believed_boresight)
    theta_actual = theta_0 + theta_jumble
    phi_actual = phi_0 + phi_jumble
    direction = spherical_to_cartesian(theta_actual, phi_actual)
    return p1 + link_range * direction


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
