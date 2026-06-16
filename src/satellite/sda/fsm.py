"""Fast steering mirror — fine receive pointing on the optical bench."""

from __future__ import annotations

import numpy as np

from satellite.geometry import direction_with_tangent_offset, transmitter_basis
from satellite.math3d import Vec3, angle_between, cross, dot, norm, normalize, spherical_angles_from_direction


def _offsets_to_target(bench_boresight: Vec3, target: Vec3) -> tuple[float, float]:
    """
    Exact tangent-plane offsets so direction_with_tangent_offset reaches target.

    direction_with_tangent_offset rotates base by |w| around axis cross(base, w)
    where w = theta*u_x + phi*u_y. For rotation by tilt toward target, set
    cross(base, w) = tilt * n with n = normalize(cross(base, target)), hence
    w = tilt * cross(n, base).
    """
    base = normalize(bench_boresight)
    aim = normalize(target)
    tilt = angle_between(base, aim)
    if tilt < 1e-15:
        return 0.0, 0.0
    axis_perp = cross(base, aim)
    if norm(axis_perp) < 1e-15:
        return 0.0, 0.0
    n = normalize(axis_perp)
    w = tilt * cross(n, base)
    u_x, u_y, _ = transmitter_basis(*spherical_angles_from_direction(base))
    return float(dot(w, u_x)), float(dot(w, u_y))


class FastSteeringMirror:
    """FSM steering offset in the bench tangent frame."""

    def __init__(self) -> None:
        self.theta_offset = 0.0
        self.phi_offset = 0.0
        self.locked = False
        self.track_target: Vec3 | None = None
        self._target_theta = 0.0
        self._target_phi = 0.0

    def reset(self) -> None:
        self.theta_offset = 0.0
        self.phi_offset = 0.0
        self.locked = False
        self.track_target = None
        self._target_theta = 0.0
        self._target_phi = 0.0

    def effective_receive_boresight(self, bench_boresight: Vec3) -> Vec3:
        """Receive/transmit aim with FSM steering applied on top of bench boresight."""
        u_x, u_y, _ = transmitter_basis(
            *spherical_angles_from_direction(bench_boresight)
        )
        return direction_with_tangent_offset(
            bench_boresight,
            u_x,
            u_y,
            self.theta_offset,
            self.phi_offset,
        )

    def snap_to(self, bench_boresight: Vec3, toward_source: Vec3) -> None:
        """Set FSM target so receive path aligns with incoming direction."""
        self._target_theta, self._target_phi = _offsets_to_target(
            bench_boresight,
            toward_source,
        )
        self.track_target = toward_source.copy()
        # Mirror will reach target in update()

    def update(self, bench_boresight: Vec3, dq: float, max_fsm_speed: float, max_fsm_radius: float) -> None:
        """Slew mirror toward target at max_fsm_speed, clamped to max_fsm_radius."""
        if self.track_target is None:
            return

        # Always recompute target offsets as bench moves
        self._target_theta, self._target_phi = _offsets_to_target(
            bench_boresight,
            self.track_target,
        )

        # Clamp target offsets to the physical range of the FSM
        target_dist = np.hypot(self._target_theta, self._target_phi)
        clamped_target_theta = self._target_theta
        clamped_target_phi = self._target_phi
        if target_dist > max_fsm_radius:
            clamped_target_theta = (self._target_theta / target_dist) * max_fsm_radius
            clamped_target_phi = (self._target_phi / target_dist) * max_fsm_radius

        if max_fsm_speed <= 0:
            # Infinite speed fallback
            self.theta_offset = clamped_target_theta
            self.phi_offset = clamped_target_phi
            self.locked = (target_dist <= max_fsm_radius + 1e-12)
            return

        # Simple 2D slew in tangent plane (approximation of mirror DOF)
        du = clamped_target_theta - self.theta_offset
        dv = clamped_target_phi - self.phi_offset
        dist = np.hypot(du, dv)
        max_step = max_fsm_speed * dq

        if dist <= max_step + 1e-12:
            self.theta_offset = clamped_target_theta
            self.phi_offset = clamped_target_phi
            # Locked means we are pointing exactly at the actual target
            self.locked = (target_dist <= max_fsm_radius + 1e-12)
        else:
            self.theta_offset += (du / dist) * max_step
            self.phi_offset += (dv / dist) * max_step
            self.locked = False


    def update_with_bench(self, bench_boresight: Vec3) -> None:
        """Recompute FSM target after bench slew; mirror remains on track."""
        if self.track_target is not None:
            self._target_theta, self._target_phi = _offsets_to_target(
                bench_boresight,
                self.track_target,
            )

    @property
    def offset_magnitude(self) -> float:
        return float(np.hypot(self.theta_offset, self.phi_offset))

    def is_neutral(self, tol: float = 1e-9) -> bool:
        return self.offset_magnitude < tol
