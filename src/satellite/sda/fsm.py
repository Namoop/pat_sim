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

    def reset(self) -> None:
        self.theta_offset = 0.0
        self.phi_offset = 0.0
        self.locked = False
        self.track_target = None

    def effective_receive_boresight(self, bench_boresight: Vec3) -> Vec3:
        """Receive aim with FSM steering applied on top of bench boresight."""
        if not self.locked:
            return normalize(bench_boresight)
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
        """Instantly steer FSM so receive path aligns with incoming direction."""
        self.theta_offset, self.phi_offset = _offsets_to_target(
            bench_boresight,
            toward_source,
        )
        self.track_target = toward_source.copy()
        self.locked = True

    def update_with_bench(self, bench_boresight: Vec3) -> None:
        """Recompute FSM offset after bench slew; keeps effective RX on track_target."""
        if self.track_target is not None:
            self.theta_offset, self.phi_offset = _offsets_to_target(
                bench_boresight,
                self.track_target,
            )

    @property
    def offset_magnitude(self) -> float:
        return float(np.hypot(self.theta_offset, self.phi_offset))

    def is_neutral(self, tol: float = 1e-9) -> bool:
        return self.offset_magnitude < tol
