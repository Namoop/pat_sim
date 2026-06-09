"""Receiver-side SDA — dish on optical bench with FSM fine steering."""

from __future__ import annotations

import numpy as np

from satellite.geometry import (
    actual_target_plane_distance,
    cone_surface_mesh,
    dish_aperture_radius,
    dish_disc_mesh,
    receiver_global_frame,
    receiver_basis,
)
from satellite.math3d import Vec3
from satellite.sda.bench import BenchGeometry, OpticalBench
from satellite.sda.fsm import FastSteeringMirror


class ReceiverSDA:
    """Receiver dish on optical bench with fast steering mirror."""

    def __init__(
        self,
        bench: OpticalBench,
        fsm: FastSteeringMirror,
        gamma: float,
        beta: float,
        omega_r: float,
        l_r: float,
        body_radius: float,
        dish_fov: float,
    ) -> None:
        self.bench = bench
        self.fsm = fsm
        self.gamma = gamma
        self.beta = beta
        self.omega_r = omega_r
        self.l_r = l_r
        self.body_radius = body_radius
        self.dish_fov = dish_fov

        self.d_circ = actual_target_plane_distance(
            bench.partner_position, bench.position, l_r
        )

    # Backward-compat aliases
    @property
    def position(self) -> Vec3:
        return self.bench.position

    @property
    def partner_position(self) -> Vec3:
        return self.bench.partner_position

    @property
    def pt(self) -> Vec3:
        return self.bench.position

    @property
    def p1(self) -> Vec3:
        return self.bench.partner_position

    @property
    def bench_slew_time(self) -> float:
        return self.bench.bench_slew_time

    @property
    def nominal_boresight(self) -> Vec3:
        """True aim direction toward the partner."""
        return self.bench.toward_partner

    @property
    def dish_boresight(self) -> Vec3:
        return self.bench.geometry_snapshot(self.body_radius).dish_boresight

    @property
    def has_seen_beam(self) -> bool:
        return self.bench.acquisition.has_seen_beam

    @property
    def initial_pointing_offset(self) -> float:
        return self.bench.initial_pointing_offset

    @property
    def configured_offset_magnitude(self) -> float:
        return self.bench.configured_bench_offset_magnitude

    @property
    def initial_dish_mount(self) -> Vec3:
        return self.bench.dish_mount_for_boresight(
            self.bench.initial_dish_boresight,
            self.body_radius,
        )

    @property
    def initial_dish_boresight(self) -> Vec3:
        return self.bench.initial_dish_boresight

    def reset_dish_tracking(self) -> None:
        self.bench.reset_tracking()
        self.fsm.reset()

    @property
    def dish_mount(self) -> Vec3:
        return self.bench.geometry_snapshot(self.body_radius).mount

    def dish_mount_for_boresight(self, boresight: Vec3) -> Vec3:
        return self.bench.dish_mount_for_boresight(boresight, self.body_radius)

    def geometry_snapshot(self) -> BenchGeometry:
        return self.bench.geometry_snapshot(self.body_radius)

    def dish_range_from_partner(self) -> float:
        from satellite.math3d import distance

        return distance(self.partner_position, self.dish_mount)

    def dish_range_from_p1(self) -> float:
        return self.dish_range_from_partner()

    def dish_aperture_radius(self) -> float:
        return dish_aperture_radius(self.body_radius, self.dish_fov)

    def observe_beam(
        self,
        in_cone: bool,
        beam_direction: Vec3,
        dq: float,
        *,
        dish_at_step_start: Vec3 | None = None,
    ) -> Vec3:
        return self.bench.observe_beam(
            self.fsm,
            in_cone,
            beam_direction,
            dq,
            dish_at_step_start=dish_at_step_start,
        )

    def dish_mesh_at(
        self,
        segments: int = 32,
        *,
        boresight: Vec3 | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        aim = boresight if boresight is not None else self.dish_boresight
        mount = self.dish_mount_for_boresight(aim)
        return dish_disc_mesh(
            mount,
            aim,
            self.dish_aperture_radius(),
            segments,
        )

    def frame_at(self, t: float) -> tuple[Vec3, Vec3, Vec3]:
        return receiver_global_frame(
            self.position, self.partner_position, self.beta, self.omega_r, t
        )

    def receiver_cone_mesh_at(
        self,
        q: float,
        u_steps: int,
        v_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_rec, b_rec, c_rec = self.frame_at(q)
        return cone_surface_mesh(
            self.position,
            a_rec,
            b_rec,
            c_rec,
            self.gamma,
            self.l_r,
            u_steps,
            v_steps,
        )

    def receiver_cap_mesh_at(
        self,
        q: float,
        u_steps: int,
        v_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_rec, b_rec, c_rec = self.frame_at(q)
        return cone_surface_mesh(
            self.position,
            a_rec,
            b_rec,
            c_rec,
            self.gamma,
            self.l_r,
            u_steps,
            v_steps,
        )

    def is_target_in_aperture(self, _direction: Vec3, _t: float) -> bool:
        raise NotImplementedError("Receiver aperture detection not yet implemented")

    def basis(self) -> tuple[Vec3, Vec3, Vec3]:
        return receiver_basis(self.position, self.partner_position)
