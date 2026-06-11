"""Receiver-side SDA — dish on optical bench with FSM fine steering."""

from __future__ import annotations

import numpy as np

from satellite.geometry import dish_aperture_radius, dish_disc_mesh
from satellite.math3d import Vec3
from satellite.sda.bench import BenchGeometry, OpticalBench
from satellite.sda.fsm import FastSteeringMirror


class ReceiverSDA:
    """Receiver dish on optical bench with fast steering mirror."""

    def __init__(
        self,
        bench: OpticalBench,
        fsm: FastSteeringMirror,
        *,
        body_radius: float,
        dish_fov: float,
    ) -> None:
        self.bench = bench
        self.fsm = fsm
        self.body_radius = body_radius
        self.dish_fov = dish_fov

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

    def dish_aperture_radius(self) -> float:
        return dish_aperture_radius(self.body_radius, self.dish_fov)

    def observe_beam(
        self,
        in_cone: bool,
        source_position: Vec3,
        dq: float,
        *,
        dish_at_step_start: Vec3 | None = None,
    ) -> Vec3:
        dish = (
            dish_at_step_start
            if dish_at_step_start is not None
            else self.dish_boresight
        )
        mount = self.dish_mount_for_boresight(dish)
        return self.bench.observe_beam(
            self.fsm,
            in_cone,
            source_position,
            dq,
            dish_mount=mount,
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
