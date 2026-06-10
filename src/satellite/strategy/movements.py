"""Bench aim movement patterns (dish == beam)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from satellite.math3d import Vec3
from satellite.strategy.patterns import (
    basis_at_direction,
    circle_aim_at,
    grid_aim_at,
    line_aim_at,
    spiral_aim_at,
)


@dataclass
class AimContext:
    """Per-satellite per-epoch aim state (set at epoch boundary)."""

    center: Vec3
    epoch_start_aim: Vec3
    u_x: Vec3
    u_y: Vec3
    u_z: Vec3


class MovementPattern(ABC):
    @abstractmethod
    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        """Return inertial unit aim direction at local_t within epoch."""


@dataclass(frozen=True)
class Hold(MovementPattern):
    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return ctx.epoch_start_aim


@dataclass(frozen=True)
class Reset(MovementPattern):
    """Reset bench to initial offset at epoch start, then hold."""

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return ctx.center


@dataclass(frozen=True)
class Spiral(MovementPattern):
    w: float
    k: float
    max_radius: float
    speed: float = 1.0

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return spiral_aim_at(
            local_t,
            w=self.w,
            k=self.k,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
            max_radius=self.max_radius,
            speed=self.speed,
        )


@dataclass(frozen=True)
class Circle(MovementPattern):
    radius: float
    period: float | None = None

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        period = self.period if self.period is not None else duration
        return circle_aim_at(
            local_t,
            duration=period,
            radius=self.radius,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
        )


@dataclass(frozen=True)
class Line(MovementPattern):
    extent: float
    axis_angle: float = 0.0

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return line_aim_at(
            local_t,
            duration=duration,
            extent=self.extent,
            axis_angle=self.axis_angle,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
        )


@dataclass(frozen=True)
class Grid(MovementPattern):
    spacing: float
    extent: float

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return grid_aim_at(
            local_t,
            duration=duration,
            spacing=self.spacing,
            extent=self.extent,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
        )


def build_aim_context(satellite, *, reset: bool = False) -> AimContext:
    from satellite.sda.satellite import Satellite

    sat: Satellite = satellite
    if reset:
        sat.receiver.reset_dish_tracking()
    center = sat.bench.initial_boresight.copy()
    u_x, u_y, u_z = basis_at_direction(center)
    return AimContext(
        center=center,
        epoch_start_aim=sat.bench.bench_boresight.copy(),
        u_x=u_x,
        u_y=u_y,
        u_z=u_z,
    )
