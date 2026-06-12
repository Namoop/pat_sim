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
    lissajous_aim_at,
    raster_aim_at,
    rosette_aim_at,
    spiral_aim_at,
)


@dataclass
class AimContext:
    """Per-satellite per-step aim state (set at movement step boundary)."""

    center: Vec3
    step_start_aim: Vec3
    u_x: Vec3
    u_y: Vec3
    u_z: Vec3


class MovementPattern(ABC):
    @abstractmethod
    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        """Return inertial unit aim direction at local_t within the step."""


@dataclass(frozen=True)
class Hold(MovementPattern):
    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return ctx.step_start_aim


@dataclass(frozen=True)
class Reset(MovementPattern):
    """Slew bench from current offset back to center boresight."""

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        if duration <= 0:
            return ctx.center
        progress = min(local_t / duration, 1.0)
        from satellite.math3d import slerp
        return slerp(ctx.step_start_aim, ctx.center, progress)


@dataclass(frozen=True)
class Spiral(MovementPattern):
    w: float
    k: float
    max_radius: float
    speed: float = 1.0

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        if self.max_radius == 0.0:
            from satellite.math3d import angle_between, normalize
            import math
            R_start = angle_between(ctx.step_start_aim, ctx.center)
            if R_start <= 0.0 or duration <= 0.0:
                return ctx.center
            progress = min(local_t / duration, 1.0)
            theta_l = R_start * (1.0 - progress)
            u = (R_start / self.w) * (1.0 - progress) if self.w > 0.0 else 0.0
            phi_l = self.k * u
            
            sin_theta = math.sin(theta_l)
            cos_theta = math.cos(theta_l)
            sin_phi = math.sin(phi_l)
            cos_phi = math.cos(phi_l)
            
            al0 = sin_theta * cos_phi
            al1 = sin_theta * sin_phi
            al2 = cos_theta
            
            asx = al0 * ctx.u_x[0] + al1 * ctx.u_y[0] + al2 * ctx.u_z[0]
            asy = al0 * ctx.u_x[1] + al1 * ctx.u_y[1] + al2 * ctx.u_z[1]
            asz = al0 * ctx.u_x[2] + al1 * ctx.u_y[2] + al2 * ctx.u_z[2]
            import numpy as np
            return normalize(np.array([asx, asy, asz], dtype=np.float64))

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


@dataclass(frozen=True)
class Rosette(MovementPattern):
    A: float
    w1: float
    w2: float

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return rosette_aim_at(
            local_t,
            A=self.A,
            w1=self.w1,
            w2=self.w2,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
        )


@dataclass(frozen=True)
class Lissajous(MovementPattern):
    A: float
    wx: float
    wy: float
    delta: float

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return lissajous_aim_at(
            local_t,
            A=self.A,
            wx=self.wx,
            wy=self.wy,
            delta=self.delta,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
        )


@dataclass(frozen=True)
class SerpentineRaster(MovementPattern):
    radius: float
    steps: int
    horizontal: bool = True

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        return raster_aim_at(
            local_t,
            duration=duration,
            radius=self.radius,
            steps=self.steps,
            horizontal=self.horizontal,
            serpentine=True,
            u_x=ctx.u_x,
            u_y=ctx.u_y,
            u_z=ctx.u_z,
        )


@dataclass(frozen=True)
class DiscretePattern(MovementPattern):
    points: tuple[tuple[float, float], ...]
    step_duration: float

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        idx = min(int(local_t / self.step_duration), len(self.points) - 1)
        u_off, v_off = self.points[idx]
        # patterns.py normalize unrolled is faster but we need it here
        from satellite.math3d import normalize as norm3d
        return norm3d(ctx.u_z + u_off * ctx.u_x + v_off * ctx.u_y)


def build_aim_context(satellite, *, reset: bool = False) -> AimContext:
    from satellite.sda.satellite import Satellite

    sat: Satellite = satellite
    step_start_aim = sat.bench.bench_boresight.copy()
    if reset:
        sat.receiver.reset_dish_tracking()
    center = sat.bench.initial_boresight.copy()
    u_x, u_y, u_z = basis_at_direction(center)
    return AimContext(
        center=center,
        step_start_aim=step_start_aim,
        u_x=u_x,
        u_y=u_y,
        u_z=u_z,
    )
