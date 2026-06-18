"""Tests for scan amplitude envelope ramp from center."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from satellite.math.math3d import angle_between
from strategy.movements import Lissajous, Rosette, build_aim_context
from strategy.patterns import lissajous_aim_at, scan_envelope_scale
from satellite.physics.satellite import Satellite
from tests.conftest import base_config


def _aim_ctx(*, ramp: float = 2.0):
    cfg = base_config(chain=("lissajous_scan",))
    cfg = replace(
        cfg,
        satellite=replace(cfg.satellite, scan_envelope_ramp=ramp),
    )
    s1 = Satellite.build("S1", cfg.s1, cfg.s2.position, cfg)
    return build_aim_context(s1)


def test_scan_envelope_scale_endpoints():
    assert scan_envelope_scale(0.0, 2.0) == 0.0
    assert scan_envelope_scale(2.0, 2.0) == 1.0
    assert scan_envelope_scale(10.0, 2.0) == 1.0
    assert scan_envelope_scale(1.0, 0.0) == 1.0


def test_lissajous_starts_at_center_with_envelope():
    ctx = _aim_ctx(ramp=2.0)
    mv = Lissajous(A=0.003, wx=15.5, wy=14.2, delta=2.957)
    aim0 = mv.aim_at(0.0, 100.0, ctx)
    assert angle_between(aim0, ctx.u_z) < 1e-9

    aim_after = mv.aim_at(5.0, 100.0, ctx)
    assert angle_between(aim_after, ctx.u_z) > 1e-6


def test_lissajous_without_envelope_can_start_off_center():
    u_x = np.array([1.0, 0.0, 0.0])
    u_y = np.array([0.0, 1.0, 0.0])
    u_z = np.array([0.0, 0.0, 1.0])
    aim = lissajous_aim_at(
        0.0,
        A=1.0,
        wx=1.0,
        wy=1.0,
        delta=math.pi / 4,
        u_x=u_x,
        u_y=u_y,
        u_z=u_z,
        envelope_ramp=0.0,
    )
    assert angle_between(aim, u_z) > 0.1


def test_rosette_starts_at_center_with_envelope():
    ctx = _aim_ctx(ramp=1.5)
    mv = Rosette(A=0.004, w1=7.0, w2=9.0)
    aim0 = mv.aim_at(0.0, 100.0, ctx)
    assert angle_between(aim0, ctx.u_z) < 1e-9
