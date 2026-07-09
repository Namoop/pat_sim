"""Tests for 3D visualization squish transform."""

from __future__ import annotations

import numpy as np

from visualize.viz_squish import (
    SquishContext,
    squish_angle,
    squish_direction,
    squish_length,
    squish_position,
)


def test_squish_identity_is_noop():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1000.0, 0.0, 0.0])
    ctx = SquishContext.from_satellites(p1, p2, 1.0)
    point = np.array([500.0, 2.0, -1.0])
    np.testing.assert_allclose(ctx.position(point), point)
    raw_dir = np.array([1.0, 0.01, 0.0])
    toward = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        ctx.direction(raw_dir, toward_partner=toward),
        raw_dir / np.linalg.norm(raw_dir),
    )
    assert ctx.length(1005.0) == 1005.0
    assert ctx.angle(0.002) == 0.002


def test_squish_compresses_link_axis_distance():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1000.0, 0.0, 0.0])
    ctx = SquishContext.from_satellites(p1, p2, 10.0)
    np.testing.assert_allclose(ctx.position(p1), p1)
    np.testing.assert_allclose(ctx.position(p2), [100.0, 0.0, 0.0])
    np.testing.assert_allclose(ctx.position([500.0, 3.0, 0.0]), [50.0, 3.0, 0.0])


def test_squish_preserves_apparent_cone_radius():
    squish = 10.0
    length = 1005.0
    alpha = 0.00025
    squished_length = squish_length(length, squish)
    squished_alpha = squish_angle(alpha, squish)
    original_radius = length * np.tan(alpha)
    squished_radius = squished_length * np.tan(squished_alpha)
    np.testing.assert_allclose(squished_radius, original_radius, rtol=1e-5)


def test_squish_position_relative_to_origin():
    origin = np.array([0.0, 0.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    squished = squish_position([200.0, 4.0, 0.0], origin, axis, 4.0)
    np.testing.assert_allclose(squished, [50.0, 4.0, 0.0])


def test_squish_direction_keeps_s2_toward_partner():
    toward_s1 = np.array([-1.0, 0.0, 0.0])
    raw = np.array([-0.99999688, -0.0015, -0.002])
    squished = squish_direction(raw, toward_s1, 10.0)
    assert float(np.dot(squished, toward_s1)) > 0.99


def test_environment_toml_default_squish():
    from satellite.config import load_simulation_config

    sim = load_simulation_config("config/Environment.toml")
    assert sim.three_d_viz.squish == 10.0
