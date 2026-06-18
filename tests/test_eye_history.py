# -*- coding: utf-8 -*-
"""Eye history cache and correlation query tests."""

from __future__ import annotations

import numpy as np

from scenario.run import run_scenario
from visualize.eye_history import (
    EyeHistoryCache,
    HeatmapAccumulator,
    blob_mask_stats,
    build_correlation_heatmap,
    build_eye_history,
    default_heatmap_area_ceiling,
    query_correlated_fov,
)
from visualize.frames import discs_overlap
from visualize.scene import build_view
from tests.conftest import base_config


def test_discs_overlap_separated():
    assert not discs_overlap((0.0, 0.0), 0.1, (1.0, 0.0), 0.1)


def test_discs_overlap_touching():
    assert discs_overlap((0.0, 0.0), 0.1, (0.2, 0.0), 0.1)


def test_discs_overlap_nested():
    assert discs_overlap((0.0, 0.0), 0.5, (0.0, 0.0), 0.1)


def test_build_eye_history_matches_build_view_at_t0():
    cfg = base_config(chain=("asymmetric_swap",))
    result = run_scenario(cfg)
    timeline = result.ensure_replay_timeline()
    cache = build_eye_history(result, timeline)

    assert cache.s1_beam_center.shape == (timeline.step_count, 2)
    assert cache.s2_fov_center.shape == (timeline.step_count, 2)
    assert len(cache.t_values) == timeline.step_count

    s1_view = build_view(result, "S1", 0.0)
    s2_view = build_view(result, "S2", 0.0)
    assert s1_view.beam is not None
    assert s2_view.fov is not None

    np.testing.assert_allclose(
        cache.s1_beam_center[0],
        (s1_view.beam.center_theta, s1_view.beam.center_phi),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cache.s2_fov_center[0],
        (s2_view.fov.center_theta, s2_view.fov.center_phi),
        atol=1e-12,
    )


def test_query_correlated_fov_synthetic_sweep():
    t_values = np.linspace(0.0, 1.0, 21)
    s1_beam = np.column_stack([t_values * 0.01, np.zeros(21)])
    s2_fov = np.column_stack([np.zeros(21), t_values * 0.02])
    cache = EyeHistoryCache(
        t_values=t_values,
        s1_beam_center=s1_beam,
        s2_fov_center=s2_fov,
        alpha=0.005,
        fov_radius=0.02,
    )

    hover = (0.005, 0.0)
    matches = query_correlated_fov(cache, hover, t_max=1.0)
    assert matches.shape[0] > 0
    expected_idx = np.where(
        np.hypot(s1_beam[:, 0] - hover[0], s1_beam[:, 1] - hover[1]) < 2 * cache.alpha
    )[0]
    np.testing.assert_array_equal(matches, s2_fov[expected_idx])


def test_query_respects_t_max():
    t_values = np.array([0.0, 0.5, 1.0])
    s1_beam = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    s2_fov = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    cache = EyeHistoryCache(
        t_values=t_values,
        s1_beam_center=s1_beam,
        s2_fov_center=s2_fov,
        alpha=1.0,
        fov_radius=0.1,
    )

    matches = query_correlated_fov(cache, (0.0, 0.0), t_max=0.5)
    assert matches.shape == (2, 2)
    np.testing.assert_allclose(matches[0], [1.0, 1.0])
    np.testing.assert_allclose(matches[1], [2.0, 2.0])

    matches_end = query_correlated_fov(cache, (0.0, 0.0), t_max=0.0)
    assert matches_end.shape == (1, 2)


def test_query_correlated_fov_s2_to_s1_reverse():
    t_values = np.linspace(0.0, 1.0, 21)
    s1_fov = np.column_stack([np.zeros(21), t_values * 0.02])
    s2_beam = np.column_stack([t_values * 0.01, np.zeros(21)])
    cache = EyeHistoryCache(
        t_values=t_values,
        s1_beam_center=s1_fov,
        s2_fov_center=s2_beam,
        alpha=0.005,
        fov_radius=0.02,
    )

    hover = (0.005, 0.0)
    matches = query_correlated_fov(cache, hover, t_max=1.0, source="S2")
    assert matches.shape[0] > 0
    expected_idx = np.where(
        np.hypot(s2_beam[:, 0] - hover[0], s2_beam[:, 1] - hover[1]) < 2 * cache.alpha
    )[0]
    np.testing.assert_array_equal(matches, s1_fov[expected_idx])


def test_blob_mask_stats_diversity_single_quadrant():
    limit = 0.1
    radius = 0.02
    centers = np.array([[0.04, 0.04]], dtype=np.float64)
    area, diversity = blob_mask_stats(
        centers,
        radius,
        axis_limit=limit,
        mask_size=64,
    )
    assert area > 0.0
    assert diversity < 0.1


def test_blob_mask_stats_diversity_four_quadrants():
    limit = 0.1
    radius = 0.015
    offset = 0.04
    centers = np.array(
        [
            [offset, offset],
            [-offset, offset],
            [-offset, -offset],
            [offset, -offset],
        ],
        dtype=np.float64,
    )
    area, diversity = blob_mask_stats(
        centers,
        radius,
        axis_limit=limit,
        mask_size=64,
    )
    assert area > 0.0
    assert diversity > 0.85


def test_blob_mask_stats_area_monotonicity():
    limit = 0.1
    radius = 0.02
    few = np.array([[0.0, 0.0]], dtype=np.float64)
    many = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]], dtype=np.float64)
    area_few, _ = blob_mask_stats(few, radius, axis_limit=limit, mask_size=64)
    area_many, _ = blob_mask_stats(many, radius, axis_limit=limit, mask_size=64)
    assert area_many >= area_few


def test_build_correlation_heatmap_synthetic_sweep():
    t_values = np.linspace(0.0, 1.0, 21)
    s1_beam = np.column_stack([t_values * 0.01, np.zeros(21)])
    s2_fov = np.column_stack([np.zeros(21), t_values * 0.02])
    cache = EyeHistoryCache(
        t_values=t_values,
        s1_beam_center=s1_beam,
        s2_fov_center=s2_fov,
        alpha=0.005,
        fov_radius=0.02,
    )

    heat = build_correlation_heatmap(
        cache,
        t_max=1.0,
        source="S1",
        grid_res=32,
        axis_limit=0.05,
        diversity_floor=0.15,
    )
    assert heat.shape == (32, 32)
    assert heat.max() > 0.0
    assert np.count_nonzero(heat) > 0

    heat_short = build_correlation_heatmap(
        cache,
        t_max=-1.0,
        source="S1",
        grid_res=32,
        axis_limit=0.05,
    )
    assert heat_short.max() == 0.0


def test_build_correlation_heatmap_diversity_only_boosts():
    limit = 0.1
    radius = 0.02
    centers = np.array([[0.04, 0.04]], dtype=np.float64)
    _, diversity = blob_mask_stats(
        centers,
        radius,
        axis_limit=limit,
        mask_size=64,
    )
    assert diversity < 0.05
    area_norm = 0.5
    floor = 0.15
    heat_low_div = area_norm * (1.0 + floor * diversity)
    heat_high_div = area_norm * (1.0 + floor * 0.9)
    assert heat_high_div > heat_low_div


def test_union_area_ignores_repeated_partner_position():
    limit = 0.1
    radius = 0.02
    one = np.array([[0.02, 0.02]], dtype=np.float64)
    many = np.repeat(one, 50, axis=0)
    area_one, _ = blob_mask_stats(one, radius, axis_limit=limit, mask_size=32)
    area_many, _ = blob_mask_stats(many, radius, axis_limit=limit, mask_size=32)
    assert area_many <= area_one * 1.05


def test_heatmap_heat_monotonic_during_forward_extend():
    t_values = np.linspace(0.0, 1.0, 21)
    s1_beam = np.column_stack([t_values * 0.01, np.zeros(21)])
    s2_fov = np.column_stack([np.zeros(21), t_values * 0.02])
    cache = EyeHistoryCache(
        t_values=t_values,
        s1_beam_center=s1_beam,
        s2_fov_center=s2_fov,
        alpha=0.005,
        fov_radius=0.02,
    )
    acc = HeatmapAccumulator(
        cache,
        source="S1",
        grid_res=32,
        axis_limit=0.05,
        diversity_floor=0.15,
        area_ceiling=default_heatmap_area_ceiling(0.05, cache.fov_radius),
    )
    prev = np.zeros((32, 32), dtype=np.float64)
    for end in range(1, 21):
        acc.extend_to(end)
        heat = acc.finalize()
        assert np.all(heat >= prev - 1e-12)
        prev = heat


def test_heatmap_accumulator_incremental_matches_full_build():
    t_values = np.linspace(0.0, 1.0, 21)
    s1_beam = np.column_stack([t_values * 0.01, np.zeros(21)])
    s2_fov = np.column_stack([np.zeros(21), t_values * 0.02])
    cache = EyeHistoryCache(
        t_values=t_values,
        s1_beam_center=s1_beam,
        s2_fov_center=s2_fov,
        alpha=0.005,
        fov_radius=0.02,
    )
    limit = 0.05
    grid_res = 32
    ceiling = default_heatmap_area_ceiling(limit, cache.fov_radius)

    acc = HeatmapAccumulator(
        cache,
        source="S1",
        grid_res=grid_res,
        axis_limit=limit,
        diversity_floor=0.15,
        area_ceiling=ceiling,
    )
    acc.extend_to(8)
    acc.extend_to(15)
    incremental = acc.finalize()

    full = build_correlation_heatmap(
        cache,
        t_max=float(t_values[14]),
        source="S1",
        grid_res=grid_res,
        axis_limit=limit,
        diversity_floor=0.15,
        area_ceiling=ceiling,
    )
    np.testing.assert_allclose(incremental, full)
