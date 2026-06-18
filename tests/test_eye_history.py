# -*- coding: utf-8 -*-
"""Eye history cache and correlation query tests."""

from __future__ import annotations

import numpy as np

from scenario.run import run_scenario
from visualize.eye_history import EyeHistoryCache, build_eye_history, query_correlated_fov
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
