"""Precomputed eye-view aim trajectories and hover correlation queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from visualize.frames import direction_to_tangent_angles, effective_aim_from_snapshot

if TYPE_CHECKING:
    from scenario.replay import ReplayTimeline
    from scenario.run import ScenarioResult


@dataclass(frozen=True)
class EyeHistoryCache:
    t_values: np.ndarray
    s1_beam_center: np.ndarray
    s2_fov_center: np.ndarray
    alpha: float
    fov_radius: float


def _aim_center_from_snapshot(
    origin: np.ndarray,
    bench_boresight: np.ndarray,
    fsm_theta: float,
    fsm_phi: float,
) -> tuple[float, float]:
    aim = effective_aim_from_snapshot(bench_boresight, fsm_theta, fsm_phi)
    return direction_to_tangent_angles(origin, aim)


def build_eye_history(
    result: ScenarioResult,
    timeline: ReplayTimeline,
) -> EyeHistoryCache:
    """Derive per-step S1 beam and S2 FOV centers from replay snapshots."""
    n = timeline.step_count
    s1_origin = result.s1.bench.initial_boresight
    s2_origin = result.s2.bench.initial_boresight

    s1_beam = np.empty((n, 2), dtype=np.float64)
    s2_fov = np.empty((n, 2), dtype=np.float64)

    for i in range(n):
        s1_snap = timeline.s1_snapshots[i]
        s2_snap = timeline.s2_snapshots[i]
        s1_beam[i] = _aim_center_from_snapshot(
            s1_origin,
            s1_snap.bench_boresight,
            s1_snap.fsm_theta,
            s1_snap.fsm_phi,
        )
        s2_fov[i] = _aim_center_from_snapshot(
            s2_origin,
            s2_snap.bench_boresight,
            s2_snap.fsm_theta,
            s2_snap.fsm_phi,
        )

    hw = result.config.satellite
    return EyeHistoryCache(
        t_values=timeline.t_values.copy(),
        s1_beam_center=s1_beam,
        s2_fov_center=s2_fov,
        alpha=hw.alpha,
        fov_radius=hw.dish_fov,
    )


def query_correlated_fov(
    cache: EyeHistoryCache,
    hover_center: tuple[float, float],
    t_max: float,
    *,
    source: Literal["S1", "S2"] = "S1",
) -> np.ndarray:
    """
    Return partner FOV centers at steps where source beam overlapped the hover disc.

    S1 source: S2 FOV centers when S1 beam overlapped hover.
    S2 source: S1 FOV centers when S2 beam overlapped hover.

    Overlap uses disc-disc test: center distance < 2 * alpha.
    """
    if source == "S1":
        beam_centers = cache.s1_beam_center
        partner_fov = cache.s2_fov_center
    else:
        beam_centers = cache.s2_fov_center
        partner_fov = cache.s1_beam_center

    idx_end = int(np.searchsorted(cache.t_values, t_max, side="right"))
    if idx_end <= 0:
        return np.empty((0, 2), dtype=np.float64)

    centers = beam_centers[:idx_end]
    dists = np.hypot(
        centers[:, 0] - hover_center[0],
        centers[:, 1] - hover_center[1],
    )
    mask = dists < (2.0 * cache.alpha)
    return partner_fov[:idx_end][mask]
