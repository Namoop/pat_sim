"""Precomputed eye-view aim trajectories and hover correlation queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from visualize.frames import direction_to_tangent_angles, effective_aim_from_snapshot

if TYPE_CHECKING:
    from scenario.replay import ReplayTimeline
    from scenario.run import ScenarioResult

_MASK_SIZE_DEFAULT = 24
# Typical max union blob area / circular plot area for Lissajous scans.
_HEATMAP_AREA_FRACTION = 0.32


def default_heatmap_area_ceiling(axis_limit: float, fov_radius: float) -> float:
    """Display normalization cap (~max union area seen at end of typical scans)."""
    plot_disc = np.pi * axis_limit * axis_limit
    return max(plot_disc * _HEATMAP_AREA_FRACTION, np.pi * fov_radius * fov_radius)


@dataclass(frozen=True)
class EyeHistoryCache:
    t_values: np.ndarray
    s1_beam_center: np.ndarray
    s2_fov_center: np.ndarray
    alpha: float
    fov_radius: float
    s1_fov_radius: float | None = None
    s2_fov_radius: float | None = None

    @property
    def s1_fov(self) -> float:
        return self.fov_radius if self.s1_fov_radius is None else self.s1_fov_radius

    @property
    def s2_fov(self) -> float:
        return self.fov_radius if self.s2_fov_radius is None else self.s2_fov_radius


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
        s1_fov_radius=result.s1.receiver.dish_fov,
        s2_fov_radius=result.s2.receiver.dish_fov,
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


def _tangent_grids(
    axis_limit: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center θ and φ grids of shape (size, size)."""
    coords = np.linspace(-axis_limit, axis_limit, size, dtype=np.float64)
    theta_g, phi_g = np.meshgrid(coords, coords, indexing="ij")
    return theta_g, phi_g


def _partner_bin_key(theta: float, phi: float, bin_size: float) -> tuple[int, int]:
    inv = 1.0 / bin_size
    return (int(round(theta * inv)), int(round(phi * inv)))


def _partner_centers_from_keys(
    keys: set[tuple[int, int]],
    bin_size: float,
) -> np.ndarray:
    if not keys:
        return np.empty((0, 2), dtype=np.float64)
    centers = np.empty((len(keys), 2), dtype=np.float64)
    for i, (ki, kj) in enumerate(keys):
        centers[i, 0] = ki * bin_size
        centers[i, 1] = kj * bin_size
    return centers


def _stamp_fov_disc(
    mask: np.ndarray,
    theta: float,
    phi: float,
    fov_radius: float,
    theta_g: np.ndarray,
    phi_g: np.ndarray,
    aa_width: float,
) -> None:
    """Merge one antialiased FOV disc into mask in place."""
    dist = np.hypot(theta_g - theta, phi_g - phi)
    edge = np.clip((fov_radius - dist) / aa_width, 0.0, 1.0)
    np.maximum(mask, edge, out=mask)


def _raster_fov_union_mask(
    fov_centers: np.ndarray,
    fov_radius: float,
    axis_limit: float,
    mask_size: int,
) -> np.ndarray:
    """Antialiased union of FOV discs on a square tangent mask."""
    if fov_centers.size == 0:
        return np.zeros((mask_size, mask_size), dtype=np.float64)

    theta_g, phi_g = _tangent_grids(axis_limit, mask_size)
    aa_width = ((2.0 * axis_limit) / mask_size) * 0.5
    mask = np.zeros((mask_size, mask_size), dtype=np.float64)

    for theta, phi in fov_centers:
        _stamp_fov_disc(
            mask,
            float(theta),
            float(phi),
            fov_radius,
            theta_g,
            phi_g,
            aa_width,
        )

    return mask


def _mask_union_area(mask: np.ndarray, pixel_area: float) -> float:
    return float((mask > 0.01).sum()) * pixel_area


def _union_area_from_partner_keys(
    keys: set[tuple[int, int]],
    *,
    fov_radius: float,
    axis_limit: float,
    bin_size: float,
    mask_size: int,
) -> float:
    centers = _partner_centers_from_keys(keys, bin_size)
    mask = _raster_fov_union_mask(centers, fov_radius, axis_limit, mask_size)
    pixel_area = ((2.0 * axis_limit) / mask_size) ** 2
    return _mask_union_area(mask, pixel_area)


def _quadrant_diversity_from_masses(masses: np.ndarray) -> float:
    """Normalized Shannon entropy from four quadrant masses."""
    total = float(masses.sum())
    if total < 1e-12:
        return 0.0
    probs = masses[masses > 0.0] / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / np.log(4.0)


def _quadrant_diversity(
    mask: np.ndarray,
    theta_g: np.ndarray,
    phi_g: np.ndarray,
) -> float:
    """Normalized Shannon entropy of mask mass across θ/φ quadrants."""
    total = float(mask.sum())
    if total < 1e-12:
        return 0.0

    quadrant_masks = (
        (theta_g >= 0.0) & (phi_g >= 0.0),
        (theta_g < 0.0) & (phi_g >= 0.0),
        (theta_g < 0.0) & (phi_g < 0.0),
        (theta_g >= 0.0) & (phi_g < 0.0),
    )
    masses = np.array([float(mask[qm].sum()) for qm in quadrant_masks], dtype=np.float64)
    return _quadrant_diversity_from_masses(masses)


def _partner_quadrant_index(theta: float, phi: float) -> int:
    if theta >= 0.0:
        return 0 if phi >= 0.0 else 3
    return 1 if phi >= 0.0 else 2


def compute_heatmap_area_ceiling(
    cache: EyeHistoryCache,
    *,
    grid_res: int,
    axis_limit: float,
    mask_size: int = _MASK_SIZE_DEFAULT,
) -> float:
    """Return display normalization cap for heatmap union area."""
    del cache, grid_res, mask_size
    return default_heatmap_area_ceiling(axis_limit, cache.fov_radius)


class HeatmapAccumulator:
    """
    Incremental heatmap builder: append replay steps forward, finalize to grid.

    Partner FOV positions are binned per probe cell during extend (fast). Union
    area is rasterized only for dirty cells at finalize.
    """

    def __init__(
        self,
        cache: EyeHistoryCache,
        *,
        source: Literal["S1", "S2"],
        grid_res: int,
        axis_limit: float,
        diversity_floor: float = 0.15,
        area_ceiling: float,
        mask_size: int = _MASK_SIZE_DEFAULT,
    ) -> None:
        partner_fov_radius = cache.s2_fov if source == "S1" else cache.s1_fov

        self._grid_res = grid_res
        self._axis_limit = axis_limit
        self._diversity_floor = diversity_floor
        self._fov_radius = partner_fov_radius
        self._overlap_r = 2.0 * cache.alpha
        self._mask_size = mask_size
        self._area_ceiling = max(area_ceiling, np.pi * partner_fov_radius * partner_fov_radius)
        self._partner_bin_size = partner_fov_radius * 0.5
        self._theta_g, self._phi_g = _tangent_grids(axis_limit, mask_size)
        self._mask_pixel_area = ((2.0 * axis_limit) / mask_size) ** 2
        self._mask_aa = ((2.0 * axis_limit) / mask_size) * 0.5

        coords = np.linspace(-axis_limit, axis_limit, grid_res, dtype=np.float64)
        theta_g, phi_g = np.meshgrid(coords, coords, indexing="ij")
        self._inside = (theta_g * theta_g + phi_g * phi_g) <= axis_limit * axis_limit
        self._grid_pts = np.column_stack([theta_g.ravel(), phi_g.ravel()])

        n_cells = grid_res * grid_res
        self._quadrant_mass = np.zeros((n_cells, 4), dtype=np.int32)
        self._diversity_peak = np.zeros(n_cells, dtype=np.float64)
        self._cell_base = np.zeros(n_cells, dtype=np.float64)
        self._cell_partner_keys: dict[int, set[tuple[int, int]]] = {}
        self._cell_masks: dict[int, np.ndarray] = {}
        self._dirty_cells: set[int] = set()
        self._heat_grid = np.zeros((grid_res, grid_res), dtype=np.float64)
        self._processed_idx = 0
        n_steps = len(cache.t_values)
        self._observation_scale = max(50.0, 0.08 * n_steps)

        if source == "S1":
            self._beams = cache.s1_beam_center
            self._partners = cache.s2_fov_center
        else:
            self._beams = cache.s2_fov_center
            self._partners = cache.s1_beam_center

    @property
    def processed_idx(self) -> int:
        return self._processed_idx

    def _mask_for_cell(self, cell: int) -> np.ndarray:
        mask = self._cell_masks.get(cell)
        if mask is None:
            mask = np.zeros((self._mask_size, self._mask_size), dtype=np.float64)
            self._cell_masks[cell] = mask
        return mask

    def reset(self) -> None:
        self._quadrant_mass.fill(0)
        self._diversity_peak.fill(0.0)
        self._cell_base.fill(0.0)
        self._cell_partner_keys.clear()
        self._cell_masks.clear()
        self._dirty_cells.clear()
        self._heat_grid.fill(0.0)
        self._processed_idx = 0

    def extend_to(self, idx_end: int) -> None:
        """Incorporate replay steps [processed_idx, idx_end)."""
        if idx_end <= self._processed_idx:
            return

        grid_pts = self._grid_pts
        overlap_r = self._overlap_r
        partners = self._partners
        beams = self._beams
        bin_size = self._partner_bin_size
        theta_g = self._theta_g
        phi_g = self._phi_g
        aa_width = self._mask_aa
        fov_radius = self._fov_radius

        for t in range(self._processed_idx, idx_end):
            beam = beams[t]
            diff = grid_pts - beam
            dists = np.hypot(diff[:, 0], diff[:, 1])
            hit_cells = np.flatnonzero(dists < overlap_r)
            if hit_cells.size == 0:
                continue
            partner = partners[t]
            p_theta = float(partner[0])
            p_phi = float(partner[1])
            qi = _partner_quadrant_index(p_theta, p_phi)
            self._quadrant_mass[hit_cells, qi] += 1
            p_key = _partner_bin_key(p_theta, p_phi, bin_size)
            for cell in hit_cells:
                ci = int(cell)
                keys = self._cell_partner_keys.get(ci)
                if keys is None:
                    keys = set()
                    self._cell_partner_keys[ci] = keys
                if p_key in keys:
                    continue
                keys.add(p_key)
                _stamp_fov_disc(
                    self._mask_for_cell(ci),
                    p_theta,
                    p_phi,
                    fov_radius,
                    theta_g,
                    phi_g,
                    aa_width,
                )
                self._dirty_cells.add(ci)

        self._processed_idx = idx_end

    def rebuild_to(self, idx_end: int) -> None:
        """Reset and incorporate replay steps [0, idx_end)."""
        self.reset()
        self.extend_to(idx_end)
        self._dirty_cells = set(self._cell_masks.keys())

    def _update_cell_base(self, cell: int) -> float:
        mask = self._cell_masks.get(cell)
        if mask is None:
            self._cell_base[cell] = 0.0
            return 0.0
        area = _mask_union_area(mask, self._mask_pixel_area)
        if area <= 0.0:
            self._cell_base[cell] = 0.0
            return 0.0

        diversity = _quadrant_diversity(mask, self._theta_g, self._phi_g)
        self._diversity_peak[cell] = max(self._diversity_peak[cell], diversity)

        area_norm = min(area / self._area_ceiling, 1.0)
        diversity_peak = self._diversity_peak[cell]
        base = area_norm * (1.0 + self._diversity_floor * diversity_peak)
        self._cell_base[cell] = base
        return base

    def finalize(self) -> np.ndarray:
        """Score accumulated cells into a display heat grid."""
        for cell in self._dirty_cells:
            self._update_cell_base(cell)
        self._dirty_cells.clear()

        if self._processed_idx <= 0:
            return self._heat_grid

        hits = self._quadrant_mass.sum(axis=1, dtype=np.float64)
        confidence = np.minimum(hits / self._observation_scale, 1.0)
        scores = np.minimum(self._cell_base * confidence, 1.0)

        grid_res = self._grid_res
        score_grid = scores.reshape(grid_res, grid_res)
        updated = np.maximum(self._heat_grid, score_grid)
        self._heat_grid = np.where(self._inside, updated, 0.0)
        return self._heat_grid


def blob_mask_stats(
    fov_centers: np.ndarray,
    fov_radius: float,
    *,
    mask_size: int = _MASK_SIZE_DEFAULT,
    axis_limit: float,
) -> tuple[float, float]:
    """Return (area_rad2, diversity_0_1) from a low-res union raster."""
    mask = _raster_fov_union_mask(
        fov_centers,
        fov_radius,
        axis_limit,
        mask_size,
    )
    theta_g, phi_g = _tangent_grids(axis_limit, mask_size)
    pixel_area = ((2.0 * axis_limit) / mask_size) ** 2
    area = float((mask > 0.01).sum()) * pixel_area
    diversity = _quadrant_diversity(mask, theta_g, phi_g)
    return area, diversity


def build_correlation_heatmap(
    cache: EyeHistoryCache,
    t_max: float,
    *,
    source: Literal["S1", "S2"] = "S1",
    grid_res: int,
    axis_limit: float,
    diversity_floor: float = 0.15,
    mask_size: int = _MASK_SIZE_DEFAULT,
    area_ceiling: float | None = None,
) -> np.ndarray:
    """
    Per-cell heat on a coarse θ/φ grid for probe positions on the source map.

    Score at each cell: area_norm * (1 + diversity_floor * diversity_peak), clipped to [0, 1].
    """
    if area_ceiling is None:
        partner_fov_radius = cache.s2_fov if source == "S1" else cache.s1_fov
        area_ceiling = default_heatmap_area_ceiling(axis_limit, partner_fov_radius)
    idx_end = int(np.searchsorted(cache.t_values, t_max, side="right"))
    acc = HeatmapAccumulator(
        cache,
        source=source,
        grid_res=grid_res,
        axis_limit=axis_limit,
        diversity_floor=diversity_floor,
        area_ceiling=area_ceiling,
        mask_size=mask_size,
    )
    acc.rebuild_to(idx_end)
    return acc.finalize()
