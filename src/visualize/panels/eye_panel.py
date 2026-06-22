"""Angular eye view panel (no playback controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from scenario.run import ScenarioResult
from visualize.diagnostics import FrameProfiler
from visualize.eye_history import (
    EyeHistoryCache,
    HeatmapAccumulator,
    build_eye_history,
    default_heatmap_area_ceiling,
    query_correlated_fov,
)
from visualize.frames import pixel_to_tangent
from visualize.scene import EyeScene, EyeView, build_scene

_EYE_BTN_STYLE = (
    "QPushButton {"
    "  background-color: #f5f5f7;"
    "  border: 1px solid #d2d2d7;"
    "  border-radius: 4px;"
    "  padding: 4px 12px;"
    "  color: #1d1d1f;"
    "  font-family: 'Sans';"
    "  font-size: 11px;"
    "  font-weight: bold;"
    "}"
    "QPushButton:hover {"
    "  background-color: #e8e8ed;"
    "}"
    "QPushButton:checked {"
    "  background-color: #0071e3;"
    "  color: white;"
    "  border-color: #0071e3;"
    "}"
)

_EYE_TITLE_STYLE = (
    "color: #282828; font-family: 'Sans'; font-size: 11px;"
    " font-weight: bold; background: transparent;"
)

CorrelationMode = Literal["none", "target", "cursor", "heatmap"]


@dataclass(frozen=True)
class EyeFrameInfo:
    capture_active: bool
    event_log: tuple[str, ...]


@dataclass(frozen=True)
class _EyeCanvasState:
    view: EyeView
    partner_label: str


@dataclass(frozen=True)
class _BlobCacheKey:
    source: str
    hover_center: tuple[float, float]
    t_max: float
    match_count: int
    plot_w: int
    plot_h: int


@dataclass(frozen=True)
class _HeatmapSetupKey:
    grid_res: int
    axis_limit: float
    alpha: float
    fov_radius: float
    diversity_floor: float
    step_count: int


@dataclass(frozen=True)
class _HeatmapDisplayKey:
    source: str
    plot_w: int
    plot_h: int
    overlay_alpha: int
    idx_end: int


def _enable_smooth_painting(painter: QPainter) -> None:
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)


def _render_correlation_blob(
    plot: QRectF,
    fov_centers: np.ndarray,
    fov_radius: float,
    axis_limit: float,
    alpha: int,
) -> QImage | None:
    """Raster-union of FOV discs into a single tinted image."""
    if fov_centers.size == 0:
        return None

    w = max(1, int(plot.width()))
    h = max(1, int(plot.height()))
    mask = QImage(w, h, QImage.Format.Format_ARGB32)
    mask.fill(0)

    painter = QPainter(mask)
    _enable_smooth_painting(painter)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(255, 255, 255, 255)))

    limit = axis_limit
    for theta, phi in fov_centers:
        u = (theta + limit) / (2.0 * limit)
        v = (limit - phi) / (2.0 * limit)
        cx = u * plot.width()
        cy = v * plot.height()
        r_px = (fov_radius / (2.0 * limit)) * plot.width()
        painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

    painter.end()

    tinted = QImage(w, h, QImage.Format.Format_ARGB32)
    tinted.fill(0)
    tint_painter = QPainter(tinted)
    tint_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    tint_painter.drawImage(0, 0, mask)
    tint_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    tint_painter.fillRect(0, 0, w, h, QColor(68, 136, 255, alpha))
    tint_painter.end()
    return tinted


def _heatmap_to_qimage(
    heat: np.ndarray,
    axis_limit: float,
    width: int,
    height: int,
    overlay_alpha: int,
) -> QImage | None:
    """Map coarse heat grid to a plot-sized warm-orange overlay."""
    g = heat.shape[0]
    if g == 0 or heat.max() <= 0.0:
        return None

    coords = np.linspace(-axis_limit, axis_limit, g, dtype=np.float64)
    theta, phi = np.meshgrid(coords, coords, indexing="ij")
    inside = (theta * theta + phi * phi) <= axis_limit * axis_limit

    alpha_channel = np.zeros((g, g), dtype=np.uint8)
    # Gamma lifts mid-range values for visibility; raw heat stays monotonic.
    scaled = np.power(np.clip(heat, 0.0, 1.0), 0.55)
    mask = inside & (scaled > 0.0)
    alpha_channel[mask] = np.round(overlay_alpha * scaled[mask]).astype(np.uint8)

    # QImage Format_ARGB32 on little-endian: bytes B, G, R, A per pixel.
    argb = np.zeros((g, g, 4), dtype=np.uint8)
    argb[..., 0] = 0
    argb[..., 1] = 136
    argb[..., 2] = 255
    argb[..., 3] = np.flipud(alpha_channel)

    low = QImage(
        argb.tobytes(),
        g,
        g,
        4 * g,
        QImage.Format.Format_ARGB32,
    ).copy()

    return low.scaled(
        max(1, width),
        max(1, height),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


class EyeCanvas(QWidget):
    """Single satellite θ/φ eye plot drawn with QPainter."""

    def __init__(
        self,
        *,
        axis_limit: float,
        parent: QWidget | None = None,
        hover_source: bool = False,
    ) -> None:
        super().__init__(parent)
        self._axis_limit = axis_limit
        self._hover_source = hover_source
        self._state: _EyeCanvasState | None = None
        self._on_hover: Callable[[tuple[float, float] | None], None] | None = None
        self._on_resize: Callable[[], None] | None = None

        self._hover_beam_center: tuple[float, float] | None = None
        self._hover_beam_radius: float = 0.0

        self._correlation_centers: np.ndarray | None = None
        self._correlation_fov_radius: float = 0.0
        self._correlation_blob_alpha: int = 100
        self._blob_cache_key: _BlobCacheKey | None = None
        self._blob_cache_image: QImage | None = None

        self._heatmap_image: QImage | None = None

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        if hover_source:
            self.setMouseTracking(True)

    def set_hover_enabled(self, enabled: bool) -> None:
        self._hover_source = enabled
        self.setMouseTracking(enabled)
        if not enabled and self._on_hover is not None:
            self._on_hover(None)

    @property
    def axis_limit(self) -> float:
        return self._axis_limit

    def set_axis_limit(self, limit: float) -> None:
        self._axis_limit = limit

    def set_hover_callback(
        self,
        callback: Callable[[tuple[float, float] | None], None] | None,
    ) -> None:
        self._on_hover = callback

    def set_resize_callback(self, callback: Callable[[], None] | None) -> None:
        self._on_resize = callback

    def set_view(self, view: EyeView, *, partner_label: str) -> bool:
        """Update view state. Returns True if a repaint was requested."""
        new_state = _EyeCanvasState(view=view, partner_label=partner_label)
        if new_state == self._state:
            return False
        self._state = new_state
        self.update()
        return True

    def set_hover_beam(
        self,
        center: tuple[float, float] | None,
        *,
        radius: float,
    ) -> None:
        if center == self._hover_beam_center and radius == self._hover_beam_radius:
            return
        self._hover_beam_center = center
        self._hover_beam_radius = radius
        self.update()

    def set_correlation_blob(
        self,
        fov_centers: np.ndarray | None,
        *,
        fov_radius: float,
        alpha: int,
        cache_key: _BlobCacheKey | None,
    ) -> None:
        self._correlation_centers = fov_centers
        self._correlation_fov_radius = fov_radius
        self._correlation_blob_alpha = alpha
        if cache_key != self._blob_cache_key:
            self._blob_cache_key = cache_key
            if fov_centers is not None and cache_key is not None:
                plot = self._plot_rect()
                self._blob_cache_image = _render_correlation_blob(
                    plot,
                    fov_centers,
                    fov_radius,
                    self._axis_limit,
                    alpha,
                )
            else:
                self._blob_cache_image = None
        self.update()

    def clear_correlation_blob(self) -> None:
        self.set_correlation_blob(
            None,
            fov_radius=0.0,
            alpha=0,
            cache_key=None,
        )

    def set_heatmap(self, image: QImage | None) -> None:
        if image is self._heatmap_image:
            return
        self._heatmap_image = image
        self.update()

    def clear_heatmap(self) -> None:
        self.set_heatmap(None)

    def _plot_rect(self) -> QRectF:
        margin = 36.0
        title_h = 22.0
        w = float(self.width())
        h = float(self.height())
        avail_w = max(1.0, w - 2 * margin)
        avail_h = max(1.0, h - margin - title_h - margin)
        side = min(avail_w, avail_h)
        left = (w - side) / 2.0
        top = title_h + (avail_h - side) / 2.0 + margin * 0.25
        return QRectF(left, top, side, side)

    def _to_pixel(self, plot: QRectF, theta: float, phi: float) -> QPointF:
        limit = self._axis_limit
        u = (theta + limit) / (2.0 * limit)
        v = (limit - phi) / (2.0 * limit)
        return QPointF(
            plot.left() + u * plot.width(),
            plot.top() + v * plot.height(),
        )

    def _radius_px(self, plot: QRectF, radius_rad: float) -> float:
        return (radius_rad / (2.0 * self._axis_limit)) * plot.width()

    def _tangent_from_mouse(self, pos: QPointF) -> tuple[float, float] | None:
        plot = self._plot_rect()
        if not plot.contains(pos):
            return None
        theta, phi = pixel_to_tangent(
            plot.left(),
            plot.top(),
            plot.width(),
            plot.height(),
            pos.x(),
            pos.y(),
            self._axis_limit,
        )
        limit = self._axis_limit
        if abs(theta) > limit or abs(phi) > limit:
            return None
        return theta, phi

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._hover_source or self._on_hover is None:
            return
        self._on_hover(self._tangent_from_mouse(event.position()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_source and self._on_hover is not None:
            self._on_hover(None)
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._on_resize is not None:
            self._on_resize()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        _enable_smooth_painting(painter)

        plot = self._plot_rect()
        limit = self._axis_limit

        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self._state is None:
            painter.end()
            return

        view = self._state.view
        partner_label = self._state.partner_label

        pen = QPen(QColor(200, 200, 200))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        origin = self._to_pixel(plot, 0.0, 0.0)
        painter.drawLine(
            QPointF(plot.left(), origin.y()),
            QPointF(plot.right(), origin.y()),
        )
        painter.drawLine(
            QPointF(origin.x(), plot.top()),
            QPointF(origin.x(), plot.bottom()),
        )

        boundary_pen = QPen(QColor(136, 136, 136))
        boundary_pen.setStyle(Qt.PenStyle.DashLine)
        boundary_pen.setWidthF(1.0)
        painter.setPen(boundary_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        boundary_center = self._to_pixel(plot, 0.0, 0.0)
        boundary_r = self._radius_px(plot, limit)
        painter.drawEllipse(boundary_center, boundary_r, boundary_r)

        if view.fov is not None:
            fov = view.fov
            center = self._to_pixel(plot, fov.center_theta, fov.center_phi)
            r = self._radius_px(plot, fov.radius)
            painter.setPen(QPen(QColor(17, 85, 204), 2.0))
            painter.setBrush(QBrush(QColor(68, 136, 255, 25)))
            painter.drawEllipse(center, r, r)

        if view.beam is not None:
            beam = view.beam
            center = self._to_pixel(plot, beam.center_theta, beam.center_phi)
            r = self._radius_px(plot, beam.radius)
            if view.is_transmitting:
                fill = QColor(255, 136, 0, 115)
                edge = QColor(204, 102, 0)
            else:
                fill = QColor(255, 204, 136, 90)
                edge = QColor(204, 153, 102)
            painter.setPen(QPen(edge, 1.5))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(center, r, r)

        if view.beam_director is not None:
            bd_pt = self._to_pixel(plot, view.beam_director[0], view.beam_director[1])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(128, 128, 128, 200)))
            painter.drawEllipse(bd_pt, 4.0, 4.0)

        if self._hover_beam_center is not None and self._hover_beam_radius > 0.0:
            h_theta, h_phi = self._hover_beam_center
            h_center = self._to_pixel(plot, h_theta, h_phi)
            h_r = self._radius_px(plot, self._hover_beam_radius)
            hover_pen = QPen(QColor(255, 136, 0), 1.5)
            hover_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(hover_pen)
            painter.setBrush(QBrush(QColor(255, 136, 0, 40)))
            painter.drawEllipse(h_center, h_r, h_r)

        if self._blob_cache_image is not None:
            painter.drawImage(int(plot.left()), int(plot.top()), self._blob_cache_image)

        if self._heatmap_image is not None:
            painter.drawImage(int(plot.left()), int(plot.top()), self._heatmap_image)

        partner_color = QColor(34, 170, 34) if partner_label == "S2" else QColor(204, 34, 34)
        pt = self._to_pixel(plot, view.partner[0], view.partner[1])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(partner_color))
        painter.drawEllipse(pt, 5.0, 5.0)

        tick_font = QFont("Sans", 8)
        painter.setFont(tick_font)
        painter.setPen(QColor(100, 100, 100))
        painter.drawText(int(plot.center().x()) - 20, int(plot.bottom()) + 16, "θ (rad)")
        painter.save()
        painter.translate(10, int(plot.center().y()) + 20)
        painter.rotate(-90)
        painter.drawText(0, 0, "φ (rad)")
        painter.restore()

        lim_label = f"±{limit:.2f}"
        painter.drawText(int(plot.right()) - 36, int(plot.bottom()) + 16, lim_label)

        painter.end()


class _OverlayHost(QWidget):
    """Canvas host that repositions floating overlay controls on resize."""

    def __init__(self, panel: "EyePanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._panel._position_overlay()


class EyePanel:
    """Embedded angular eye view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        self._result: ScenarioResult | None = None

        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_host = _OverlayHost(self, self._widget)
        host_layout = QVBoxLayout(self._canvas_host)
        host_layout.setContentsMargins(0, 0, 0, 0)

        self._view_container = QWidget(self._canvas_host)
        canvas_host_layout = QHBoxLayout(self._view_container)
        canvas_host_layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_s1 = EyeCanvas(axis_limit=1.0)
        self._canvas_s2 = EyeCanvas(axis_limit=1.0)
        canvas_host_layout.addWidget(self._canvas_s1, stretch=1)
        canvas_host_layout.addWidget(self._canvas_s2, stretch=1)
        host_layout.addWidget(self._view_container, stretch=1)

        layout.addWidget(self._canvas_host, stretch=1)

        self._title_s1 = QLabel(self._canvas_host)
        self._title_s1.setStyleSheet(_EYE_TITLE_STYLE)
        self._title_s2 = QLabel(self._canvas_host)
        self._title_s2.setStyleSheet(_EYE_TITLE_STYLE)

        self._btn_bar = QWidget(self._canvas_host)
        self._btn_bar.setAutoFillBackground(True)
        self._btn_bar.setStyleSheet("background-color: #ffffff;")
        btn_bar_layout = QVBoxLayout(self._btn_bar)
        btn_bar_layout.setContentsMargins(0, 0, 0, 0)
        btn_bar_layout.setSpacing(4)

        self._mode_label = QLabel("Coverage overlay", self._btn_bar)
        self._mode_label.setStyleSheet(
            "color: #6e6e73; font-family: 'Sans'; font-size: 11px;"
            " background-color: #ffffff;"
        )
        self._mode_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        btn_bar_layout.addWidget(self._mode_label)

        btn_grid = QWidget(self._btn_bar)
        btn_grid_layout = QGridLayout(btn_grid)
        btn_grid_layout.setContentsMargins(0, 0, 0, 0)
        btn_grid_layout.setHorizontalSpacing(6)
        btn_grid_layout.setVerticalSpacing(6)

        self._btn_group = QButtonGroup(self._btn_bar)
        self._btn_group.setExclusive(True)
        self._mode_buttons: dict[CorrelationMode, QPushButton] = {}
        for row, modes in enumerate(
            [
                [("none", "None"), ("heatmap", "Heatmap")],
                [("target", "Target"), ("cursor", "Cursor")],
            ]
        ):
            for col, (mode, label) in enumerate(modes):
                btn = QPushButton(label, btn_grid)
                btn.setCheckable(True)
                btn.setStyleSheet(_EYE_BTN_STYLE)
                btn.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                self._btn_group.addButton(btn)
                self._mode_buttons[mode] = btn
                btn.clicked.connect(
                    lambda checked, m=mode: self._set_correlation_mode(m)
                )
                btn_grid_layout.addWidget(btn, row, col)

        for col in range(2):
            btn_grid_layout.setColumnStretch(col, 1)

        btn_bar_layout.addWidget(btn_grid)

        self._mode_buttons["target"].setChecked(True)
        self._position_overlay()

        self._profiler = FrameProfiler.from_env(False)
        self._sim_profile_enabled = False
        self._profiling_active = False
        self._replay_step_count = 0
        self._profile_callback = None
        self._initialized = False

        self._history: EyeHistoryCache | None = None
        self._s1_hover_center: tuple[float, float] | None = None
        self._s2_hover_center: tuple[float, float] | None = None
        self._current_t: float = 0.0
        self._correlation_mode: CorrelationMode = "target"
        self._hover_correlation_enabled = True
        self._correlation_blob_alpha = 100
        self._heatmap_grid_resolution = 96
        self._heatmap_diversity_floor = 0.15
        self._last_scene: EyeScene | None = None

        self._heatmap_active_setup: _HeatmapSetupKey | None = None
        self._heatmap_s1_acc: HeatmapAccumulator | None = None
        self._heatmap_s2_acc: HeatmapAccumulator | None = None
        self._heatmap_s1_grid: np.ndarray | None = None
        self._heatmap_s2_grid: np.ndarray | None = None
        self._heatmap_s1_display_key: _HeatmapDisplayKey | None = None
        self._heatmap_s2_display_key: _HeatmapDisplayKey | None = None
        self._heatmap_s1_image: QImage | None = None
        self._heatmap_s2_image: QImage | None = None
        self._heatmap_area_ceiling: float | None = None

        self._canvas_s1.set_hover_callback(self._on_s1_hover)
        self._canvas_s2.set_hover_callback(self._on_s2_hover)
        self._canvas_s1.set_resize_callback(self._on_canvas_resize)
        self._canvas_s2.set_resize_callback(self._on_canvas_resize)
        self._update_hover_tracking()

    @property
    def widget(self):
        return self._widget

    def _position_overlay(self) -> None:
        margin = 12
        host = self._canvas_host
        half_w = max(1, host.width() // 2)
        title_h = 22

        self._title_s1.setGeometry(8, 4, half_w - 16, title_h)
        self._title_s2.setGeometry(half_w + 8, 4, half_w - 16, title_h)

        self._btn_bar.adjustSize()
        bar_w = self._btn_bar.sizeHint().width()
        bar_h = self._btn_bar.sizeHint().height()
        self._btn_bar.setGeometry(
            host.width() - margin - bar_w,
            margin,
            bar_w,
            bar_h,
        )
        if self._title_s1.isVisible():
            self._title_s1.raise_()
        if self._title_s2.isVisible():
            self._title_s2.raise_()
        if self._btn_bar.isVisible():
            self._btn_bar.raise_()

    def set_profile_callback(self, callback) -> None:
        self._profile_callback = callback

    @property
    def profiling_active(self) -> bool:
        return self._profiling_active

    def set_result(self, result: ScenarioResult) -> None:
        self._result = result
        eye_cfg = result.config.eye_viz
        self._canvas_s1.set_axis_limit(eye_cfg.axis_limit)
        self._canvas_s2.set_axis_limit(eye_cfg.axis_limit)
        self._profiler = FrameProfiler.from_env(eye_cfg.profile_frames)
        self._sim_profile_enabled = result.config.simulation.profile_replay
        self._profiling_active = (
            self._profiler.enabled or self._sim_profile_enabled
        )
        self._hover_correlation_enabled = eye_cfg.hover_correlation_enabled
        self._correlation_blob_alpha = eye_cfg.correlation_blob_alpha
        self._heatmap_grid_resolution = eye_cfg.heatmap_grid_resolution
        self._heatmap_diversity_floor = eye_cfg.heatmap_diversity_floor
        self._history = None
        self._s1_hover_center = None
        self._s2_hover_center = None
        self._last_scene = None
        self._initialized = False
        self._clear_heatmap_state()
        self._update_hover_tracking()
        self._clear_all_correlation_overlays()

    def ensure_initialized(self) -> None:
        if self._initialized or self._result is None:
            return
        self._initialized = True
        timeline = self._result.ensure_replay_timeline()
        if self._hover_correlation_enabled:
            self._history = build_eye_history(self._result, timeline)

    def apply_t(self, t: float) -> EyeFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("EyePanel.set_result must be called first")
        self.ensure_initialized()

        total_t = result.playable_t_end
        t = float(np.clip(t, 0.0, total_t))
        self._current_t = t

        if self._profiling_active:
            frame_start = time.perf_counter()
            with self._profiler.measure("replay"):
                event_log = self._replay_to(t)
            self._profiler.set_gauge(
                "replay_steps",
                float(self._replay_step_count),
            )
            with self._profiler.measure("build_scene"):
                scene = build_scene(result, t)
        else:
            event_log = self._replay_to(t)
            scene = build_scene(result, t)

        if self._profiling_active:
            t_paint = time.perf_counter()
            self._update_canvases(scene)
            self._profiler.record("paint", time.perf_counter() - t_paint)
            self._profiler.record(
                "frame_total",
                time.perf_counter() - frame_start,
            )
            self._profiler.end_frame()
            self._emit_profile()
        else:
            self._update_canvases(scene)

        self._last_scene = scene
        self._apply_correlation_overlays(scene)

        return EyeFrameInfo(
            capture_active=scene.capture_active,
            event_log=tuple(event_log),
        )

    def close_panel(self) -> None:
        pass

    def set_overlay_visible(self, visible: bool) -> None:
        self._title_s1.setVisible(visible)
        self._title_s2.setVisible(visible)
        self._btn_bar.setVisible(visible)
        if visible:
            self._position_overlay()

    def capture_record_image(self) -> np.ndarray:
        from visualize.record import pixmap_to_rgb_array

        return pixmap_to_rgb_array(self._view_container.grab())

    def _set_correlation_mode(self, mode: CorrelationMode) -> None:
        if self._correlation_mode == mode:
            return
        self._correlation_mode = mode
        self._s1_hover_center = None
        self._s2_hover_center = None
        self._update_hover_tracking()
        if mode == "heatmap":
            self._clear_heatmap_state()
        self._apply_correlation_overlays(self._last_scene)

    def _update_hover_tracking(self) -> None:
        cursor_active = (
            self._hover_correlation_enabled and self._correlation_mode == "cursor"
        )
        self._canvas_s1.set_hover_enabled(cursor_active)
        self._canvas_s2.set_hover_enabled(cursor_active)

    def _clear_all_correlation_overlays(self) -> None:
        self._canvas_s1.set_hover_beam(None, radius=0.0)
        self._canvas_s2.set_hover_beam(None, radius=0.0)
        self._canvas_s1.clear_correlation_blob()
        self._canvas_s2.clear_correlation_blob()
        self._canvas_s1.clear_heatmap()
        self._canvas_s2.clear_heatmap()

    def _apply_correlation_overlays(self, scene: EyeScene | None) -> None:
        if not self._hover_correlation_enabled or self._history is None:
            self._clear_all_correlation_overlays()
            return

        if self._correlation_mode == "none":
            self._clear_all_correlation_overlays()
            return

        if self._correlation_mode == "heatmap":
            self._canvas_s1.set_hover_beam(None, radius=0.0)
            self._canvas_s2.set_hover_beam(None, radius=0.0)
            self._canvas_s1.clear_correlation_blob()
            self._canvas_s2.clear_correlation_blob()
            self._refresh_heatmap_overlays()
            return

        self._canvas_s1.clear_heatmap()
        self._canvas_s2.clear_heatmap()

        alpha = self._history.alpha

        if self._correlation_mode == "target":
            if scene is None:
                return
            s1_center = scene.s1.partner
            s2_center = scene.s2.partner
            self._canvas_s1.set_hover_beam(s1_center, radius=alpha)
            self._canvas_s2.set_hover_beam(s2_center, radius=alpha)
            self._refresh_s1_to_s2_overlay(s1_center)
            self._refresh_s2_to_s1_overlay(s2_center)
            return

        # cursor mode
        if self._s1_hover_center is None:
            self._canvas_s1.set_hover_beam(None, radius=0.0)
            self._canvas_s2.clear_correlation_blob()
        else:
            self._canvas_s1.set_hover_beam(self._s1_hover_center, radius=alpha)
            self._refresh_s1_to_s2_overlay(self._s1_hover_center)

        if self._s2_hover_center is None:
            self._canvas_s2.set_hover_beam(None, radius=0.0)
            self._canvas_s1.clear_correlation_blob()
        else:
            self._canvas_s2.set_hover_beam(self._s2_hover_center, radius=alpha)
            self._refresh_s2_to_s1_overlay(self._s2_hover_center)

    def _on_s1_hover(self, center: tuple[float, float] | None) -> None:
        if not self._hover_correlation_enabled or self._correlation_mode != "cursor":
            return
        self._s1_hover_center = center
        if center is not None:
            self._s2_hover_center = None
        self._apply_correlation_overlays(self._last_scene)

    def _on_s2_hover(self, center: tuple[float, float] | None) -> None:
        if not self._hover_correlation_enabled or self._correlation_mode != "cursor":
            return
        self._s2_hover_center = center
        if center is not None:
            self._s1_hover_center = None
        self._apply_correlation_overlays(self._last_scene)

    def _on_canvas_resize(self) -> None:
        if self._correlation_mode != "heatmap":
            return
        self._heatmap_s1_display_key = None
        self._heatmap_s2_display_key = None
        self._heatmap_s1_image = None
        self._heatmap_s2_image = None
        self._refresh_heatmap_overlays()

    def _heatmap_setup_key(self) -> _HeatmapSetupKey | None:
        if self._history is None:
            return None
        return _HeatmapSetupKey(
            grid_res=self._heatmap_grid_resolution,
            axis_limit=self._canvas_s1.axis_limit,
            alpha=self._history.alpha,
            fov_radius=self._history.fov_radius,
            diversity_floor=self._heatmap_diversity_floor,
            step_count=int(self._history.t_values.shape[0]),
        )

    def _clear_heatmap_state(self) -> None:
        self._heatmap_active_setup = None
        self._heatmap_s1_acc = None
        self._heatmap_s2_acc = None
        self._heatmap_s1_grid = None
        self._heatmap_s2_grid = None
        self._heatmap_s1_display_key = None
        self._heatmap_s2_display_key = None
        self._heatmap_s1_image = None
        self._heatmap_s2_image = None
        self._heatmap_area_ceiling = None

    def _ensure_heatmap_accumulators(self) -> None:
        if self._history is None:
            return

        setup_key = self._heatmap_setup_key()
        if setup_key is None:
            return
        if (
            self._heatmap_s1_acc is not None
            and self._heatmap_s2_acc is not None
            and setup_key == self._heatmap_active_setup
        ):
            return

        axis_limit = self._canvas_s1.axis_limit
        grid_res = self._heatmap_grid_resolution
        if self._heatmap_area_ceiling is None:
            self._heatmap_area_ceiling = default_heatmap_area_ceiling(
                axis_limit,
                self._history.fov_radius,
            )
        acc_kwargs = {
            "cache": self._history,
            "grid_res": grid_res,
            "axis_limit": axis_limit,
            "diversity_floor": self._heatmap_diversity_floor,
            "area_ceiling": self._heatmap_area_ceiling,
        }
        self._heatmap_s1_acc = HeatmapAccumulator(source="S1", **acc_kwargs)
        self._heatmap_s2_acc = HeatmapAccumulator(source="S2", **acc_kwargs)
        self._heatmap_active_setup = setup_key
        self._heatmap_s1_display_key = None
        self._heatmap_s2_display_key = None
        self._heatmap_s1_image = None
        self._heatmap_s2_image = None

    def _update_heatmap_grids(self, *, force_full: bool = False) -> None:
        if self._history is None:
            return

        self._ensure_heatmap_accumulators()
        if self._heatmap_s1_acc is None or self._heatmap_s2_acc is None:
            return

        idx_end = int(
            np.searchsorted(self._history.t_values, self._current_t, side="right")
        )
        if (
            not force_full
            and self._heatmap_s1_grid is not None
            and idx_end == self._heatmap_s1_acc.processed_idx
        ):
            return

        if force_full or idx_end < self._heatmap_s1_acc.processed_idx:
            self._heatmap_s1_acc.rebuild_to(idx_end)
            self._heatmap_s2_acc.rebuild_to(idx_end)
        else:
            self._heatmap_s1_acc.extend_to(idx_end)
            self._heatmap_s2_acc.extend_to(idx_end)

        self._heatmap_s1_grid = self._heatmap_s1_acc.finalize()
        self._heatmap_s2_grid = self._heatmap_s2_acc.finalize()

    def _heatmap_image_for_canvas(
        self,
        *,
        source: Literal["S1", "S2"],
        canvas: EyeCanvas,
        grid: np.ndarray | None,
    ) -> QImage | None:
        if grid is None:
            return None
        plot = canvas._plot_rect()
        plot_w = max(1, int(plot.width()))
        plot_h = max(1, int(plot.height()))
        display_key = _HeatmapDisplayKey(
            source=source,
            plot_w=plot_w,
            plot_h=plot_h,
            overlay_alpha=self._correlation_blob_alpha,
            idx_end=self._heatmap_s1_acc.processed_idx
            if source == "S1" and self._heatmap_s1_acc is not None
            else self._heatmap_s2_acc.processed_idx
            if self._heatmap_s2_acc is not None
            else 0,
        )
        if source == "S1":
            if display_key == self._heatmap_s1_display_key and self._heatmap_s1_image is not None:
                return self._heatmap_s1_image
            image = _heatmap_to_qimage(
                grid,
                canvas.axis_limit,
                plot_w,
                plot_h,
                self._correlation_blob_alpha,
            )
            self._heatmap_s1_display_key = display_key
            self._heatmap_s1_image = image
            return image

        if display_key == self._heatmap_s2_display_key and self._heatmap_s2_image is not None:
            return self._heatmap_s2_image
        image = _heatmap_to_qimage(
            grid,
            canvas.axis_limit,
            plot_w,
            plot_h,
            self._correlation_blob_alpha,
        )
        self._heatmap_s2_display_key = display_key
        self._heatmap_s2_image = image
        return image

    def _refresh_heatmap_overlays(self) -> None:
        if self._history is None:
            self._canvas_s1.clear_heatmap()
            self._canvas_s2.clear_heatmap()
            return

        force_full = self._heatmap_s1_acc is None
        self._update_heatmap_grids(force_full=force_full)
        self._canvas_s1.set_heatmap(
            self._heatmap_image_for_canvas(
                source="S1",
                canvas=self._canvas_s1,
                grid=self._heatmap_s1_grid,
            )
        )
        self._canvas_s2.set_heatmap(
            self._heatmap_image_for_canvas(
                source="S2",
                canvas=self._canvas_s2,
                grid=self._heatmap_s2_grid,
            )
        )

    def _refresh_s1_to_s2_overlay(
        self,
        center: tuple[float, float],
    ) -> None:
        if not self._hover_correlation_enabled or self._history is None:
            return

        fov_centers = query_correlated_fov(
            self._history,
            center,
            self._current_t,
            source="S1",
        )
        plot = self._canvas_s2._plot_rect()
        cache_key = _BlobCacheKey(
            source="S1",
            hover_center=center,
            t_max=self._current_t,
            match_count=int(fov_centers.shape[0]),
            plot_w=int(plot.width()),
            plot_h=int(plot.height()),
        )
        self._canvas_s2.set_correlation_blob(
            fov_centers,
            fov_radius=self._history.fov_radius,
            alpha=self._correlation_blob_alpha,
            cache_key=cache_key,
        )

    def _refresh_s2_to_s1_overlay(
        self,
        center: tuple[float, float],
    ) -> None:
        if not self._hover_correlation_enabled or self._history is None:
            return

        fov_centers = query_correlated_fov(
            self._history,
            center,
            self._current_t,
            source="S2",
        )
        plot = self._canvas_s1._plot_rect()
        cache_key = _BlobCacheKey(
            source="S2",
            hover_center=center,
            t_max=self._current_t,
            match_count=int(fov_centers.shape[0]),
            plot_w=int(plot.width()),
            plot_h=int(plot.height()),
        )
        self._canvas_s1.set_correlation_blob(
            fov_centers,
            fov_radius=self._history.fov_radius,
            alpha=self._correlation_blob_alpha,
            cache_key=cache_key,
        )

    def _emit_profile(self) -> None:
        if self._profile_callback is None:
            return
        parts: list[str] = []
        result = self._result
        if result is not None:
            sim = result.last_sim_profiler
            if sim is not None and sim.enabled:
                parts.append(sim.format_overlay())
        if self._profiler.enabled:
            parts.append(self._profiler.format_overlay())
        self._profile_callback("\n\n".join(parts))

    def _replay_to(self, t_end: float) -> list[str]:
        result = self._result
        assert result is not None
        log_lines: list[str] = []
        result.replay_to(t_end, event_log=log_lines)
        if self._profiling_active and result.last_sim_profiler is not None:
            self._replay_step_count = result.last_sim_profiler.step_count
        return log_lines

    def _update_canvases(self, scene: EyeScene) -> None:
        role_s1 = "TX" if scene.s1.is_transmitting else "RX"
        role_s2 = "TX" if scene.s2.is_transmitting else "RX"
        self._title_s1.setText(f"{scene.s1.satellite} ({role_s1})")
        self._title_s2.setText(f"{scene.s2.satellite} ({role_s2})")

        if self._canvas_s1.set_view(scene.s1, partner_label="S2"):
            self._canvas_s1.repaint()
        if self._canvas_s2.set_view(scene.s2, partner_label="S1"):
            self._canvas_s2.repaint()
