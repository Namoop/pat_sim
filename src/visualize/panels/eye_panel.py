"""Angular eye view panel (no playback controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from scenario.run import ScenarioResult
from visualize.diagnostics import FrameProfiler
from visualize.scene import EyeScene, EyeView, build_scene


@dataclass(frozen=True)
class EyeFrameInfo:
    capture_active: bool
    event_log: tuple[str, ...]


@dataclass(frozen=True)
class _EyeCanvasState:
    view: EyeView
    partner_label: str


class EyeCanvas(QWidget):
    """Single satellite θ/φ eye plot drawn with QPainter."""

    def __init__(self, *, axis_limit: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._axis_limit = axis_limit
        self._state: _EyeCanvasState | None = None
        self._last_paint_seconds = 0.0
        self._on_paint_complete: Callable[[float], None] | None = None
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    @property
    def axis_limit(self) -> float:
        return self._axis_limit

    def set_axis_limit(self, limit: float) -> None:
        self._axis_limit = limit

    @property
    def last_paint_seconds(self) -> float:
        return self._last_paint_seconds

    def set_view(self, view: EyeView, *, partner_label: str) -> bool:
        """Update view state. Returns True if a repaint was requested."""
        new_state = _EyeCanvasState(view=view, partner_label=partner_label)
        if new_state == self._state:
            return False
        self._state = new_state
        self.update()
        return True

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

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        t0 = time.perf_counter()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        plot = self._plot_rect()
        limit = self._axis_limit

        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self._state is None:
            painter.end()
            self._last_paint_seconds = time.perf_counter() - t0
            return

        view = self._state.view
        partner_label = self._state.partner_label
        role = "TX" if view.is_transmitting else "RX"

        painter.setPen(QColor(40, 40, 40))
        painter.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        painter.drawText(
            8,
            18,
            f"{view.satellite} ({role})",
        )

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

        partner_color = QColor(34, 170, 34) if partner_label == "S2" else QColor(204, 34, 34)
        pt = self._to_pixel(plot, view.partner[0], view.partner[1])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(partner_color))
        painter.drawEllipse(pt, 5.0, 5.0)

        painter.setPen(QColor(80, 80, 80))
        painter.setFont(QFont("Sans", 9))
        painter.drawText(
            int(plot.right()) - 28,
            int(plot.top()) + 14,
            partner_label,
        )

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
        self._last_paint_seconds = time.perf_counter() - t0
        if self._on_paint_complete is not None:
            self._on_paint_complete(self._last_paint_seconds)


class EyePanel:
    """Embedded angular eye view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        self._result: ScenarioResult | None = None

        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas_host = QWidget(self._widget)
        canvas_host_layout = QHBoxLayout(canvas_host)
        canvas_host_layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_s1 = EyeCanvas(axis_limit=1.0)
        self._canvas_s2 = EyeCanvas(axis_limit=1.0)
        canvas_host_layout.addWidget(self._canvas_s1, stretch=1)
        canvas_host_layout.addWidget(self._canvas_s2, stretch=1)

        layout.addWidget(canvas_host, stretch=1)

        self._profiler = FrameProfiler.from_env(False)
        self._sim_profile_enabled = False
        self._profiling_active = False
        self._replay_step_count = 0
        self._profile_callback = None
        self._initialized = False

    @property
    def widget(self):
        return self._widget

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
        self._initialized = False

    def ensure_initialized(self) -> None:
        if self._initialized or self._result is None:
            return
        self._initialized = True
        self._result.ensure_replay_timeline()

    def apply_t(self, t: float) -> EyeFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("EyePanel.set_result must be called first")
        self.ensure_initialized()

        total_t = result.playable_t_end
        t = float(np.clip(t, 0.0, total_t))

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

        return EyeFrameInfo(
            capture_active=scene.capture_active,
            event_log=tuple(event_log),
        )

    def close_panel(self) -> None:
        pass

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
        if self._canvas_s1.set_view(scene.s1, partner_label="S2"):
            self._canvas_s1.repaint()
        if self._canvas_s2.set_view(scene.s2, partner_label="S1"):
            self._canvas_s2.repaint()
