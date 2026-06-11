"""QPainter-based angular map panel (fast 2D render path)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from satellite.mapviz.scene import MapPanel


@dataclass(frozen=True)
class _RenderState:
    panel: MapPanel
    partner_label: str


class AngularMapPanel(QWidget):
    """Single satellite θ/φ map drawn with QPainter."""

    def __init__(self, *, axis_limit: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._axis_limit = axis_limit
        self._state: _RenderState | None = None
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

    def set_panel(self, panel: MapPanel, *, partner_label: str) -> bool:
        """Update panel state. Returns True if a repaint was requested."""
        new_state = _RenderState(panel=panel, partner_label=partner_label)
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

        panel = self._state.panel
        partner_label = self._state.partner_label
        role = "TX" if panel.is_transmitting else "RX"

        painter.setPen(QColor(40, 40, 40))
        painter.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        painter.drawText(
            8,
            18,
            f"{panel.satellite} ({role})",
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

        if panel.fov is not None:
            fov = panel.fov
            center = self._to_pixel(plot, fov.center_theta, fov.center_phi)
            r = self._radius_px(plot, fov.radius)
            painter.setPen(QPen(QColor(17, 85, 204), 2.0))
            painter.setBrush(QBrush(QColor(68, 136, 255, 25)))
            painter.drawEllipse(center, r, r)

        if panel.beam is not None:
            beam = panel.beam
            center = self._to_pixel(plot, beam.center_theta, beam.center_phi)
            r = self._radius_px(plot, beam.radius)
            if panel.is_transmitting:
                fill = QColor(255, 136, 0, 115)
                edge = QColor(204, 102, 0)
            else:
                fill = QColor(255, 204, 136, 90)
                edge = QColor(204, 153, 102)
            painter.setPen(QPen(edge, 1.5))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(center, r, r)

        partner_color = QColor(34, 170, 34) if partner_label == "S2" else QColor(204, 34, 34)
        pt = self._to_pixel(plot, panel.partner[0], panel.partner[1])
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
