"""Angular distance 2D visualization panel."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QButtonGroup, QGridLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from satellite.mapviz.frames import direction_to_tangent_angles
from satellite.mapviz.scene import MapScene, build_scene
from satellite.math3d import angle_between
from satellite.scenario import ScenarioResult
from satellite.visualize.diagnostics import FrameProfiler

ViewMode = Literal["center", "s1", "s2", "follow_s1", "follow_s2", "balance"]


@dataclass(frozen=True)
class MagFrameInfo:
    capture_active: bool
    event_log: tuple[str, ...]


class MagPanelWidget(QWidget):
    """Custom drawn 2D side-view widget for the 'mag' visualization."""

    _TILT_ANIM_DURATION = 0.4
    _DIAGRAM_SCALE = 1.5
    _SAT_RADIUS = 8.0

    @staticmethod
    def _ease_tilt(t: float) -> float:
        """Ease-in-out cubic: zero slope at start and end."""
        if t < 0.5:
            return 4.0 * t * t * t
        return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene: MapScene | None = None
        self._result: ScenarioResult | None = None
        self._config = None
        self._last_paint_seconds = 0.0
        self._view_mode: ViewMode = "center"
        self._scene_signed_tilt = 0.0
        self._display_signed_tilt = 0.0
        self._display_pin_blend = 0.0
        self._display_vertical_center_blend = 0.0
        self._camera_target_tilt = 0.0
        self._camera_target_pin_blend = 0.0
        self._camera_target_vertical_center_blend = 0.0
        self._anim_start_tilt = 0.0
        self._anim_start_pin_blend = 0.0
        self._anim_start_vertical_center_blend = 0.0
        self._anim_start_time = 0.0
        self._tilt_timer = QTimer(self)
        self._tilt_timer.setInterval(16)
        self._tilt_timer.timeout.connect(self._on_tilt_tick)

        # Create overlay button group in a top-right grid
        self._btn_bar = QWidget(self)
        self._btn_layout = QGridLayout(self._btn_bar)
        self._btn_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_layout.setHorizontalSpacing(6)
        self._btn_layout.setVerticalSpacing(6)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._buttons: dict[ViewMode, QPushButton] = {}
        for row, modes in enumerate(
            [
                [("s1", "Focus S1"), ("center", "Center"), ("s2", "Focus S2")],
                [("follow_s1", "Follow S1"), ("balance", "Balance"), ("follow_s2", "Follow S2")],
            ]
        ):
            for col, (mode, label) in enumerate(modes):
                btn = QPushButton(label, self._btn_bar)
                btn.setCheckable(True)
                if mode == "center":
                    btn.setChecked(True)

                btn.setStyleSheet(
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
                btn.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                self._btn_group.addButton(btn)
                self._buttons[mode] = btn
                btn.clicked.connect(lambda checked, m=mode: self.set_view_mode(m))
                self._btn_layout.addWidget(btn, row, col)

        for col in range(3):
            self._btn_layout.setColumnStretch(col, 1)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(400, 300)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_view_mode(self, mode: ViewMode) -> None:
        if mode == self._view_mode:
            return
        self._view_mode = mode
        if mode in self._buttons:
            self._buttons[mode].setChecked(True)
        self._scene_signed_tilt = self._compute_target_signed_tilt()
        self._begin_camera_animation(
            self._scene_signed_tilt,
            self._mode_pin_blend(mode),
            self._mode_vertical_center_blend(mode),
        )

    @staticmethod
    def _mode_pin_blend(mode: ViewMode) -> float:
        if mode in ("center", "balance"):
            return 0.0
        return 1.0

    @staticmethod
    def _mode_vertical_center_blend(mode: ViewMode) -> float:
        return 1.0 if mode in ("follow_s1", "follow_s2") else 0.0

    @staticmethod
    def _is_live_tilt_mode(mode: ViewMode) -> bool:
        return mode in ("follow_s1", "follow_s2", "balance")

    def _diagram_reference_cy(
        self,
        height: float,
        bottom_margin: float,
        link_span: float,
        baseline_tilt: float,
        pin_blend: float,
        vertical_center_blend: float,
    ) -> tuple[float, float, float]:
        cy1_off, cy2_off = self._satellite_y_positions(
            0.0, link_span, baseline_tilt, pin_blend
        )
        lowest_off = max(cy1_off, cy2_off)
        cy_bottom = height - bottom_margin - (self._SAT_RADIUS + lowest_off) * self._DIAGRAM_SCALE
        cy_center = height / 2.0
        cy = (1.0 - vertical_center_blend) * cy_bottom + vertical_center_blend * cy_center
        return cy, cy + cy1_off, cy + cy2_off

    def _scale_factor(self) -> float:
        if self._config is None:
            return 1.0
        axis_limit = self._config.eye_viz.axis_limit
        visual_limit_deg = self._config.mag_viz.visual_limit_deg
        if axis_limit <= 1e-9:
            return 1.0
        return math.radians(visual_limit_deg) / axis_limit

    def _compute_target_signed_tilt(self) -> float:
        if self._result is None:
            return 0.0
        scale_factor = self._scale_factor()
        if self._view_mode in ("s1", "follow_s1"):
            boresight = (
                self._result.s1.bench.bench_boresight
                if self._view_mode == "follow_s1"
                else self._result.s1.bench.initial_boresight
            )
            tilt = angle_between(
                boresight,
                self._result.s1.bench.toward_partner,
            ) * scale_factor
            return +tilt
        if self._view_mode in ("s2", "follow_s2"):
            boresight = (
                self._result.s2.bench.bench_boresight
                if self._view_mode == "follow_s2"
                else self._result.s2.bench.initial_boresight
            )
            tilt = angle_between(
                boresight,
                self._result.s2.bench.toward_partner,
            ) * scale_factor
            return -tilt
        if self._view_mode == "balance":
            alpha1 = angle_between(
                self._result.s1.bench.bench_boresight,
                self._result.s1.bench.toward_partner,
            ) * scale_factor
            alpha2 = angle_between(
                self._result.s2.bench.bench_boresight,
                self._result.s2.bench.toward_partner,
            ) * scale_factor
            return (alpha1 - alpha2) / 2.0
        return 0.0

    @staticmethod
    def _satellite_y_positions(
        cy: float,
        link_span: float,
        signed_tilt: float,
        pin_blend: float,
    ) -> tuple[float, float]:
        half = link_span / 2.0
        sin_t = math.sin(signed_tilt)
        bal_y1 = cy - half * sin_t
        bal_y2 = cy + half * sin_t

        sin_m = math.sin(abs(signed_tilt))
        if signed_tilt > 0.0:
            pin_y1, pin_y2 = cy, cy + link_span * sin_m
        elif signed_tilt < 0.0:
            pin_y1, pin_y2 = cy + link_span * sin_m, cy
        else:
            pin_y1, pin_y2 = cy, cy

        b = pin_blend
        return (1.0 - b) * bal_y1 + b * pin_y1, (1.0 - b) * bal_y2 + b * pin_y2

    def _begin_camera_animation(
        self,
        target_signed_tilt: float,
        target_pin_blend: float,
        target_vertical_center_blend: float,
    ) -> None:
        self._camera_target_tilt = target_signed_tilt
        self._camera_target_pin_blend = target_pin_blend
        self._camera_target_vertical_center_blend = target_vertical_center_blend
        if (
            abs(self._camera_target_tilt - self._display_signed_tilt) < 1e-9
            and abs(self._camera_target_pin_blend - self._display_pin_blend) < 1e-9
            and abs(
                self._camera_target_vertical_center_blend
                - self._display_vertical_center_blend
            )
            < 1e-9
        ):
            self._display_signed_tilt = self._camera_target_tilt
            self._display_pin_blend = self._camera_target_pin_blend
            self._display_vertical_center_blend = self._camera_target_vertical_center_blend
            self._tilt_timer.stop()
            self.update()
            return
        self._anim_start_tilt = self._display_signed_tilt
        self._anim_start_pin_blend = self._display_pin_blend
        self._anim_start_vertical_center_blend = self._display_vertical_center_blend
        self._anim_start_time = time.perf_counter()
        if not self._tilt_timer.isActive():
            self._tilt_timer.start()

    def _on_tilt_tick(self) -> None:
        elapsed = time.perf_counter() - self._anim_start_time
        t = min(1.0, elapsed / self._TILT_ANIM_DURATION)
        eased = self._ease_tilt(t)
        self._display_signed_tilt = (
            self._anim_start_tilt
            + (self._camera_target_tilt - self._anim_start_tilt) * eased
        )
        self._display_pin_blend = (
            self._anim_start_pin_blend
            + (self._camera_target_pin_blend - self._anim_start_pin_blend) * eased
        )
        self._display_vertical_center_blend = (
            self._anim_start_vertical_center_blend
            + (
                self._camera_target_vertical_center_blend
                - self._anim_start_vertical_center_blend
            )
            * eased
        )
        if t >= 1.0:
            self._display_signed_tilt = self._scene_signed_tilt
            self._display_pin_blend = self._camera_target_pin_blend
            self._display_vertical_center_blend = self._camera_target_vertical_center_blend
            self._tilt_timer.stop()
        self.update()

    def _position_buttons(self) -> None:
        margin = 12
        self._btn_bar.adjustSize()
        bar_w = self._btn_bar.sizeHint().width()
        bar_h = self._btn_bar.sizeHint().height()
        self._btn_bar.setGeometry(self.width() - margin - bar_w, margin, bar_w, bar_h)
        self._btn_bar.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_buttons()

    def set_scene(self, scene: MapScene, result: ScenarioResult) -> None:
        result_changed = result is not self._result
        self._scene = scene
        self._result = result
        self._config = result.config
        if result_changed:
            self._tilt_timer.stop()
            self._scene_signed_tilt = self._compute_target_signed_tilt()
            pin_blend = self._mode_pin_blend(self._view_mode)
            vertical_center_blend = self._mode_vertical_center_blend(self._view_mode)
            self._display_signed_tilt = self._scene_signed_tilt
            self._display_pin_blend = pin_blend
            self._display_vertical_center_blend = vertical_center_blend
            self._camera_target_tilt = self._scene_signed_tilt
            self._camera_target_pin_blend = pin_blend
            self._camera_target_vertical_center_blend = vertical_center_blend
            if self._view_mode in self._buttons:
                self._buttons[self._view_mode].setChecked(True)
        else:
            self._scene_signed_tilt = self._compute_target_signed_tilt()
            if self._is_live_tilt_mode(self._view_mode):
                self._camera_target_tilt = self._scene_signed_tilt
                if not self._tilt_timer.isActive():
                    self._display_signed_tilt = self._scene_signed_tilt
            elif not self._tilt_timer.isActive():
                self._display_signed_tilt = self._scene_signed_tilt
        self.update()

    def _make_cone_polygon(
        self,
        cx: float,
        cy: float,
        angle_center: float,
        half_angle: float,
        length: float,
    ) -> QPolygonF:
        p0 = QPointF(cx, cy)
        p1 = QPointF(
            cx + length * math.cos(angle_center - half_angle),
            cy + length * math.sin(angle_center - half_angle),
        )
        p2 = QPointF(
            cx + length * math.cos(angle_center + half_angle),
            cy + length * math.sin(angle_center + half_angle),
        )
        poly = QPolygonF()
        poly.append(p0)
        poly.append(p1)
        poly.append(p2)
        return poly

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        t0 = time.perf_counter()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Clear background (white)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self._scene is None or self._result is None or self._config is None:
            painter.end()
            self._last_paint_seconds = time.perf_counter() - t0
            return

        scene = self._scene
        result = self._result
        config = self._config

        W = float(self.width())
        H = float(self.height())
        cx1 = W * 0.25
        cx2 = W * 0.75
        link_span = cx2 - cx1

        # Compute scaling factors based on config
        axis_limit = config.eye_viz.axis_limit
        visual_limit_deg = config.mag_viz.visual_limit_deg
        bottom_margin = config.mag_viz.bottom_margin

        fov_cone_length = link_span * config.mag_viz.fov_cone_length
        beam_cone_length = link_span * config.mag_viz.beam_cone_length

        scale_factor = (
            math.radians(visual_limit_deg) / axis_limit
            if axis_limit > 1e-9
            else 1.0
        )

        # Everything is drawn in the baseline frame (_display_signed_tilt).
        # On mode switch the baseline slews; during playback it snaps to scene tilt
        # when the camera is not animating.
        baseline_tilt = self._display_signed_tilt
        pin_blend = self._display_pin_blend
        cy, cy1, cy2 = self._diagram_reference_cy(
            H,
            bottom_margin,
            link_span,
            baseline_tilt,
            pin_blend,
            self._display_vertical_center_blend,
        )

        # Title (fixed size; not part of the scaled diagram)
        painter.setPen(QColor(40, 40, 40))
        painter.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        painter.drawText(15, 25, "2D Magnitude Alignment Profile")

        painter.save()
        painter.translate(W * 0.5, cy)
        painter.scale(self._DIAGRAM_SCALE, self._DIAGRAM_SCALE)
        painter.translate(-W * 0.5, -cy)

        # Draw the baseline
        los_pen = QPen(QColor(210, 210, 210), 1.0)
        los_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(los_pen)
        painter.drawLine(QPointF(cx1, cy1), QPointF(cx2, cy2))

        # While tilted, draw the horizontal reference.
        if abs(baseline_tilt) > 1e-6:
            ref_pen = QPen(QColor(220, 220, 220), 1.0)
            ref_pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(ref_pen)
            painter.drawLine(QPointF(cx1, cy), QPointF(cx2, cy))

        alpha = config.satellite.alpha
        dish_fov = config.satellite.dish_fov

        visual_alpha = alpha * scale_factor
        visual_fov = dish_fov * scale_factor

        for sat_idx, (sat, panel, cx, cy_sat, name, base_angle, color) in enumerate(
            [
                (result.s1, scene.s1, cx1, cy1, "S1", 0.0 + baseline_tilt, QColor(0, 102, 204)),
                (result.s2, scene.s2, cx2, cy2, "S2", math.pi + baseline_tilt, QColor(204, 34, 34)),
            ]
        ):
            toward_partner = sat.bench.toward_partner  # always the reference

            # 1. Transmitter beam pointing offset relative to toward_partner
            tx_aim = sat.bench.bench_boresight
            theta_tx, phi_tx = direction_to_tangent_angles(toward_partner, tx_aim)
            offset_mag_tx = math.hypot(theta_tx, phi_tx)
            scaled_offset_tx = offset_mag_tx * scale_factor

            # 2. Receiver FOV pointing offset relative to toward_partner
            rx_aim = sat.receiver.fsm.effective_receive_boresight(sat.bench.bench_boresight)
            theta_rx, phi_rx = direction_to_tangent_angles(toward_partner, rx_aim)
            offset_mag_rx = math.hypot(theta_rx, phi_rx)
            scaled_offset_rx = offset_mag_rx * scale_factor

            # Cone offsets come from the current replay step; base_angle is in the
            # baseline frame so beams move with the slew during mode changes.
            if name == "S1":
                angle_center_tx = base_angle - scaled_offset_tx
                angle_center_rx = base_angle - scaled_offset_rx
            else:
                angle_center_tx = base_angle + scaled_offset_tx
                angle_center_rx = base_angle + scaled_offset_rx

            # 1. Draw Transmitter Orange Beam Cone if transmitting
            if panel.is_transmitting:
                beam_poly = self._make_cone_polygon(
                    cx,
                    cy_sat,
                    angle_center_tx,
                    visual_alpha,
                    beam_cone_length,
                )
                # Fill the polygon with NoPen outline
                painter.setPen(Qt.PenStyle.NoPen)
                partner_panel = scene.s2 if name == "S1" else scene.s1
                if partner_panel.partner_in_fov:
                    # Normal orange beam
                    painter.setBrush(QBrush(QColor(255, 136, 0, 80)))
                    outline_pen = QPen(QColor(255, 102, 0), 1.5)
                else:
                    # Gray beam with decent opacity
                    painter.setBrush(QBrush(QColor(180, 180, 180, 100)))
                    outline_pen = QPen(QColor(140, 140, 140), 1.5)
                painter.drawPolygon(beam_poly)

                # Draw only the two side lines
                painter.setPen(outline_pen)
                painter.drawLine(beam_poly[0], beam_poly[1])
                painter.drawLine(beam_poly[0], beam_poly[2])

            # 2. Draw Receiver FOV Dotted Cone if active
            if panel.fov is not None:
                fov_poly = self._make_cone_polygon(
                    cx,
                    cy_sat,
                    angle_center_rx,
                    visual_fov,
                    fov_cone_length,
                )
                # Apply transparency to the receiver FOV dotted lines
                fov_color = QColor(color)
                fov_color.setAlpha(150)
                fov_pen = QPen(fov_color, 1.5)
                fov_pen.setStyle(Qt.PenStyle.DotLine)
                painter.setPen(fov_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # Draw only the two side lines
                painter.drawLine(fov_poly[0], fov_poly[1])
                painter.drawLine(fov_poly[0], fov_poly[2])

            # 3. Draw Satellite Point
            painter.setPen(QPen(QColor(0, 0, 0), 1.5))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(cx, cy_sat), self._SAT_RADIUS, self._SAT_RADIUS)

            # 4. Draw Label
            painter.setPen(QColor(40, 40, 40))
            painter.setFont(QFont("Sans", 10, QFont.Weight.Bold))
            painter.drawText(int(cx) - 10, int(cy_sat) - 14, name)

        painter.restore()
        painter.end()
        self._last_paint_seconds = time.perf_counter() - t0


class MagPanel:
    """Embedded angular magnitude view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        self._result: ScenarioResult | None = None
        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = MagPanelWidget(self._widget)
        layout.addWidget(self._canvas, stretch=1)

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
        self._profiler = FrameProfiler.from_env(result.config.eye_viz.profile_frames)
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

    def apply_t(self, t: float) -> MagFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("MagPanel.set_result must be called first")
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
            self._canvas.set_scene(scene, result)
            self._profiler.record("paint", time.perf_counter() - t_paint)
            self._profiler.record(
                "frame_total",
                time.perf_counter() - frame_start,
            )
            self._profiler.end_frame()
            self._emit_profile()
        else:
            self._canvas.set_scene(scene, result)

        return MagFrameInfo(
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
