"""Angular map view panel (no playback controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from satellite.mapviz.panel_widget import AngularMapPanel
from satellite.mapviz.scene import MapScene, build_scene
from satellite.scenario import ScenarioResult
from satellite.visualize.diagnostics import FrameProfiler


@dataclass(frozen=True)
class MapTabFrameInfo:
    phase_label: str
    capture_active: bool


class MapTabPanel:
    """Embedded angular map view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

        self._result: ScenarioResult | None = None
        self._current_q = 0.0

        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_host = QWidget(self._widget)
        canvas_host_layout = QHBoxLayout(self._canvas_host)
        canvas_host_layout.setContentsMargins(0, 0, 0, 0)

        self._panel_s1 = AngularMapPanel(axis_limit=1.0)
        self._panel_s2 = AngularMapPanel(axis_limit=1.0)
        canvas_host_layout.addWidget(self._panel_s1, stretch=1)
        canvas_host_layout.addWidget(self._panel_s2, stretch=1)

        self._log_frame = QFrame(self._canvas_host)
        self._log_frame.setObjectName("eventLog")
        self._log_frame.setStyleSheet(
            "#eventLog {"
            "  background-color: rgba(18, 18, 28, 215);"
            "  border: 1px solid rgba(220, 220, 235, 90);"
            "  border-radius: 4px;"
            "}"
        )
        log_layout = QVBoxLayout(self._log_frame)
        log_layout.setContentsMargins(10, 8, 10, 8)
        log_title = QLabel("Event Log")
        log_title.setStyleSheet(
            "color: #f0f0f8; font-weight: bold; font-size: 12px;"
        )
        self._log_label = QLabel()
        self._log_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._log_label.setWordWrap(True)
        self._log_label.setFont(QFont("Monospace", 10))
        self._log_label.setStyleSheet("color: #e2e2ee;")
        log_layout.addWidget(log_title)
        log_layout.addWidget(self._log_label)
        self._log_frame.setMaximumWidth(380)
        self._log_frame.raise_()

        layout.addWidget(self._canvas_host, stretch=1)

        self._profiler = FrameProfiler.from_env(False)
        self._sim_profile_enabled = False
        self._profiling_active = False
        self._replay_step_count = 0
        self._profile_callback = None
        self._initialized = False

        from PyQt6.QtCore import QObject, QEvent

        class _ResizeForwarder(QObject):
            def __init__(self, panel: MapTabPanel) -> None:
                super().__init__()
                self._panel = panel

            def eventFilter(self, obj, event):  # noqa: N802
                if event.type() == QEvent.Type.Resize:
                    self._panel._position_log_overlay()
                return False

        self._resize_forwarder = _ResizeForwarder(self)
        self._widget.installEventFilter(self._resize_forwarder)

    @property
    def widget(self):
        return self._widget

    def set_profile_callback(self, callback) -> None:
        self._profile_callback = callback

    def set_result(self, result: ScenarioResult) -> None:
        self._result = result
        map_cfg = result.config.map_visualization
        self._panel_s1.set_axis_limit(map_cfg.axis_limit)
        self._panel_s2.set_axis_limit(map_cfg.axis_limit)
        self._profiler = FrameProfiler.from_env(map_cfg.profile_frames)
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

    def apply_q(self, q: float) -> MapTabFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("MapTabPanel.set_result must be called first")
        self.ensure_initialized()

        total_q = result.schedule.total_duration
        q = float(np.clip(q, 0.0, total_q))
        self._current_q = q

        if self._profiling_active:
            frame_start = time.perf_counter()
            with self._profiler.measure("replay"):
                self._replay_to(q)
            self._profiler.set_gauge(
                "replay_steps",
                float(self._replay_step_count),
            )
            with self._profiler.measure("build_scene"):
                scene = build_scene(result, q)
        else:
            self._replay_to(q)
            scene = build_scene(result, q)

        if self._profiling_active:
            t_paint = time.perf_counter()
            self._update_panels(scene)
            self._profiler.record("paint", time.perf_counter() - t_paint)
            self._profiler.record(
                "frame_total",
                time.perf_counter() - frame_start,
            )
            self._profiler.end_frame()
            self._emit_profile()
        else:
            self._update_panels(scene)

        return MapTabFrameInfo(
            phase_label=scene.s1.phase_label,
            capture_active=scene.capture_active,
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

    def _position_log_overlay(self) -> None:
        margin = 12
        self._log_frame.adjustSize()
        w = min(self._log_frame.sizeHint().width(), 380)
        h = self._log_frame.sizeHint().height()
        self._log_frame.setGeometry(
            self._canvas_host.width() - w - margin,
            margin,
            w,
            h,
        )
        self._log_frame.raise_()

    def _replay_to(self, q_end: float) -> None:
        result = self._result
        assert result is not None
        log_lines: list[str] = []
        result.replay_to(q_end, event_log=log_lines)
        self._log_label.setText("\n".join(log_lines))
        self._position_log_overlay()
        if self._profiling_active and result.last_sim_profiler is not None:
            self._replay_step_count = result.last_sim_profiler.step_count

    def _update_panels(self, scene: MapScene) -> None:
        if self._panel_s1.set_panel(scene.s1, partner_label="S2"):
            self._panel_s1.repaint()
        if self._panel_s2.set_panel(scene.s2, partner_label="S1"):
            self._panel_s2.repaint()
