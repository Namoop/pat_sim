"""Qt angular map visualization window (QPainter render path)."""

from __future__ import annotations

import time

import numpy as np

from satellite.mapviz.panel_widget import AngularMapPanel
from satellite.mapviz.scene import MapScene, build_scene
from satellite.scenario import ScenarioResult
from satellite.visualize.diagnostics import FrameProfiler


def _configure_qt_platform() -> None:
    """Qt needs X11 on many Linux Wayland sessions."""
    import os
    import sys

    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def _install_sigint_handler(app, window) -> None:
    """Allow Ctrl+C to close the window cleanly while Qt owns the event loop."""
    import signal

    from PyQt6.QtCore import QTimer

    def _handle_sigint(_signum, _frame) -> None:
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)
    window._sigint_timer = QTimer(app)
    window._sigint_timer.timeout.connect(lambda: None)
    window._sigint_timer.start(200)


def run_map_visualizer(result: ScenarioResult, start_q: float = 0.0) -> None:
    """Open interactive 2D angular map window."""
    _configure_qt_platform()

    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    config = result.config
    map_cfg = config.map_visualization
    total_q = result.schedule.total_duration
    q_step = config.simulation.q_step

    app = QApplication.instance() or QApplication([])

    class MapWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Satellite SDA — {config.name} (angular map)")
            self.resize(1100, 650)

            self.current_q = float(np.clip(start_q, 0.0, total_q))
            self._pending_q: float | None = None
            self._playing = False
            self._scene_built = False
            self._frame_busy = False
            self._log_lines: list[str] = []
            # Frame timing (off by default; enable via map_visualization.profile_frames
            # or SATELLITE_VIZ_PROFILE=1).
            self._profiler = FrameProfiler.from_env(map_cfg.profile_frames)
            self._sim_profile_enabled = config.simulation.profile_replay
            self._profiling_active = (
                self._profiler.enabled or self._sim_profile_enabled
            )
            self._replay_step_count = 0

            self._play_timer = QTimer(self)
            self._play_timer.setInterval(50)
            self._play_timer.timeout.connect(self._on_play_tick)

            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._on_debounced_q)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self._canvas_host = QWidget()
            canvas_host_layout = QHBoxLayout(self._canvas_host)
            canvas_host_layout.setContentsMargins(0, 0, 0, 0)

            self._panel_s1 = AngularMapPanel(axis_limit=map_cfg.axis_limit)
            self._panel_s2 = AngularMapPanel(axis_limit=map_cfg.axis_limit)
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
            self._log_title = QLabel("Event Log")
            self._log_title.setStyleSheet(
                "color: #f0f0f8; font-weight: bold; font-size: 12px;"
            )
            self._log_label = QLabel()
            self._log_label.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
            )
            self._log_label.setWordWrap(True)
            self._log_label.setFont(QFont("Monospace", 10))
            self._log_label.setStyleSheet("color: #e2e2ee;")
            log_layout.addWidget(self._log_title)
            log_layout.addWidget(self._log_label)
            self._log_frame.setMaximumWidth(380)
            self._log_frame.raise_()

            layout.addWidget(self._canvas_host, stretch=1)

            controls = QHBoxLayout()
            self.time_label = QLabel()
            controls.addWidget(self.time_label)

            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, int(total_q / q_step)))
            self.slider.valueChanged.connect(self._on_slider)
            controls.addWidget(self.slider, stretch=1)

            self.play_btn = QPushButton("Play")
            self.play_btn.clicked.connect(self._toggle_play)
            controls.addWidget(self.play_btn)

            self.capture_label = QLabel("")
            controls.addWidget(self.capture_label)

            layout.addLayout(controls)

            self._profile_label: QLabel | None = None
            if self._profiling_active:
                self._profile_label = QLabel()
                self._profile_label.setFont(QFont("Monospace", 9))
                self._profile_label.setStyleSheet("color: #555;")
                self._profile_label.setWordWrap(True)
                layout.addWidget(self._profile_label)

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._position_log_overlay()

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

        def _update_profile_overlay(self) -> None:
            if not self._profiling_active or self._profile_label is None:
                return
            parts: list[str] = []
            sim = result.last_sim_profiler
            if sim is not None and sim.enabled:
                parts.append(sim.format_overlay())
            if self._profiler.enabled:
                parts.append(self._profiler.format_overlay())
            self._profile_label.setText("\n\n".join(parts))

        def _replay_to(self, q_end: float) -> None:
            log_lines: list[str] = []
            result.replay_to(q_end, event_log=log_lines)
            self._log_lines = log_lines
            self._log_label.setText("\n".join(log_lines))
            self._position_log_overlay()
            if self._profiling_active and result.last_sim_profiler is not None:
                self._replay_step_count = result.last_sim_profiler.step_count

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._scene_built:
                return
            self._scene_built = True
            QTimer.singleShot(0, self._init_scene)

        def _init_scene(self) -> None:
            result.ensure_replay_timeline()
            self._apply_q(self.current_q)

        def _on_slider(self, value: int) -> None:
            q = value * q_step
            if map_cfg.slider_debounce_ms > 0:
                self._pending_q = q
                self._debounce_timer.start(map_cfg.slider_debounce_ms)
            else:
                self._apply_q(q)

        def _on_debounced_q(self) -> None:
            if self._pending_q is not None:
                self._apply_q(self._pending_q)
                self._pending_q = None

        def _toggle_play(self) -> None:
            self._playing = not self._playing
            self.play_btn.setText("Pause" if self._playing else "Play")
            if self._playing:
                self._play_timer.start()
            else:
                self._play_timer.stop()

        def _on_play_tick(self) -> None:
            if self._frame_busy:
                return
            next_q = self.current_q + q_step
            if next_q > total_q + 1e-12:
                self._toggle_play()
                return
            self._apply_q(next_q)

        def _apply_q(self, q: float) -> None:
            if self._frame_busy:
                return
            self._frame_busy = True
            try:
                self._set_q(q)
            finally:
                self._frame_busy = False

        def _set_q(self, q: float) -> None:
            q = float(np.clip(q, 0.0, total_q))
            self.current_q = q

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

            self.time_label.setText(
                f"q = {q:.3f} / {total_q:.3f} ({scene.s1.phase_label})"
            )
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(q / q_step)))
            self.slider.blockSignals(False)

            self.capture_label.setText(
                "CAPTURE" if scene.capture_active else ""
            )
            self.capture_label.setStyleSheet(
                "color: #008800; font-weight: bold;"
                if scene.capture_active
                else ""
            )

            if self._profiling_active:
                t_paint = time.perf_counter()
                self._update_panels(scene)
                self._profiler.record("paint", time.perf_counter() - t_paint)
                self._profiler.record(
                    "frame_total",
                    time.perf_counter() - frame_start,
                )
                self._profiler.end_frame()
                self._update_profile_overlay()
            else:
                self._update_panels(scene)

        def _update_panels(self, scene: MapScene) -> None:
            if self._panel_s1.set_panel(scene.s1, partner_label="S2"):
                self._panel_s1.repaint()
            if self._panel_s2.set_panel(scene.s2, partner_label="S1"):
                self._panel_s2.repaint()

    window = MapWindow()
    _install_sigint_handler(app, window)
    window.show()
    app.exec()
