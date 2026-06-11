"""Unified tabbed visualization window."""

from __future__ import annotations

from typing import Literal

import numpy as np

from satellite.scenario import format_summary
from satellite.visualize.panels.map_tab import MapTabPanel
from satellite.visualize.panels.view3d import View3DPanel
from satellite.visualize.qt_util import configure_qt_platform, install_sigint_handler
from satellite.visualize.session import MonteCarloVizSession, SingleResultSession, VizSession


PLAY_INTERVAL_MS = 50


def play_step_delta(
    q_step: float,
    *,
    autoplay_active: bool,
    autoplay_speed: float | None,
) -> float:
    if autoplay_active and autoplay_speed is not None:
        return q_step * autoplay_speed
    return q_step


def run_visualizer(
    session: VizSession,
    *,
    default_tab: Literal["3d", "map"] = "3d",
    start_q: float = 0.0,
    autoplay_speed: float | None = None,
) -> int:
    """Open unified 3D + map visualizer. Returns process exit code."""
    configure_qt_platform()

    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSlider,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    result = session.current()
    if not isinstance(session, MonteCarloVizSession):
        print(format_summary(result))

    config = result.config
    total_q = result.schedule.total_duration
    q_step = config.simulation.q_step
    map_debounce_ms = config.map_visualization.slider_debounce_ms

    app = QApplication.instance() or QApplication([])

    class VisualizerWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._session = session
            self._result = result
            self._start_q = float(start_q)
            self.current_q = float(np.clip(start_q, 0.0, total_q))
            self._active_tab: Literal["3d", "map"] = default_tab
            self._3d_dirty = True
            self._map_dirty = True
            self._playing = False
            self._autoplay_speed = autoplay_speed
            self._autoplay_active = False
            self._pending_q: float | None = None
            self._last_capture = False
            self._shown_once = False

            self._play_timer = QTimer(self)
            self._play_timer.setInterval(PLAY_INTERVAL_MS)
            self._play_timer.timeout.connect(self._on_play_tick)

            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._on_debounced_q)

            self.setWindowTitle(f"Satellite SDA — {session.status_label()}")
            self.resize(1100, 800)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self._stack = QStackedWidget()
            self._panel_3d = View3DPanel(self._stack)
            self._panel_map = MapTabPanel(self._stack)
            self._stack.addWidget(self._panel_3d.widget)
            self._stack.addWidget(self._panel_map.widget)
            layout.addWidget(self._stack, stretch=1)

            controls = QHBoxLayout()

            self._tab_group = QButtonGroup(self)
            self._tab_3d_btn = QPushButton("3D")
            self._tab_3d_btn.setCheckable(True)
            self._tab_map_btn = QPushButton("Map")
            self._tab_map_btn.setCheckable(True)
            self._tab_group.addButton(self._tab_3d_btn, 0)
            self._tab_group.addButton(self._tab_map_btn, 1)
            self._tab_3d_btn.clicked.connect(lambda: self._switch_tab("3d"))
            self._tab_map_btn.clicked.connect(lambda: self._switch_tab("map"))
            controls.addWidget(self._tab_3d_btn)
            controls.addWidget(self._tab_map_btn)

            controls.addWidget(QLabel("Time q:"))
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, int(total_q / q_step)))
            self.slider.valueChanged.connect(self._on_slider_changed)
            controls.addWidget(self.slider, stretch=1)

            self.time_label = QLabel()
            controls.addWidget(self.time_label)

            self.capture_label = QLabel("")
            controls.addWidget(self.capture_label)

            self.play_btn = QPushButton("Play")
            self.play_btn.clicked.connect(self._toggle_play)
            controls.addWidget(self.play_btn)

            self.next_btn = QPushButton("Next")
            self.next_btn.setToolTip("Advance to next Monte Carlo run, or close if none remain")
            self.next_btn.clicked.connect(self._on_next)
            controls.addWidget(self.next_btn)

            self._progress = QProgressBar()
            self._progress.setRange(0, 0)
            self._progress.setFixedWidth(120)
            self._progress.setVisible(False)
            controls.addWidget(self._progress)

            layout.addLayout(controls)

            self._profile_label = QLabel()
            self._profile_label.setFont(QFont("Monospace", 9))
            self._profile_label.setStyleSheet("color: #555;")
            self._profile_label.setWordWrap(True)
            self._profile_label.setVisible(False)
            layout.addWidget(self._profile_label)

            self._panel_3d.set_profile_callback(self._set_profile_text)
            self._panel_map.set_profile_callback(self._set_profile_text)

            self._load_result(result)
            self._set_tab_ui(default_tab)

        def _set_tab_ui(self, tab: Literal["3d", "map"]) -> None:
            self._active_tab = tab
            self._tab_3d_btn.setChecked(tab == "3d")
            self._tab_map_btn.setChecked(tab == "map")
            self._stack.setCurrentIndex(0 if tab == "3d" else 1)

        def _switch_tab(self, tab: Literal["3d", "map"]) -> None:
            if tab == self._active_tab:
                return
            self._set_tab_ui(tab)
            if tab == "3d" and self._3d_dirty:
                self._panel_3d.ensure_initialized()
                info = self._panel_3d.apply_q(self.current_q)
                self._3d_dirty = False
                self._update_time_label(info.phase_label, self._last_capture)
            elif tab == "map" and self._map_dirty:
                info = self._panel_map.apply_q(self.current_q)
                self._map_dirty = False
                self._update_time_label(info.phase_label, info.capture_active)
            elif tab == "3d":
                self._panel_3d.on_tab_shown()
            self._sync_profile_label_visibility()

        def _set_profile_text(self, text: str) -> None:
            self._profile_label.setText(text)

        def _sync_profile_label_visibility(self) -> None:
            panel = self._panel_3d if self._active_tab == "3d" else self._panel_map
            active = panel.profiling_active
            self._profile_label.setVisible(active)
            if not active:
                self._profile_label.clear()

        def _load_result(self, new_result) -> None:
            self._result = new_result
            self._panel_3d.set_result(new_result)
            self._panel_map.set_result(new_result)
            self._3d_dirty = True
            self._map_dirty = True
            total = new_result.schedule.total_duration
            self.current_q = float(np.clip(self._start_q, 0.0, total))
            self.slider.setMaximum(max(0, int(total / q_step)))
            self.setWindowTitle(f"Satellite SDA — {self._session.status_label()}")
            self._sync_profile_label_visibility()

        def _update_time_label(self, phase: str, capture_active: bool) -> None:
            total = self._result.schedule.total_duration
            self.time_label.setText(
                f"{self.current_q:.3f} / {total:.3f} ({phase})"
            )
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(self.current_q / q_step)))
            self.slider.blockSignals(False)
            self.capture_label.setText("CAPTURE" if capture_active else "")
            self.capture_label.setStyleSheet(
                "color: #008800; font-weight: bold;" if capture_active else ""
            )
            self._last_capture = capture_active

        def _apply_q_active(self, q: float) -> None:
            total = self._result.schedule.total_duration
            self.current_q = float(np.clip(q, 0.0, total))
            if self._active_tab == "3d":
                self._panel_3d.ensure_initialized()
                info = self._panel_3d.apply_q(self.current_q)
                self._3d_dirty = False
                self._map_dirty = True
                self._update_time_label(info.phase_label, self._last_capture)
            else:
                info = self._panel_map.apply_q(self.current_q)
                self._map_dirty = False
                self._3d_dirty = True
                self._update_time_label(info.phase_label, info.capture_active)

        def _on_slider_changed(self, value: int) -> None:
            self._pause()
            q = value * q_step
            if (
                self._active_tab == "map"
                and map_debounce_ms > 0
            ):
                self._pending_q = q
                self._debounce_timer.start(map_debounce_ms)
            else:
                self._apply_q_active(q)

        def _on_debounced_q(self) -> None:
            if self._pending_q is not None:
                self._apply_q_active(self._pending_q)
                self._pending_q = None

        def _toggle_play(self) -> None:
            if self._playing:
                self._pause()
            else:
                self._play()

        def _play(self, *, autoplay: bool = False) -> None:
            total = self._result.schedule.total_duration
            if self.current_q >= total:
                self._apply_q_active(0.0)
            self._playing = True
            self.play_btn.setText("Pause")
            if autoplay and self._autoplay_speed is not None:
                self._autoplay_active = True
            self._play_timer.setInterval(PLAY_INTERVAL_MS)
            self._play_timer.start()

        def _pause(self, *, user: bool = True) -> None:
            self._playing = False
            self._play_timer.stop()
            self.play_btn.setText("Play")
            if user:
                self._autoplay_active = False

        def _on_play_tick(self) -> None:
            total = self._result.schedule.total_duration
            next_q = self.current_q + play_step_delta(
                q_step,
                autoplay_active=self._autoplay_active,
                autoplay_speed=self._autoplay_speed,
            )
            if self._autoplay_active and isinstance(self._session, MonteCarloVizSession):
                hit_q = self._result.hit_at_q
                if hit_q is not None:
                    if self.current_q < hit_q - 1e-12:
                        if next_q >= hit_q - 1e-12:
                            self._apply_q_active(hit_q)
                            return
                    else:
                        self._advance_to_next(autoplay_resume=True)
                        return
            if next_q > total + 1e-12:
                if (
                    self._autoplay_active
                    and isinstance(self._session, MonteCarloVizSession)
                ):
                    self._advance_to_next(autoplay_resume=True)
                else:
                    self._pause(user=False)
                return
            self._apply_q_active(next_q)

        def _set_controls_enabled(self, enabled: bool) -> None:
            self.slider.setEnabled(enabled)
            self.play_btn.setEnabled(enabled)
            self.next_btn.setEnabled(enabled)
            self._tab_3d_btn.setEnabled(enabled)
            self._tab_map_btn.setEnabled(enabled)

        def _advance_to_next(self, *, autoplay_resume: bool = False) -> None:
            self._play_timer.stop()
            self._playing = False
            self._set_controls_enabled(False)
            self._progress.setVisible(True)
            QApplication.processEvents()

            new_result = self._session.advance()

            self._progress.setVisible(False)
            self._set_controls_enabled(True)

            if new_result is None:
                self._autoplay_active = False
                self.play_btn.setText("Play")
                self.close()
                return

            self._start_q = 0.0
            self._load_result(new_result)
            self._apply_q_active(self.current_q)
            if autoplay_resume:
                self._play(autoplay=True)

        def _on_next(self) -> None:
            self._pause()
            self._advance_to_next()

        def closeEvent(self, event) -> None:  # noqa: N802
            self._play_timer.stop()
            self._panel_3d.close_panel()
            self._panel_map.close_panel()
            super().closeEvent(event)

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._shown_once:
                return
            self._shown_once = True
            QTimer.singleShot(0, self._initial_frame)

        def _initial_frame(self) -> None:
            self._apply_q_active(self.current_q)
            if (
                self._autoplay_speed is not None
                and isinstance(self._session, MonteCarloVizSession)
            ):
                self._play(autoplay=True)

    window = VisualizerWindow()
    install_sigint_handler(app, window)
    window.show()
    return app.exec()
