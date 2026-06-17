"""Unified tabbed visualization window."""

from __future__ import annotations

from typing import Literal

import numpy as np

from satellite.scenario import format_summary
from satellite.visualize.panels.mag_panel import MagPanel
from satellite.visualize.panels.event_log_panel import EventLogPanel
from satellite.visualize.panels.eye_panel import EyePanel
from satellite.visualize.panels.view3d import View3DPanel
from satellite.visualize.qt_util import configure_qt_platform, install_sigint_handler
from satellite.visualize.session import MonteCarloVizSession, SingleResultSession, VizSession


PLAY_INTERVAL_MS = 50


def play_step_delta(
    t_step: float,
    *,
    autoplay_active: bool,
    autoplay_speed: float | None,
) -> float:
    if autoplay_active and autoplay_speed is not None:
        return t_step * autoplay_speed
    return t_step


def clamp_playable_t(t: float, playable_end: float) -> float:
    return float(np.clip(t, 0.0, playable_end))


def play_reaches_end(next_t: float, playable_end: float) -> bool:
    return next_t > playable_end + 1e-12


def run_visualizer(
    session: VizSession,
    *,
    default_tab: Literal["3d", "eye", "mag"] = "3d",
    start_t: float = 0.0,
    autoplay_speed: float | None = None,
) -> int:
    """Open unified 3D + eye + mag visualizer. Returns process exit code."""
    configure_qt_platform()

    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QHBoxLayout,
        QLabel,
        QMainWindow,
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
    playable_t = result.playable_t_end
    t_step = config.simulation.t_step
    eye_debounce_ms = config.eye_viz.slider_debounce_ms

    app = QApplication.instance() or QApplication([])

    class VisualizerWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._session = session
            self._result = result
            self._start_t = float(start_t)
            self.current_t = clamp_playable_t(start_t, playable_t)
            self._active_tab: Literal["3d", "eye", "mag"] = default_tab
            self._3d_dirty = True
            self._eye_dirty = True
            self._mag_dirty = True
            self._playing = False
            self._autoplay_speed = autoplay_speed
            self._autoplay_active = False
            self._pending_t: float | None = None
            self._shown_once = False

            self._play_timer = QTimer(self)
            self._play_timer.setInterval(PLAY_INTERVAL_MS)
            self._play_timer.timeout.connect(self._on_play_tick)

            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._on_debounced_q)

            self.setWindowTitle(f"Satellite SDA — {session.status_label()}")
            self.resize(1300, 800)

            central = QWidget()
            self.setCentralWidget(central)
            root_layout = QVBoxLayout(central)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            self._controls_bar = QWidget()
            controls = QHBoxLayout(self._controls_bar)
            controls.setContentsMargins(8, 6, 8, 6)

            self._tab_group = QButtonGroup(self)
            self._tab_3d_btn = QPushButton("3D")
            self._tab_3d_btn.setCheckable(True)
            self._tab_eye_btn = QPushButton("Eye")
            self._tab_eye_btn.setCheckable(True)
            self._tab_mag_btn = QPushButton("Mag")
            self._tab_mag_btn.setCheckable(True)
            self._tab_group.addButton(self._tab_3d_btn, 0)
            self._tab_group.addButton(self._tab_eye_btn, 1)
            self._tab_group.addButton(self._tab_mag_btn, 2)
            self._tab_3d_btn.clicked.connect(lambda: self._switch_tab("3d"))
            self._tab_eye_btn.clicked.connect(lambda: self._switch_tab("eye"))
            self._tab_mag_btn.clicked.connect(lambda: self._switch_tab("mag"))
            controls.addWidget(self._tab_3d_btn)
            controls.addWidget(self._tab_eye_btn)
            controls.addWidget(self._tab_mag_btn)

            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, int(playable_t / t_step)))
            self.slider.valueChanged.connect(self._on_slider_changed)
            controls.addWidget(self.slider, stretch=1)

            self._time_label = QLabel()
            self._time_label.setFont(QFont("Monospace", 10))
            controls.addWidget(self._time_label)

            self.play_btn = QPushButton("Play")
            self.play_btn.clicked.connect(self._toggle_play)
            controls.addWidget(self.play_btn)

            self.next_btn = QPushButton("Next")
            self.next_btn.setToolTip("Advance to next Monte Carlo run, or close if none remain")
            self.next_btn.clicked.connect(self._on_next)
            controls.addWidget(self.next_btn)

            controls.addStretch()

            root_layout.addWidget(self._controls_bar)

            self._stack = QStackedWidget()
            self._panel_3d = View3DPanel(self._stack)
            self._panel_eye = EyePanel(self._stack)
            self._panel_mag = MagPanel(self._stack)
            self._stack.addWidget(self._panel_3d.widget)
            self._stack.addWidget(self._panel_eye.widget)
            self._stack.addWidget(self._panel_mag.widget)
            root_layout.addWidget(self._stack, stretch=1)

            self._profile_label = QLabel()
            self._profile_label.setFont(QFont("Monospace", 9))
            self._profile_label.setStyleSheet("color: #555;")
            self._profile_label.setWordWrap(True)
            self._profile_label.setVisible(False)
            root_layout.addWidget(self._profile_label)

            self._event_log = EventLogPanel(central, chrome=self._controls_bar)
            root_layout.addWidget(self._event_log.widget)

            self._panel_3d.set_profile_callback(self._set_profile_text)
            self._panel_eye.set_profile_callback(self._set_profile_text)
            self._panel_mag.set_profile_callback(self._set_profile_text)

            self._load_result(result)
            self._set_tab_ui(default_tab)

        def _set_tab_ui(self, tab: Literal["3d", "eye", "mag"]) -> None:
            self._active_tab = tab
            self._tab_3d_btn.setChecked(tab == "3d")
            self._tab_eye_btn.setChecked(tab == "eye")
            self._tab_mag_btn.setChecked(tab == "mag")
            if tab == "3d":
                self._stack.setCurrentIndex(0)
            elif tab == "eye":
                self._stack.setCurrentIndex(1)
            else:
                self._stack.setCurrentIndex(2)

        def _switch_tab(self, tab: Literal["3d", "eye", "mag"]) -> None:
            if tab == self._active_tab:
                return
            self._set_tab_ui(tab)
            if tab == "3d" and self._3d_dirty:
                self._panel_3d.ensure_initialized()
                info = self._panel_3d.apply_t(self.current_t)
                self._3d_dirty = False
                self._update_frame(info)
            elif tab == "eye" and self._eye_dirty:
                info = self._panel_eye.apply_t(self.current_t)
                self._eye_dirty = False
                self._update_frame(info)
            elif tab == "mag" and self._mag_dirty:
                info = self._panel_mag.apply_t(self.current_t)
                self._mag_dirty = False
                self._update_frame(info)
            elif tab == "3d":
                self._panel_3d.on_tab_shown()
            self._sync_profile_label_visibility()

        def _set_profile_text(self, text: str) -> None:
            self._profile_label.setText(text)

        def _sync_profile_label_visibility(self) -> None:
            if self._active_tab == "3d":
                panel = self._panel_3d
            elif self._active_tab == "eye":
                panel = self._panel_eye
            else:
                panel = self._panel_mag
            active = panel.profiling_active
            self._profile_label.setVisible(active)
            if not active:
                self._profile_label.clear()

        def _load_result(self, new_result) -> None:
            self._result = new_result
            self._panel_3d.set_result(new_result)
            self._panel_eye.set_result(new_result)
            self._panel_mag.set_result(new_result)
            self._3d_dirty = True
            self._eye_dirty = True
            self._mag_dirty = True
            playable = new_result.playable_t_end
            self.current_t = clamp_playable_t(self._start_t, playable)
            self.slider.setMaximum(max(0, int(playable / t_step)))
            self.setWindowTitle(f"Satellite SDA — {self._session.status_label()}")
            self._sync_profile_label_visibility()

        def _update_frame(self, info) -> None:
            playable = self._result.playable_t_end
            self._time_label.setText(f"t {self.current_t:.3f} / {playable:.3f}")
            self.slider.blockSignals(True)
            self.slider.setValue(int(round(self.current_t / t_step)))
            self.slider.blockSignals(False)
            self._event_log.set_lines(info.event_log)

        def _apply_t_active(self, t: float) -> None:
            playable = self._result.playable_t_end
            self.current_t = clamp_playable_t(t, playable)
            if self._active_tab == "3d":
                self._panel_3d.ensure_initialized()
                info = self._panel_3d.apply_t(self.current_t)
                self._3d_dirty = False
                self._eye_dirty = True
                self._mag_dirty = True
                self._update_frame(info)
            elif self._active_tab == "eye":
                info = self._panel_eye.apply_t(self.current_t)
                self._eye_dirty = False
                self._3d_dirty = True
                self._mag_dirty = True
                self._update_frame(info)
            else:
                info = self._panel_mag.apply_t(self.current_t)
                self._mag_dirty = False
                self._3d_dirty = True
                self._eye_dirty = True
                self._update_frame(info)

        def _on_slider_changed(self, value: int) -> None:
            self._pause()
            t = value * t_step
            if (
                self._active_tab == "eye"
                and eye_debounce_ms > 0
            ):
                self._pending_t = t
                self._debounce_timer.start(eye_debounce_ms)
            else:
                self._apply_t_active(t)

        def _on_debounced_q(self) -> None:
            if self._pending_t is not None:
                self._apply_t_active(self._pending_t)
                self._pending_t = None

        def _toggle_play(self) -> None:
            if self._playing:
                self._pause()
            else:
                self._play()

        def _play(self, *, autoplay: bool = False) -> None:
            playable = self._result.playable_t_end
            if self.current_t >= playable:
                self._apply_t_active(0.0)
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
            playable = self._result.playable_t_end
            next_t = self.current_t + play_step_delta(
                t_step,
                autoplay_active=self._autoplay_active,
                autoplay_speed=self._autoplay_speed,
            )
            if play_reaches_end(next_t, playable):
                self._apply_t_active(playable)
                if (
                    self._autoplay_active
                    and isinstance(self._session, MonteCarloVizSession)
                ):
                    self._advance_to_next(autoplay_resume=True)
                else:
                    self._pause(user=False)
                return
            self._apply_t_active(next_t)

        def _set_controls_enabled(self, enabled: bool) -> None:
            self.slider.setEnabled(enabled)
            self.play_btn.setEnabled(enabled)
            self.next_btn.setEnabled(enabled)
            self._tab_3d_btn.setEnabled(enabled)
            self._tab_eye_btn.setEnabled(enabled)
            self._tab_mag_btn.setEnabled(enabled)

        def _advance_to_next(self, *, autoplay_resume: bool = False) -> None:
            self._play_timer.stop()
            self._playing = False
            self._set_controls_enabled(False)
            QApplication.processEvents()

            new_result = self._session.advance()

            self._set_controls_enabled(True)

            if new_result is None:
                self._autoplay_active = False
                self.play_btn.setText("Play")
                self.close()
                return

            self._start_t = 0.0
            self._load_result(new_result)
            self._apply_t_active(self.current_t)
            if autoplay_resume:
                self._play(autoplay=True)

        def _on_next(self) -> None:
            self._pause()
            self._advance_to_next()

        def closeEvent(self, event) -> None:  # noqa: N802
            self._play_timer.stop()
            self._panel_3d.close_panel()
            self._panel_eye.close_panel()
            self._panel_mag.close_panel()
            super().closeEvent(event)

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._shown_once:
                return
            self._shown_once = True
            QTimer.singleShot(0, self._initial_frame)

        def _initial_frame(self) -> None:
            self._apply_t_active(self.current_t)
            if (
                self._autoplay_speed is not None
                and isinstance(self._session, MonteCarloVizSession)
            ):
                self._play(autoplay=True)

    window = VisualizerWindow()
    install_sigint_handler(app, window)
    window.show()
    return app.exec()
