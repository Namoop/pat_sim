"""Qt + matplotlib angular map visualization window."""

from __future__ import annotations

import numpy as np

from satellite.mapviz.frames import tangent_disc
from satellite.mapviz.scene import MapPanel, MapScene, build_scene
from satellite.scenario import ScenarioResult


def _configure_qt_platform() -> None:
    """Qt embedded in matplotlib needs X11 on many Linux Wayland sessions."""
    import os
    import sys

    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def _install_sigint_handler(app, window) -> None:
    import signal

    def _handle_sigint(_signum, _frame) -> None:
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)

    from PyQt6.QtCore import QTimer

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)


def run_map_visualizer(result: ScenarioResult, start_q: float = 0.0) -> None:
    """Open interactive 2D angular map window. Imports matplotlib/Qt lazily."""
    _configure_qt_platform()

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication,
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
            self._playing = False
            self._scene_built = False

            self._play_timer = QTimer(self)
            self._play_timer.setInterval(50)
            self._play_timer.timeout.connect(self._on_play_tick)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self._figure = Figure(figsize=(11, 5), dpi=100)
            self._canvas = FigureCanvasQTAgg(self._figure)
            layout.addWidget(self._canvas, stretch=1)

            self._ax_s1 = self._figure.add_subplot(1, 2, 1)
            self._ax_s2 = self._figure.add_subplot(1, 2, 2)
            self._figure.tight_layout(pad=2.0)

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

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._scene_built:
                return
            self._scene_built = True
            QTimer.singleShot(0, self._init_scene)

        def _init_scene(self) -> None:
            timeline = result.ensure_replay_timeline()
            print(f"Replay timeline: {timeline.memory_summary()}")
            self._set_q(self.current_q)

        def _on_slider(self, value: int) -> None:
            self._set_q(value * q_step)

        def _toggle_play(self) -> None:
            self._playing = not self._playing
            self.play_btn.setText("Pause" if self._playing else "Play")
            if self._playing:
                self._play_timer.start()
            else:
                self._play_timer.stop()

        def _on_play_tick(self) -> None:
            next_q = self.current_q + q_step
            if next_q > total_q + 1e-12:
                self._toggle_play()
                return
            self._set_q(next_q)

        def _set_q(self, q: float) -> None:
            q = float(np.clip(q, 0.0, total_q))
            self.current_q = q

            result.replay_to(q)
            scene = build_scene(
                result,
                q,
                spiral_trail_steps=map_cfg.spiral_trail_steps,
            )

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

            self._draw_panel(self._ax_s1, scene.s1, partner_label="S2")
            self._draw_panel(self._ax_s2, scene.s2, partner_label="S1")
            self._canvas.draw_idle()

        def _draw_panel(
            self,
            ax,
            panel: MapPanel,
            *,
            partner_label: str,
        ) -> None:
            ax.clear()
            limit = map_cfg.axis_limit

            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.axhline(0.0, color="#cccccc", linewidth=0.5)
            ax.axvline(0.0, color="#cccccc", linewidth=0.5)
            ax.set_xlabel("θ (rad)")
            ax.set_ylabel("φ (rad)")

            role = "TX" if panel.is_transmitting else "RX"
            ax.set_title(f"{panel.satellite} ({role})")

            boundary = plt.Circle(
                (0.0, 0.0),
                limit,
                fill=False,
                edgecolor="#888888",
                linewidth=1.0,
                linestyle="--",
            )
            ax.add_patch(boundary)

            if panel.fov is not None:
                self._draw_disc(ax, panel.fov, "#4488ff", alpha=0.15, edge="#2266cc")

            if panel.beam is not None:
                color = "#ff8800" if panel.is_transmitting else "#ffcc88"
                self._draw_disc(ax, panel.beam, color, alpha=0.25, edge="#cc6600")

            if len(panel.spiral_trail) > 1:
                ax.plot(
                    panel.spiral_trail[:, 0],
                    panel.spiral_trail[:, 1],
                    color="#ff6600",
                    linewidth=1.5,
                    alpha=0.85,
                )

            partner_color = "#22aa22" if partner_label == "S2" else "#cc2222"
            ax.plot(
                panel.partner[0],
                panel.partner[1],
                "o",
                color=partner_color,
                markersize=8,
                label=partner_label,
            )
            ax.legend(loc="upper right", fontsize=8)

        def _draw_disc(self, ax, disc, facecolor: str, *, alpha: float, edge: str) -> None:
            poly = Polygon(
                tangent_disc(
                    disc.center_theta,
                    disc.center_phi,
                    disc.radius,
                    map_cfg.disc_segments,
                ),
                closed=True,
                facecolor=facecolor,
                edgecolor=edge,
                alpha=alpha,
                linewidth=1.0,
            )
            ax.add_patch(poly)

    window = MapWindow()
    window.show()
    _install_sigint_handler(app, window)
    app.exec()
