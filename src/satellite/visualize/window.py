"""PyVista + Qt visualization window."""

from __future__ import annotations

import numpy as np

from satellite.detection import beam_missed_dish_fov_at_q
from satellite.scenario import ScenarioResult


def _configure_qt_platform() -> None:
    """VTK embedded in Qt needs X11 on many Linux Wayland sessions."""
    import os
    import sys

    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def _install_sigint_handler(app, window) -> None:
    """Allow Ctrl+C to close the window cleanly while Qt owns the event loop."""
    import signal

    def _handle_sigint(_signum, _frame) -> None:
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)

    # Periodically yield to Python so SIGINT is delivered during app.exec().
    from PyQt6.QtCore import QTimer

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)


def run_visualizer(result: ScenarioResult, start_q: float = 0.0) -> None:
    """Open interactive 3D window. Imports pyvista/Qt lazily."""
    _configure_qt_platform()

    import pyvista as pv
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
    from pyvistaqt import QtInteractor

    config = result.config
    viz = config.visualization
    q_max = config.simulation.q_max
    q_step = config.simulation.q_step

    app = QApplication.instance() or QApplication([])

    class SatelliteWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Satellite SDA — {config.name}")
            self.resize(1100, 800)

            self.current_q = float(np.clip(start_q, 0.0, q_max))
            self._cone_poly: pv.PolyData | None = None
            self._cone_actor = None
            self._swept_poly: pv.PolyData | None = None
            self._swept_actor = None
            self._receiver_actor = None
            self._dish_poly: pv.PolyData | None = None
            self._dish_actor = None
            self._dish_ray_poly: pv.PolyData | None = None
            self._dish_ray_actor = None
            self._dish_ray_length = float(np.linalg.norm(result.p1 - result.pt))
            self._scene_built = False
            self._playing = False
            self._log_lines: list[str] = []

            self._play_timer = QTimer(self)
            self._play_timer.setInterval(50)
            self._play_timer.timeout.connect(self._on_play_tick)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self._plot_host = QWidget()
            plot_host_layout = QVBoxLayout(self._plot_host)
            plot_host_layout.setContentsMargins(0, 0, 0, 0)

            self.plotter = QtInteractor(self._plot_host)
            plot_host_layout.addWidget(self.plotter.interactor)

            self._log_frame = QFrame(self._plot_host)
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
            self._log_frame.setMaximumWidth(340)
            self._log_frame.raise_()

            layout.addWidget(self._plot_host, stretch=1)

            controls = QHBoxLayout()
            self.time_label = QLabel()
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(int(q_max / q_step))
            self.slider.setValue(int(self.current_q / q_step))
            self.slider.valueChanged.connect(self._on_slider_changed)

            self.play_btn = QPushButton("Play")
            self.play_btn.clicked.connect(self._toggle_play)

            self.next_btn = QPushButton("Next")
            self.next_btn.setToolTip("Advance to next scenario (not yet implemented)")
            self.next_btn.clicked.connect(self._on_next_scenario)

            controls.addWidget(QLabel("Time q:"))
            controls.addWidget(self.slider, stretch=1)
            controls.addWidget(self.time_label)
            controls.addWidget(self.play_btn)
            controls.addWidget(self.next_btn)
            layout.addLayout(controls)

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._position_log_overlay()

        def _position_log_overlay(self) -> None:
            margin = 12
            self._log_frame.adjustSize()
            w = min(self._log_frame.sizeHint().width(), 340)
            h = self._log_frame.sizeHint().height()
            self._log_frame.setGeometry(
                self._plot_host.width() - w - margin,
                margin,
                w,
                h,
            )
            self._log_frame.raise_()

        def _replay_to(self, q_end: float) -> None:
            """Replay dish tracking to q_end and rebuild the event log."""
            receiver = result.receiver
            transmitter = result.transmitter
            receiver.reset_dish_tracking()

            log_lines = ["S1 Search spiral started"]
            first_detect_q: float | None = None
            slew_logged = False
            miss_logged = False
            dish_fov = config.receiver.dish_fov
            half_fov = dish_fov / 2.0
            alpha = config.sda.alpha
            beam_length = transmitter.beam_length

            q = 0.0
            while q <= q_end + 1e-12:
                had_seen = receiver.dish.has_seen_beam

                if not had_seen and not miss_logged:
                    missed_angle = beam_missed_dish_fov_at_q(
                        q,
                        result.p1,
                        receiver.dish_mount,
                        receiver.dish.boresight,
                        dish_fov,
                        transmitter.boresight_at,
                        alpha,
                        beam_length,
                    )
                    if missed_angle is not None:
                        log_lines.append(
                            f"S2 Missed beam: angle [{missed_angle:.3f} rad] "
                            f"> {half_fov:.3f} rad (FOV {dish_fov:.3f}) "
                            f"(q={q:.3f})"
                        )
                        miss_logged = True

                in_cone = result.in_cone_at_q(q)
                receiver.observe_beam(
                    in_cone,
                    transmitter.boresight_at(q),
                    q_step,
                )

                if receiver.dish.has_seen_beam and not had_seen:
                    incident_deg = np.degrees(receiver.dish.incident_angle or 0.0)
                    log_lines.append(
                        f"S2 Received at angle [{incident_deg:.2f}°] (q={q:.3f})"
                    )
                    first_detect_q = q

                if (
                    first_detect_q is not None
                    and q > first_detect_q + 1e-9
                    and not slew_logged
                ):
                    log_lines.append("S2 Dish slewing toward lock")
                    slew_logged = True

                q += q_step

            if q_end >= q_max - 1e-9:
                log_lines.append("S1 Search spiral complete")
                if first_detect_q is None:
                    log_lines.append("S2 No beam acquisition")

            self._log_lines = log_lines
            self._log_label.setText("\n".join(log_lines))
            self._position_log_overlay()

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._scene_built:
                return
            self._scene_built = True
            # Defer VTK's first render until the native window exists (Wayland/X11).
            QTimer.singleShot(0, self._init_scene)

        def _init_scene(self) -> None:
            self._build_scene()
            self._set_q(self.current_q)

        def _build_scene(self) -> None:
            p = self.plotter

            p.set_background("white")
            p.add_axes()

            p.add_mesh(
                pv.Sphere(radius=0.5, center=result.p1),
                color="blue",
                label="transmitter",
            )
            p.add_mesh(
                pv.Sphere(radius=0.5, center=result.p2),
                color="gray",
                opacity=0.5,
                label="believed target",
            )

            body_radius = config.receiver.body_radius

            self._receiver_actor = p.add_mesh(
                pv.Sphere(radius=body_radius, center=result.pt),
                color="red",
                label="receiver (actual)",
            )

            p.add_mesh(
                pv.Line(result.p1, result.p2),
                color="lightgray",
                line_width=1,
            )
            p.reset_camera()
            cam = p.camera
            old_focal = np.array(cam.focal_point, dtype=np.float64)
            new_focal = np.asarray(result.pt, dtype=np.float64)
            cam.focal_point = new_focal
            cam.position = np.array(cam.position, dtype=np.float64) + (new_focal - old_focal)

        def _to_polydata(self, verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
            if len(verts) == 0 or len(faces) == 0:
                return pv.PolyData()
            faces_pv = np.hstack(
                [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
            )
            return pv.PolyData(verts, faces_pv)

        def _cone_mesh(self, q: float) -> pv.PolyData:
            verts, faces = result.transmitter.cone_mesh_at(
                q,
                viz.cone_u_steps,
                viz.cone_v_steps,
            )
            return self._to_polydata(verts, faces)

        def _swept_area_mesh(self, q: float) -> pv.PolyData:
            verts, faces = result.transmitter.swept_area_mesh_up_to(
                q,
                viz.spiral_trail_steps,
                viz.ribbon_v_steps,
                target_range=result.receiver.dish_range_from_p1(),
            )
            return self._to_polydata(verts, faces)

        def _dish_mesh(self) -> pv.PolyData:
            verts, faces = result.receiver.dish_mesh_at(viz.cone_v_steps)
            return self._to_polydata(verts, faces)

        def _dish_boresight_line(self) -> pv.PolyData:
            mount = result.receiver.dish_mount
            end = mount + result.receiver.dish.boresight * self._dish_ray_length
            return pv.Line(mount, end)

        def _update_line_actor(
            self,
            line: pv.PolyData,
            poly_attr: str,
            actor_attr: str,
            *,
            color: str,
            line_width: float,
            label: str | None = None,
        ) -> None:
            poly = getattr(self, poly_attr)
            actor = getattr(self, actor_attr)

            if poly is None or actor is None:
                setattr(self, poly_attr, line)
                kwargs: dict = {"color": color, "line_width": line_width}
                if label is not None:
                    kwargs["label"] = label
                actor = self.plotter.add_mesh(line, **kwargs)
                setattr(self, actor_attr, actor)
                return

            poly.points = line.points
            poly.Modified()
            actor.mapper.Update()

        def _set_q(self, q: float) -> None:
            """Update time display and scene without changing slider signals."""
            q = float(np.clip(q, 0.0, q_max))
            self.current_q = q
            self.time_label.setText(f"{q:.3f} / {q_max:.3f}")

            self.slider.blockSignals(True)
            self.slider.setValue(int(q / q_step))
            self.slider.blockSignals(False)

            self._replay_to(q)
            self._update_scene()

        def _update_mesh_actor(
            self,
            mesh: pv.PolyData,
            poly_attr: str,
            actor_attr: str,
            *,
            color: str,
            opacity: float,
            label: str | None = None,
        ) -> None:
            """Create actor once, then update points in place to avoid flicker."""
            poly = getattr(self, poly_attr)
            actor = getattr(self, actor_attr)

            if mesh.n_points == 0:
                if actor is not None:
                    actor.SetVisibility(0)
                return

            if poly is None or actor is None:
                setattr(self, poly_attr, mesh)
                kwargs: dict = {
                    "color": color,
                    "opacity": opacity,
                    "show_edges": False,
                }
                if label is not None:
                    kwargs["label"] = label
                actor = self.plotter.add_mesh(mesh, **kwargs)
                setattr(self, actor_attr, actor)
                return

            actor.SetVisibility(1)
            poly.points = mesh.points
            poly.Modified()
            actor.mapper.Update()

        def _update_scene(self) -> None:
            self._update_mesh_actor(
                self._cone_mesh(self.current_q),
                "_cone_poly",
                "_cone_actor",
                color="crimson",
                opacity=0.45,
            )

            self._update_mesh_actor(
                self._swept_area_mesh(self.current_q),
                "_swept_poly",
                "_swept_actor",
                color="orange",
                opacity=0.55,
                label="swept area",
            )

            in_cone = result.in_cone_at_q(self.current_q)
            color = "limegreen" if in_cone else "red"
            prop = self._receiver_actor.GetProperty()
            prop.SetColor(*pv.Color(color).float_rgb)

            self._update_mesh_actor(
                self._dish_mesh(),
                "_dish_poly",
                "_dish_actor",
                color="gold",
                opacity=0.85,
                label="receiver dish",
            )

            self._update_line_actor(
                self._dish_boresight_line(),
                "_dish_ray_poly",
                "_dish_ray_actor",
                color="cyan",
                line_width=4,
                label="dish boresight",
            )

            if self.isVisible():
                self.plotter.render()

        def _on_slider_changed(self, value: int) -> None:
            self._pause()
            self._set_q(value * q_step)

        def _toggle_play(self) -> None:
            if self._playing:
                self._pause()
            else:
                self._play()

        def _play(self) -> None:
            if self.current_q >= q_max:
                self._set_q(0.0)
            self._playing = True
            self.play_btn.setText("Pause")
            self._play_timer.start()

        def _pause(self) -> None:
            self._playing = False
            self._play_timer.stop()
            self.play_btn.setText("Play")

        def _on_play_tick(self) -> None:
            if self.current_q >= q_max:
                self._pause()
                return
            self._set_q(self.current_q + q_step)

        def _on_next_scenario(self) -> None:
            """Stub for Monte Carlo batch — advance to the next scenario."""
            self._pause()

        def closeEvent(self, event) -> None:  # noqa: N802
            self._play_timer.stop()
            self.plotter.close()
            super().closeEvent(event)

    window = SatelliteWindow()
    window.show()
    _install_sigint_handler(app, window)
    app.exec()
