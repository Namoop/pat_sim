"""PyVista + Qt visualization window."""

from __future__ import annotations

import time

import numpy as np

from satellite.scenario import ScenarioResult
from satellite.geometry import axis_perpendicular_basis
from satellite.math3d import normalize
from satellite.viz_geometry import cone_mesh_for_aim
from satellite.visualize.diagnostics import FrameProfiler


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
    total_q = result.schedule.total_duration
    q_step = config.simulation.q_step
    body_radius = config.satellite.body_radius
    alpha = config.satellite.alpha
    dish_fov = config.satellite.dish_fov

    app = QApplication.instance() or QApplication([])

    class SatelliteWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Satellite SDA — {config.name}")
            self.resize(1100, 800)

            self.current_q = float(np.clip(start_q, 0.0, total_q))
            self._cone_poly: pv.PolyData | None = None
            self._cone_actor = None
            self._s2_cone_poly: pv.PolyData | None = None
            self._s2_cone_actor = None
            self._s1_fov_poly: pv.PolyData | None = None
            self._s1_fov_actor = None
            self._s2_fov_poly: pv.PolyData | None = None
            self._s2_fov_actor = None
            self._swept_poly: pv.PolyData | None = None
            self._swept_actor = None
            self._s1_body_actor = None
            self._s2_body_actor = None
            self._s1_dish_poly: pv.PolyData | None = None
            self._s1_dish_actor = None
            self._s2_dish_poly: pv.PolyData | None = None
            self._s2_dish_actor = None
            self._s1_ray_poly: pv.PolyData | None = None
            self._s1_ray_actor = None
            self._s2_ray_poly: pv.PolyData | None = None
            self._s2_ray_actor = None
            self._scene_built = False
            self._playing = False
            self._log_lines: list[str] = []
            self._profiler = FrameProfiler.from_env(viz.profile_frames)
            self._sim_profile_enabled = config.simulation.profile_replay
            self._replay_step_count = 0

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
            self._log_frame.setMaximumWidth(380)
            self._log_frame.raise_()

            layout.addWidget(self._plot_host, stretch=1)

            controls = QHBoxLayout()
            self.time_label = QLabel()
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(int(total_q / q_step))
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

            self._profile_label = QLabel()
            self._profile_label.setFont(QFont("Monospace", 9))
            self._profile_label.setStyleSheet("color: #555;")
            self._profile_label.setWordWrap(True)
            if self._profiler.enabled or self._sim_profile_enabled:
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
                self._plot_host.width() - w - margin,
                margin,
                w,
                h,
            )
            self._log_frame.raise_()

        def _update_profile_overlay(self) -> None:
            parts: list[str] = []
            sim = result.last_sim_profiler
            if sim is not None and sim.enabled:
                parts.append(sim.format_overlay())
            if self._profiler.enabled:
                parts.append(self._profiler.format_overlay())
            self._profile_label.setText("\n\n".join(parts))

        def _phase_label(self, q: float) -> str:
            return result.epoch_label(q)

        def _replay_to(self, q_end: float) -> None:
            """Replay coupled simulation to q_end and rebuild the event log."""
            log_lines: list[str] = []
            result.replay_to(q_end, event_log=log_lines)
            self._log_lines = log_lines
            self._log_label.setText("\n".join(log_lines))
            self._position_log_overlay()
            if result.last_sim_profiler is not None:
                self._replay_step_count = result.last_sim_profiler.step_count

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._scene_built:
                return
            self._scene_built = True
            QTimer.singleShot(0, self._init_scene)

        def _init_scene(self) -> None:
            timeline = result.ensure_replay_timeline()
            print(f"Replay timeline: {timeline.memory_summary()}")
            self._build_scene()
            self._set_q(self.current_q)

        def _build_scene(self) -> None:
            p = self.plotter
            p.set_background("white")
            p.add_axes()

            self._s1_body_actor = p.add_mesh(
                pv.Sphere(radius=body_radius, center=result.p1),
                color="blue",
                label="S1",
            )
            self._s2_body_actor = p.add_mesh(
                pv.Sphere(radius=body_radius, center=result.pt),
                color="red",
                label="S2",
            )

            for sat, color, label in (
                (result.s1, "lightgray", "S1 believed aim"),
                (result.s2, "silver", "S2 believed aim"),
            ):
                ray_len = result.boresight_ray_length(sat.name)
                end = sat.position + sat.believed_boresight * ray_len
                p.add_mesh(
                    pv.Line(sat.position, end),
                    color=color,
                    line_width=1,
                    opacity=0.5,
                    label=label,
                )

            p.reset_camera()
            cam = p.camera
            old_focal = np.array(cam.focal_point, dtype=np.float64)
            new_focal = np.asarray(result.pt, dtype=np.float64)
            cam.focal_point = new_focal
            cam.position = np.array(cam.position, dtype=np.float64) + (
                new_focal - old_focal
            )

        def _to_polydata(self, verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
            if len(verts) == 0 or len(faces) == 0:
                return pv.PolyData()
            faces_pv = np.hstack(
                [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
            )
            return pv.PolyData(verts, faces_pv)

        def _cone_mesh(self, satellite: str, q: float) -> pv.PolyData:
            sat = result.s1 if satellite == "S1" else result.s2
            aim = result.bench_aim(satellite, q)
            beam_len = result.beam_length(satellite)
            verts, faces = cone_mesh_for_aim(
                sat.position,
                aim,
                alpha,
                beam_len,
                viz.cone_u_steps,
                viz.cone_v_steps,
            )
            return self._to_polydata(verts, faces)

        def _fov_circle(self, satellite: str, q: float) -> pv.PolyData:
            sat = result.s1 if satellite == "S1" else result.s2
            rx = sat.receiver
            aim = result.bench_aim(satellite, q)
            mount = rx.dish_mount_for_boresight(aim)
            axis = normalize(aim)
            dist_along_aim = float(
                np.dot(sat.partner_actual - mount, axis)
            )
            base_center = mount + axis * dist_along_aim
            base_radius = abs(dist_along_aim) * np.tan(dish_fov)
            u, v = axis_perpendicular_basis(axis)
            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                viz.cone_v_steps,
                endpoint=False,
                dtype=np.float64,
            )
            ring = base_center + base_radius * (
                np.cos(angles)[:, np.newaxis] * u
                + np.sin(angles)[:, np.newaxis] * v
            )
            return pv.lines_from_points(ring, close=True)

        def _swept_area_mesh(self, q: float) -> pv.PolyData:
            return pv.PolyData()

        def _dish_mesh(self, satellite: str, q: float) -> pv.PolyData:
            rx = result.s1.receiver if satellite == "S1" else result.s2.receiver
            boresight = result.dish_boresight_for_display(satellite, q)
            verts, faces = rx.dish_mesh_at(viz.cone_v_steps, boresight=boresight)
            return self._to_polydata(verts, faces)

        def _dish_boresight_line(self, satellite: str, q: float) -> pv.PolyData:
            rx = result.s1.receiver if satellite == "S1" else result.s2.receiver
            boresight = result.dish_boresight_for_display(satellite, q)
            mount = rx.dish_mount_for_boresight(boresight)
            ray_len = result.boresight_ray_length(satellite)
            end = mount + boresight * ray_len
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
            q = float(np.clip(q, 0.0, total_q))
            self.current_q = q
            phase_str = self._phase_label(q)
            frame_start = time.perf_counter()

            self.time_label.setText(
                f"{q:.3f} / {total_q:.3f} ({phase_str})"
            )

            self.slider.blockSignals(True)
            self.slider.setValue(int(q / q_step))
            self.slider.blockSignals(False)

            with self._profiler.measure("replay"):
                self._replay_to(q)
            self._profiler.set_gauge("replay_steps", float(self._replay_step_count))

            with self._profiler.measure("update_scene"):
                self._update_scene()

            self._profiler.record("frame_total", time.perf_counter() - frame_start)
            self._profiler.end_frame()
            self._update_profile_overlay()

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
            q = self.current_q

            with self._profiler.measure("mesh_cone_s1"):
                s1_cone = self._cone_mesh("S1", q)
            with self._profiler.measure("actor_cone_s1"):
                self._update_mesh_actor(
                    s1_cone,
                    "_cone_poly",
                    "_cone_actor",
                    color="crimson",
                    opacity=0.45,
                    label="S1 beam",
                )

            with self._profiler.measure("mesh_cone_s2"):
                s2_cone = self._cone_mesh("S2", q)
            with self._profiler.measure("actor_cone_s2"):
                self._update_mesh_actor(
                    s2_cone,
                    "_s2_cone_poly",
                    "_s2_cone_actor",
                    color="salmon",
                    opacity=0.35,
                    label="S2 beam",
                )

            with self._profiler.measure("mesh_fov_s1"):
                s1_fov = self._fov_circle("S1", q)
            with self._profiler.measure("actor_fov_s1"):
                self._update_line_actor(
                    s1_fov,
                    "_s1_fov_poly",
                    "_s1_fov_actor",
                    color="skyblue",
                    line_width=2,
                    label="S1 FOV",
                )

            with self._profiler.measure("mesh_fov_s2"):
                s2_fov = self._fov_circle("S2", q)
            with self._profiler.measure("actor_fov_s2"):
                self._update_line_actor(
                    s2_fov,
                    "_s2_fov_poly",
                    "_s2_fov_actor",
                    color="skyblue",
                    line_width=2,
                    label="S2 FOV",
                )

            with self._profiler.measure("mesh_swept"):
                swept_mesh = self._swept_area_mesh(q)
            with self._profiler.measure("actor_swept"):
                self._update_mesh_actor(
                    swept_mesh,
                    "_swept_poly",
                    "_swept_actor",
                    color="orange",
                    opacity=0.55,
                    label="swept area",
                )

            with self._profiler.measure("active_in_cone"):
                in_cone = result.active_in_cone(q)
            with self._profiler.measure("actor_body_color"):
                for sat_name, actor in (
                    ("S1", self._s1_body_actor),
                    ("S2", self._s2_body_actor),
                ):
                    base = "blue" if sat_name == "S1" else "red"
                    color = "limegreen" if in_cone else base
                    prop = actor.GetProperty()
                    prop.SetColor(*pv.Color(color).float_rgb)

            with self._profiler.measure("mesh_dish_s1"):
                s1_dish = self._dish_mesh("S1", q)
            with self._profiler.measure("actor_dish_s1"):
                self._update_mesh_actor(
                    s1_dish,
                    "_s1_dish_poly",
                    "_s1_dish_actor",
                    color="gold",
                    opacity=0.85,
                    label="S1 dish",
                )
            with self._profiler.measure("mesh_dish_s2"):
                s2_dish = self._dish_mesh("S2", q)
            with self._profiler.measure("actor_dish_s2"):
                self._update_mesh_actor(
                    s2_dish,
                    "_s2_dish_poly",
                    "_s2_dish_actor",
                    color="gold",
                    opacity=0.85,
                    label="S2 dish",
                )

            # with self._profiler.measure("mesh_ray_s1"):
            #     s1_ray = self._dish_boresight_line("S1", q)
            # with self._profiler.measure("actor_ray_s1"):
            #     self._update_line_actor(
            #         s1_ray,
            #         "_s1_ray_poly",
            #         "_s1_ray_actor",
            #         color="cyan",
            #         line_width=3,
            #         label="S1 boresight",
            #     )
            # with self._profiler.measure("mesh_ray_s2"):
            #     s2_ray = self._dish_boresight_line("S2", q)
            # with self._profiler.measure("actor_ray_s2"):
            #     self._update_line_actor(
            #         s2_ray,
            #         "_s2_ray_poly",
            #         "_s2_ray_actor",
            #         color="cyan",
            #         line_width=3,
            #         label="S2 boresight",
            #     )

            if self.isVisible():
                with self._profiler.measure("render"):
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
            if self.current_q >= total_q:
                self._set_q(0.0)
            self._playing = True
            self.play_btn.setText("Pause")
            self._play_timer.start()

        def _pause(self) -> None:
            self._playing = False
            self._play_timer.stop()
            self.play_btn.setText("Play")

        def _on_play_tick(self) -> None:
            if self.current_q >= total_q:
                self._pause()
                return
            self._set_q(self.current_q + q_step)

        def _on_next_scenario(self) -> None:
            self._pause()

        def closeEvent(self, event) -> None:  # noqa: N802
            self._play_timer.stop()
            self.plotter.close()
            super().closeEvent(event)

    window = SatelliteWindow()
    window.show()
    _install_sigint_handler(app, window)
    app.exec()
