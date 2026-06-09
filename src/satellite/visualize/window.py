"""PyVista + Qt visualization window."""

from __future__ import annotations

import time

import numpy as np

from satellite.detection import beam_hits_dish_at_q, beam_missed_dish_fov_at_q
from satellite.math3d import angle_between
from satellite.schedule import SearchPhase
from satellite.scenario import ScenarioResult
from satellite.sda.satellite import build_phase2_transmitter
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
    q_max = config.simulation.q_max
    total_q = result.schedule.total_duration
    q_step = config.simulation.q_step
    body_radius = config.satellite.body_radius

    app = QApplication.instance() or QApplication([])

    class SatelliteWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"Satellite SDA — {config.name}")
            self.resize(1100, 800)

            self.current_q = float(np.clip(start_q, 0.0, total_q))
            self._cone_poly: pv.PolyData | None = None
            self._cone_actor = None
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
            if self._profiler.enabled:
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

        def _count_replay_step(self) -> None:
            self._replay_step_count += 1

        def _phase_label(self, q: float) -> str:
            phase, _ = result.schedule.phase_at(q)
            return "Phase 1" if phase is SearchPhase.S1_TRANSMIT else "Phase 2"

        def _format_rx_angles(
            self,
            q: float,
            rx_name: str,
            receiver,
            transmitter,
            dish_boresight: np.ndarray,
            *,
            prefix: str,
        ) -> str:
            dish_fov = config.satellite.dish_fov
            toward_partner = receiver.nominal_boresight
            local_q = result.local_q(q)
            beam_axis = transmitter.boresight_at(local_q)
            toward_source = -beam_axis / np.linalg.norm(beam_axis)

            receiver_offset = np.degrees(angle_between(toward_partner, dish_boresight))
            beam_offset = np.degrees(angle_between(toward_partner, beam_axis))
            incident = np.degrees(angle_between(dish_boresight, toward_source))
            fov_deg = np.degrees(dish_fov)

            return (
                f"{prefix} (q={q:.3f})\n"
                f"  Receiver offset: {receiver_offset:.2f}°\n"
                f"  Beam boresight offset: {beam_offset:.2f}°\n"
                f"  Incident angle: {incident:.2f}° (FOV {fov_deg:.2f}°)"
            )

        def _replay_to(self, q_end: float) -> None:
            """Replay coupled simulation to q_end and rebuild the event log."""
            q_end = float(np.clip(q_end, 0.0, total_q))
            q_max_local = q_max
            self._replay_step_count = 0

            s2 = result.s2.receiver
            init_offset_deg = np.degrees(s2.initial_pointing_offset)
            cfg_offset_deg = np.degrees(s2.configured_offset_magnitude)
            log_lines = [
                "S1 Search spiral started",
                (
                    f"S2 Initial receiver offset: {init_offset_deg:.2f}° "
                    f"(θ/φ magnitude {cfg_offset_deg:.2f}°)"
                ),
            ]

            result.s2.receiver.reset_dish_tracking()
            s2_miss_logged = False
            s2_first_detect: float | None = None
            s2_slew_logged = False

            with self._profiler.measure("replay_sim"):
                q = 0.0
                while q < q_max_local - 1e-12 and q <= q_end + 1e-12:
                    tx = result.s1.transmitter
                    had_seen = s2.has_seen_beam
                    dish_boresight = s2.dish_boresight.copy()

                    if not had_seen and not s2_miss_logged:
                        missed = beam_missed_dish_fov_at_q(
                            q,
                            tx.position,
                            s2.dish_mount,
                            dish_boresight,
                            s2.dish_fov,
                            tx.boresight_at,
                            tx.alpha,
                            tx.beam_length,
                        )
                        if missed is not None:
                            log_lines.append(
                                self._format_rx_angles(
                                    q,
                                    "S2",
                                    s2,
                                    tx,
                                    dish_boresight,
                                    prefix="S2 Missed beam",
                                )
                            )
                            s2_miss_logged = True

                    in_cone = beam_hits_dish_at_q(
                        q,
                        tx.position,
                        s2.dish_mount,
                        dish_boresight,
                        s2.dish_fov,
                        tx.boresight_at,
                        tx.alpha,
                        tx.beam_length,
                    )
                    s2.observe_beam(in_cone, tx.boresight_at(q), q_step)
                    self._count_replay_step()

                    if s2.has_seen_beam and not had_seen:
                        log_lines.append(
                            self._format_rx_angles(
                                q,
                                "S2",
                                s2,
                                tx,
                                dish_boresight,
                                prefix="S2 Received",
                            )
                        )
                        s2_first_detect = q

                    if (
                        s2_first_detect is not None
                        and q > s2_first_detect + 1e-9
                        and not s2_slew_logged
                    ):
                        log_lines.append("S2 Body slewing toward lock")
                        s2_slew_logged = True

                    q += q_step

                if q_end >= q_max_local - 1e-12:
                    while q < q_max_local - 1e-12:
                        result._step_phase1(q)
                        self._count_replay_step()
                        q += q_step
                    result.boresight_end = (
                        result.s2.body.beam_boresight_inertial().copy()
                    )
                    center = "locked" if s2.has_seen_beam else "initial_aim"
                    result.s2.phase2_transmitter = build_phase2_transmitter(
                        result.s2,
                        result.s1,
                        result.boresight_end,
                        config,
                    )
                    result._phase2_built = True

                    log_lines.append("S1 Search spiral complete")
                    if s2_first_detect is None:
                        log_lines.append("S2 No beam acquisition")
                    log_lines.append(
                        f"S2 Search spiral started ({center})"
                    )

                    s1 = result.s1.receiver
                    s1_init_deg = np.degrees(s1.initial_pointing_offset)
                    log_lines.append(
                        f"S1 Initial receiver offset: {s1_init_deg:.2f}°"
                    )
                    s1_miss_logged = False
                    s1_first_detect: float | None = None
                    s1_slew_logged = False

                    s1.reset_dish_tracking()
                    q = q_max_local
                    tx2 = result.s2.phase2_transmitter
                    while q <= q_end + 1e-12 and q < total_q - 1e-12:
                        local_q = q - q_max_local
                        had_seen = s1.has_seen_beam
                        dish_boresight = s1.dish_boresight.copy()

                        if not had_seen and not s1_miss_logged:
                            missed = beam_missed_dish_fov_at_q(
                                local_q,
                                tx2.position,
                                s1.dish_mount,
                                dish_boresight,
                                s1.dish_fov,
                                tx2.boresight_at,
                                tx2.alpha,
                                tx2.beam_length,
                            )
                            if missed is not None:
                                log_lines.append(
                                    self._format_rx_angles(
                                        q,
                                        "S1",
                                        s1,
                                        tx2,
                                        dish_boresight,
                                        prefix="S1 Missed beam",
                                    )
                                )
                                s1_miss_logged = True

                        in_cone = beam_hits_dish_at_q(
                            local_q,
                            tx2.position,
                            s1.dish_mount,
                            dish_boresight,
                            s1.dish_fov,
                            tx2.boresight_at,
                            tx2.alpha,
                            tx2.beam_length,
                        )
                        s1.observe_beam(in_cone, tx2.boresight_at(local_q), q_step)
                        self._count_replay_step()

                        if s1.has_seen_beam and not had_seen:
                            log_lines.append(
                                self._format_rx_angles(
                                    q,
                                    "S1",
                                    s1,
                                    tx2,
                                    dish_boresight,
                                    prefix="S1 Received",
                                )
                            )
                            s1_first_detect = q

                        if (
                            s1_first_detect is not None
                            and q > s1_first_detect + 1e-9
                            and not s1_slew_logged
                        ):
                            log_lines.append("S1 Body slewing toward lock")
                            s1_slew_logged = True

                        q += q_step

                    if q_end >= total_q - 1e-12:
                        log_lines.append("S2 Search spiral complete")
                        if s1_first_detect is None:
                            log_lines.append("S1 No beam acquisition")

            self._log_lines = log_lines
            with self._profiler.measure("replay_log_ui"):
                self._log_label.setText("\n".join(log_lines))
                self._position_log_overlay()

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._scene_built:
                return
            self._scene_built = True
            QTimer.singleShot(0, self._init_scene)

        def _init_scene(self) -> None:
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

        def _cone_mesh(self, q: float) -> pv.PolyData:
            tx = result.active_transmitter(q)
            local_q = result.local_q(q)
            verts, faces = tx.cone_mesh_at(
                local_q,
                viz.cone_u_steps,
                viz.cone_v_steps,
            )
            return self._to_polydata(verts, faces)

        def _swept_area_mesh(self, q: float) -> pv.PolyData:
            tx = result.active_transmitter(q)
            rx = result.active_receiver(q)
            local_q = result.local_q(q)
            verts, faces = tx.swept_area_mesh_up_to(
                local_q,
                viz.spiral_trail_steps,
                viz.ribbon_v_steps,
                target_range=rx.dish_range_from_partner(),
            )
            return self._to_polydata(verts, faces)

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
            if self._profiler.enabled:
                self._profile_label.setText(self._profiler.format_overlay())

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
            phase, _ = result.schedule.phase_at(q)

            with self._profiler.measure("mesh_cone"):
                cone_mesh = self._cone_mesh(q)
            with self._profiler.measure("actor_cone"):
                self._update_mesh_actor(
                    cone_mesh,
                    "_cone_poly",
                    "_cone_actor",
                    color="crimson",
                    opacity=0.45,
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
            active_rx = "S2" if phase is SearchPhase.S1_TRANSMIT else "S1"
            with self._profiler.measure("actor_body_color"):
                for sat_name, actor in (
                    ("S1", self._s1_body_actor),
                    ("S2", self._s2_body_actor),
                ):
                    base = "blue" if sat_name == "S1" else "red"
                    color = "limegreen" if in_cone and sat_name == active_rx else base
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

            with self._profiler.measure("mesh_ray_s1"):
                s1_ray = self._dish_boresight_line("S1", q)
            with self._profiler.measure("actor_ray_s1"):
                self._update_line_actor(
                    s1_ray,
                    "_s1_ray_poly",
                    "_s1_ray_actor",
                    color="cyan",
                    line_width=3,
                    label="S1 boresight",
                )
            with self._profiler.measure("mesh_ray_s2"):
                s2_ray = self._dish_boresight_line("S2", q)
            with self._profiler.measure("actor_ray_s2"):
                self._update_line_actor(
                    s2_ray,
                    "_s2_ray_poly",
                    "_s2_ray_actor",
                    color="cyan",
                    line_width=3,
                    label="S2 boresight",
                )

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
