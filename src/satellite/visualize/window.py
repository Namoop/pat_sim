"""PyVista + Qt visualization window."""

from __future__ import annotations

import numpy as np

from satellite.scenario import ScenarioResult


def _configure_qt_platform() -> None:
    """VTK embedded in Qt needs X11 on many Linux Wayland sessions."""
    import os
    import sys

    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def run_visualizer(result: ScenarioResult, start_q: float = 0.0) -> None:
    """Open interactive 3D window. Imports pyvista/Qt lazily."""
    _configure_qt_platform()

    import pyvista as pv
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
            self._cone_actor = None
            self._swept_actor = None
            self._receiver_actor = None
            self._scene_built = False

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            self.plotter = QtInteractor(central)
            layout.addWidget(self.plotter.interactor, stretch=1)

            controls = QHBoxLayout()
            self.time_label = QLabel()
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(int(q_max / q_step))
            self.slider.setValue(int(self.current_q / q_step))
            self.slider.valueChanged.connect(self._on_slider_changed)

            self.next_btn = QPushButton("Next")
            self.next_btn.clicked.connect(self._on_next)

            controls.addWidget(QLabel("Time q:"))
            controls.addWidget(self.slider, stretch=1)
            controls.addWidget(self.time_label)
            controls.addWidget(self.next_btn)
            layout.addLayout(controls)

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if self._scene_built:
                return
            self._scene_built = True
            # Defer VTK's first render until the native window exists (Wayland/X11).
            QTimer.singleShot(0, self._init_scene)

        def _init_scene(self) -> None:
            self._build_scene()
            self._update_time(self.current_q)

        def _build_scene(self) -> None:
            p = self.plotter

            p.set_background("white")
            p.add_axes()

            p.add_mesh(
                pv.Sphere(radius=0.08, center=result.p1),
                color="blue",
                label="transmitter",
            )
            p.add_mesh(
                pv.Sphere(radius=0.06, center=result.p2),
                color="gray",
                opacity=0.5,
                label="believed target",
            )

            self._receiver_actor = p.add_mesh(
                pv.Sphere(radius=0.1, center=result.pt),
                color="red",
                label="receiver (actual)",
            )

            p.add_mesh(
                pv.Line(result.p1, result.p2),
                color="lightgray",
                line_width=1,
            )
            p.reset_camera()

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
            )
            return self._to_polydata(verts, faces)

        def _update_time(self, q: float) -> None:
            self.current_q = float(np.clip(q, 0.0, q_max))
            self.time_label.setText(f"{self.current_q:.3f} / {q_max:.3f}")

            if self._cone_actor is not None:
                self.plotter.remove_actor(self._cone_actor)
            if self._swept_actor is not None:
                self.plotter.remove_actor(self._swept_actor)

            self._cone_actor = self.plotter.add_mesh(
                self._cone_mesh(self.current_q),
                color="crimson",
                opacity=0.45,
                show_edges=False,
            )

            swept = self._swept_area_mesh(self.current_q)
            if swept.n_points > 0:
                self._swept_actor = self.plotter.add_mesh(
                    swept,
                    color="orange",
                    opacity=0.55,
                    show_edges=False,
                    label="swept area",
                )

            in_cone = result.in_cone_at_q(self.current_q)
            color = "limegreen" if in_cone else "red"
            prop = self._receiver_actor.GetProperty()
            prop.SetColor(*pv.Color(color).float_rgb)

            if self.isVisible():
                self.plotter.render()

        def _on_slider_changed(self, value: int) -> None:
            self._update_time(value * q_step)

        def _on_next(self) -> None:
            new_q = min(self.current_q + q_step, q_max)
            self.slider.setValue(int(new_q / q_step))
            self._update_time(new_q)

        def closeEvent(self, event) -> None:  # noqa: N802
            self.plotter.close()
            super().closeEvent(event)

    window = SatelliteWindow()
    window.show()
    app.exec()
