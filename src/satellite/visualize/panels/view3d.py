"""PyVista 3D view panel (no playback controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from satellite.geometry import axis_perpendicular_basis
from satellite.math3d import normalize
from satellite.scenario import ScenarioResult
from satellite.viz_geometry import cone_mesh_for_aim
from satellite.visualize.diagnostics import FrameProfiler


@dataclass(frozen=True)
class View3DFrameInfo:
    phase_label: str


@dataclass(frozen=True)
class CenterCameraPose:
    focal_point: tuple[float, float, float]
    position: tuple[float, float, float]
    up: tuple[float, float, float]
    view_angle: float = 30.0


def center_camera_pose(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    body_radius: float,
    view_angle: float = 30.0,
    margin: float = 0.78,
) -> CenterCameraPose:
    """Camera pose framing both satellites from the midpoint between them."""
    midpoint = 0.5 * (p1 + p2)
    sep = p2 - p1
    sep_len = float(np.linalg.norm(sep))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if sep_len > 1e-12:
        sep_unit = sep / sep_len
        side = np.cross(sep_unit, world_up)
        side_len = float(np.linalg.norm(side))
        if side_len < 1e-12:
            side = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            side = side / side_len
        view_dir = normalize(-side + 0.35 * world_up)
    else:
        view_dir = normalize(np.array([0.0, 1.0, 0.35], dtype=np.float64))

    half_extent = 0.5 * sep_len + body_radius
    fov_rad = np.radians(view_angle)
    cam_distance = margin * half_extent / np.tan(fov_rad / 2.0)
    position = midpoint + view_dir * cam_distance

    return CenterCameraPose(
        focal_point=tuple(midpoint),
        position=tuple(position),
        up=(0.0, 0.0, 1.0),
        view_angle=view_angle,
    )


@dataclass(frozen=True)
class _CameraPreset:
    """Camera pose relative to a focal satellite (partner for S1 views, S1 for S2)."""

    focal_satellite: str
    position_offset: tuple[float, float, float]
    up: tuple[float, float, float]
    view_angle: float = 30.0


# Calibrated at inter-satellite distance 1000 m; offsets are focal-relative.
# S2 presets mirror S1 through 180° about Z, then horizontally (negate y).
_CAMERA_PRESETS: dict[str, _CameraPreset] = {
    "s1_close": _CameraPreset(
        focal_satellite="S2",
        position_offset=(134.33, -15.9782, 1.86303),
        up=(-0.0802431, -0.751044, -0.655358),
    ),
    "s1_far": _CameraPreset(
        focal_satellite="S2",
        position_offset=(-1101.251, 12.9602, 0.0900143),
        up=(0.00865572, 0.740161, -0.672374),
    ),
    "s2_close": _CameraPreset(
        focal_satellite="S1",
        position_offset=(-134.33, -15.9782, -1.86303),
        up=(0.0802431, -0.751044, -0.655358),
    ),
    "s2_far": _CameraPreset(
        focal_satellite="S1",
        position_offset=(1101.251, 12.9602, -0.0900143),
        up=(-0.00865572, 0.740161, -0.672374),
    ),
}


class View3DPanel:
    """Embedded 3D PyVista view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        import pyvista as pv
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import (
            QFrame,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
        from pyvistaqt import QtInteractor

        self._pv = pv
        self._result: ScenarioResult | None = None
        self._current_q = 0.0

        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_host = QWidget(self._widget)
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

        self._camera_preset_buttons: dict[str, QPushButton] = {}
        for key, label in (
            ("center", "Center"),
            ("s1_close", "S1 close"),
            ("s1_far", "S1 far"),
            ("s2_close", "S2 close"),
            ("s2_far", "S2 far"),
        ):
            btn = QPushButton(label, self._plot_host)
            if key == "center":
                btn.setToolTip(
                    "Frame both satellites from the midpoint between them"
                )
            else:
                btn.setToolTip(f"Jump to {label} camera preset")
            btn.clicked.connect(lambda _checked=False, k=key: self._on_camera_button(k))
            self._style_camera_button(btn)
            btn.raise_()
            self._camera_preset_buttons[key] = btn

        layout.addWidget(self._plot_host, stretch=1)

        self._cone_poly = None
        self._cone_actor = None
        self._s2_cone_poly = None
        self._s2_cone_actor = None
        self._s1_fov_poly = None
        self._s1_fov_actor = None
        self._s2_fov_poly = None
        self._s2_fov_actor = None
        self._swept_poly = None
        self._swept_actor = None
        self._s1_body_actor = None
        self._s2_body_actor = None
        self._s1_dish_poly = None
        self._s1_dish_actor = None
        self._s2_dish_poly = None
        self._s2_dish_actor = None
        self._scene_built = False
        self._profiler = FrameProfiler.from_env(False)
        self._sim_profile_enabled = False
        self._replay_step_count = 0
        self._profile_callback = None

        from PyQt6.QtCore import QObject, QEvent

        class _ResizeForwarder(QObject):
            def __init__(self, panel: View3DPanel) -> None:
                super().__init__()
                self._panel = panel

            def eventFilter(self, obj, event):  # noqa: N802
                if event.type() == QEvent.Type.Resize:
                    self._panel._position_overlays()
                return False

        self._resize_forwarder = _ResizeForwarder(self)
        self._widget.installEventFilter(self._resize_forwarder)

    @property
    def widget(self):
        return self._widget

    @staticmethod
    def _style_camera_button(btn) -> None:
        """Square opaque buttons — avoids dark parent bleeding at rounded corners over VTK."""
        btn.setAutoFillBackground(False)
        btn.setFlat(True)
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgb(55, 55, 68);"
            "  color: #f0f0f8;"
            "  border: 1px solid rgb(140, 140, 160);"
            "  border-radius: 0px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgb(70, 70, 85);"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgb(40, 40, 52);"
            "}"
        )

    def set_profile_callback(self, callback) -> None:
        self._profile_callback = callback

    @property
    def profiling_active(self) -> bool:
        return self._profiler.enabled or self._sim_profile_enabled

    def set_result(self, result: ScenarioResult) -> None:
        self._result = result
        config = result.config
        viz = config.visualization
        self._profiler = FrameProfiler.from_env(viz.profile_frames)
        self._sim_profile_enabled = config.simulation.profile_replay
        self._scene_built = False
        self._clear_dynamic_actors()
        if self.plotter is not None:
            self.plotter.clear()
        self._cone_poly = None
        self._cone_actor = None
        self._s2_cone_poly = None
        self._s2_cone_actor = None
        self._s1_fov_poly = None
        self._s1_fov_actor = None
        self._s2_fov_poly = None
        self._s2_fov_actor = None
        self._swept_poly = None
        self._swept_actor = None
        self._s1_body_actor = None
        self._s2_body_actor = None
        self._s1_dish_poly = None
        self._s1_dish_actor = None
        self._s2_dish_poly = None
        self._s2_dish_actor = None

    def ensure_initialized(self) -> None:
        if self._scene_built or self._result is None:
            return
        self._scene_built = True
        timeline = self._result.ensure_replay_timeline()
        print(f"Replay timeline: {timeline.memory_summary()}")
        self._build_scene()

    def apply_q(self, q: float) -> View3DFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("View3DPanel.set_result must be called first")
        self.ensure_initialized()

        total_q = result.schedule.total_duration
        q = float(np.clip(q, 0.0, total_q))
        self._current_q = q
        phase_str = result.epoch_label(q)
        frame_start = time.perf_counter()

        with self._profiler.measure("replay"):
            self._replay_to(q)
        self._profiler.set_gauge("replay_steps", float(self._replay_step_count))

        with self._profiler.measure("update_scene"):
            self._update_scene()

        self._profiler.record("frame_total", time.perf_counter() - frame_start)
        self._profiler.end_frame()
        self._emit_profile()

        return View3DFrameInfo(phase_label=phase_str)

    def close_panel(self) -> None:
        if self.plotter is not None:
            self.plotter.close()

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

    def _focal_position(self, satellite: str) -> np.ndarray:
        result = self._result
        assert result is not None
        if satellite == "S1":
            return np.asarray(result.p1, dtype=np.float64)
        return np.asarray(result.pt, dtype=np.float64)

    def _apply_camera_pose(
        self,
        *,
        focal_point: np.ndarray,
        position: np.ndarray,
        up: np.ndarray,
        view_angle: float,
    ) -> None:
        if not self._scene_built or self.plotter is None:
            return
        cam = self.plotter.camera
        cam.focal_point = focal_point
        cam.position = position
        cam.up = up
        cam.view_angle = view_angle
        self.plotter.render()

    def _apply_camera_preset(self, preset_key: str) -> None:
        preset = _CAMERA_PRESETS[preset_key]
        focal = self._focal_position(preset.focal_satellite)
        self._apply_camera_pose(
            focal_point=focal,
            position=focal + np.asarray(preset.position_offset, dtype=np.float64),
            up=np.asarray(preset.up, dtype=np.float64),
            view_angle=preset.view_angle,
        )

    def _apply_center_camera(self) -> None:
        result = self._result
        if result is None:
            return
        pose = center_camera_pose(
            self._focal_position("S1"),
            self._focal_position("S2"),
            body_radius=result.config.satellite.body_radius,
        )
        self._apply_camera_pose(
            focal_point=np.asarray(pose.focal_point, dtype=np.float64),
            position=np.asarray(pose.position, dtype=np.float64),
            up=np.asarray(pose.up, dtype=np.float64),
            view_angle=pose.view_angle,
        )

    def _on_camera_button(self, key: str) -> None:
        if key == "center":
            self._apply_center_camera()
        else:
            self._apply_camera_preset(key)

    def _position_overlays(self) -> None:
        self._position_log_overlay()
        self._position_camera_overlay()

    def _position_camera_overlay(self) -> None:
        margin = 12
        gap = 6
        x = margin
        y = margin
        for key in ("center", "s1_close", "s1_far", "s2_close", "s2_far"):
            btn = self._camera_preset_buttons[key]
            btn.adjustSize()
            btn.setGeometry(x, y, btn.sizeHint().width(), btn.sizeHint().height())
            btn.raise_()
            x += btn.width() + gap

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

    def _replay_to(self, q_end: float) -> None:
        result = self._result
        assert result is not None
        log_lines: list[str] = []
        result.replay_to(q_end, event_log=log_lines)
        self._log_label.setText("\n".join(log_lines))
        self._position_overlays()
        if result.last_sim_profiler is not None:
            self._replay_step_count = result.last_sim_profiler.step_count

    def _clear_dynamic_actors(self) -> None:
        pass

    def _build_scene(self) -> None:
        result = self._result
        assert result is not None
        pv = self._pv
        config = result.config
        body_radius = config.satellite.body_radius

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

        self._apply_center_camera()

    def _to_polydata(self, verts: np.ndarray, faces: np.ndarray):
        pv = self._pv
        if len(verts) == 0 or len(faces) == 0:
            return pv.PolyData()
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
        )
        return pv.PolyData(verts, faces_pv)

    def _cone_mesh(self, satellite: str, q: float):
        result = self._result
        assert result is not None
        config = result.config
        viz = config.visualization
        sat = result.s1 if satellite == "S1" else result.s2
        aim = result.bench_aim(satellite, q)
        beam_len = result.beam_length(satellite)
        verts, faces = cone_mesh_for_aim(
            sat.position,
            aim,
            config.satellite.alpha,
            beam_len,
            viz.cone_u_steps,
            viz.cone_v_steps,
        )
        return self._to_polydata(verts, faces)

    def _fov_circle(self, satellite: str, q: float):
        result = self._result
        assert result is not None
        config = result.config
        viz = config.visualization
        dish_fov = config.satellite.dish_fov
        pv = self._pv
        sat = result.s1 if satellite == "S1" else result.s2
        rx = sat.receiver
        aim = result.bench_aim(satellite, q)
        mount = rx.dish_mount_for_boresight(aim)
        axis = normalize(aim)
        dist_along_aim = float(np.dot(sat.partner_actual - mount, axis))
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

    def _swept_area_mesh(self, q: float):
        return self._pv.PolyData()

    def _dish_mesh(self, satellite: str, q: float):
        result = self._result
        assert result is not None
        viz = result.config.visualization
        rx = result.s1.receiver if satellite == "S1" else result.s2.receiver
        boresight = result.dish_boresight_for_display(satellite, q)
        verts, faces = rx.dish_mesh_at(viz.cone_v_steps, boresight=boresight)
        return self._to_polydata(verts, faces)

    def _update_line_actor(
        self,
        line,
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

    def _update_mesh_actor(
        self,
        mesh,
        poly_attr: str,
        actor_attr: str,
        *,
        color: str,
        opacity: float,
        label: str | None = None,
    ) -> None:
        pv = self._pv
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
        result = self._result
        assert result is not None
        pv = self._pv
        q = self._current_q

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

        if self._widget.isVisible():
            with self._profiler.measure("render"):
                self.plotter.render()

    def on_tab_shown(self) -> None:
        if self._scene_built and self.plotter is not None:
            self.plotter.render()
