"""PyVista 3D view panel (no playback controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtWidgets import QWidget

from satellite.math.geometry import axis_perpendicular_basis
from satellite.math.math3d import normalize
from scenario.run import ScenarioResult
from visualize.viz_geometry import cone_mesh_for_aim
from visualize.viz_squish import SquishContext
from visualize.diagnostics import FrameProfiler


@dataclass(frozen=True)
class ThreeDFrameInfo:
    capture_active: bool
    event_log: tuple[str, ...]


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


_OVERLAY_BTN_STYLE = (
    "QPushButton {"
    "  background-color: #f5f5f7;"
    "  border: 1px solid #d2d2d7;"
    "  border-radius: 4px;"
    "  padding: 4px 12px;"
    "  color: #1d1d1f;"
    "  font-family: 'Sans';"
    "  font-size: 11px;"
    "  font-weight: bold;"
    "}"
    "QPushButton:hover {"
    "  background-color: #e8e8ed;"
    "}"
    "QPushButton:checked {"
    "  background-color: #0071e3;"
    "  color: white;"
    "  border-color: #0071e3;"
    "}"
)


class _PlotOverlayHost(QWidget):
    """Plot host that repositions floating camera controls on resize."""

    def __init__(self, panel: "ThreeDPanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._panel._position_overlay()


class ThreeDPanel:
    """Embedded 3D PyVista view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        import pyvista as pv
        from PyQt6.QtWidgets import (
            QButtonGroup,
            QGridLayout,
            QPushButton,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )
        from pyvistaqt import QtInteractor

        self._pv = pv
        self._result: ScenarioResult | None = None
        self._current_t = 0.0
        self._active_camera_preset = "s1_close"
        self._pending_camera_restore: dict | None = None

        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_host = _PlotOverlayHost(self, self._widget)
        plot_host_layout = QVBoxLayout(self._plot_host)
        plot_host_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self._plot_host)
        plot_host_layout.addWidget(self.plotter.interactor, stretch=1)

        self._btn_bar = QWidget(self._plot_host)
        self._btn_bar.setAutoFillBackground(True)
        self._btn_bar.setStyleSheet("background-color: #ffffff;")
        btn_bar_layout = QGridLayout(self._btn_bar)
        btn_bar_layout.setContentsMargins(0, 0, 0, 0)
        btn_bar_layout.setHorizontalSpacing(6)
        btn_bar_layout.setVerticalSpacing(6)

        self._camera_btn_group = QButtonGroup(self._btn_bar)
        self._camera_btn_group.setExclusive(True)
        self._camera_preset_buttons: dict[str, QPushButton] = {}

        def _add_camera_button(
            key: str,
            label: str,
            row: int,
            col: int,
            *,
            row_span: int = 1,
            col_span: int = 1,
            tooltip: str,
        ) -> None:
            btn = QPushButton(label, self._btn_bar)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(_OVERLAY_BTN_STYLE)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            self._camera_btn_group.addButton(btn)
            self._camera_preset_buttons[key] = btn
            btn.clicked.connect(lambda _checked=False, k=key: self._on_camera_button(k))
            btn_bar_layout.addWidget(btn, row, col, row_span, col_span)

        _add_camera_button(
            "center",
            "Center",
            0,
            0,
            col_span=2,
            tooltip="Frame both satellites from the midpoint between them",
        )
        _add_camera_button(
            "s1_close",
            "S1 close",
            1,
            0,
            tooltip="Jump to S1 close camera preset",
        )
        _add_camera_button(
            "s1_far",
            "S1 far",
            1,
            1,
            tooltip="Jump to S1 far camera preset",
        )
        _add_camera_button(
            "s2_close",
            "S2 close",
            2,
            0,
            tooltip="Jump to S2 close camera preset",
        )
        _add_camera_button(
            "s2_far",
            "S2 far",
            2,
            1,
            tooltip="Jump to S2 far camera preset",
        )
        for col in range(2):
            btn_bar_layout.setColumnStretch(col, 1)

        self._camera_preset_buttons["s1_close"].setChecked(True)
        self._position_overlay()

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

    @property
    def widget(self):
        return self._widget

    def _position_overlay(self) -> None:
        margin = 12
        self._btn_bar.adjustSize()
        bar_w = self._btn_bar.sizeHint().width()
        bar_h = self._btn_bar.sizeHint().height()
        host = self._plot_host
        self._btn_bar.setGeometry(
            host.width() - margin - bar_w,
            margin,
            bar_w,
            bar_h,
        )
        if self._btn_bar.isVisible():
            self._btn_bar.raise_()

    def set_profile_callback(self, callback) -> None:
        self._profile_callback = callback

    @property
    def profiling_active(self) -> bool:
        return self._profiler.enabled or self._sim_profile_enabled

    def _capture_camera_state(self) -> dict | None:
        if not self._scene_built or self.plotter is None:
            return None
        cam = self.plotter.camera
        state = {
            "focal_point": tuple(cam.focal_point),
            "position": tuple(cam.position),
            "up": tuple(cam.up),
            "view_angle": float(cam.view_angle),
            "clipping_range": tuple(cam.clipping_range),
            "parallel_scale": float(cam.parallel_scale),
        }
        return state

    def _restore_camera_state(self, state: dict) -> None:
        if not self._scene_built or self.plotter is None:
            return
        cam = self.plotter.camera
        cam.focal_point = state["focal_point"]
        cam.position = state["position"]
        cam.up = state["up"]
        cam.view_angle = state["view_angle"]
        cam.clipping_range = state["clipping_range"]
        cam.parallel_scale = state["parallel_scale"]
        self.plotter.render()

    def set_result(self, result: ScenarioResult) -> None:
        self._pending_camera_restore = self._capture_camera_state()
        self._result = result
        config = result.config
        viz = config.three_d_viz
        self._profiler = FrameProfiler.from_env(viz.profile_frames)
        self._sim_profile_enabled = config.simulation.profile_replay
        self._scene_built = False
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

    def apply_t(self, t: float) -> ThreeDFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("ThreeDPanel.set_result must be called first")
        self.ensure_initialized()

        total_t = result.playable_t_end
        t = float(np.clip(t, 0.0, total_t))
        self._current_t = t
        frame_start = time.perf_counter()

        with self._profiler.measure("replay"):
            event_log = self._replay_to(t)
        self._profiler.set_gauge("replay_steps", float(self._replay_step_count))

        with self._profiler.measure("update_scene"):
            self._update_scene()

        self._profiler.record("frame_total", time.perf_counter() - frame_start)
        self._profiler.end_frame()
        self._emit_profile()

        return ThreeDFrameInfo(
            capture_active=result.mutual_lock(t),
            event_log=tuple(event_log),
        )

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

    def _squish_ctx(self) -> SquishContext:
        result = self._result
        assert result is not None
        squish = result.config.three_d_viz.squish
        return SquishContext.from_satellites(result.p1, result.pt, squish)

    def _focal_position(self, satellite: str) -> np.ndarray:
        result = self._result
        assert result is not None
        ctx = self._squish_ctx()
        if satellite == "S1":
            return ctx.position(result.p1)
        return ctx.position(result.pt)

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
        result = self._result
        assert result is not None
        preset = _CAMERA_PRESETS[preset_key]
        ctx = self._squish_ctx()
        focal = self._focal_position(preset.focal_satellite)
        raw_focal = result.p1 if preset.focal_satellite == "S1" else result.pt
        raw_cam = np.asarray(raw_focal, dtype=np.float64) + np.asarray(
            preset.position_offset, dtype=np.float64
        )
        position = ctx.position(raw_cam)
        self._apply_camera_pose(
            focal_point=focal,
            position=position,
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

    def _apply_camera_for_preset(self, key: str) -> None:
        if key == "center":
            self._apply_center_camera()
        else:
            self._apply_camera_preset(key)

    def _on_camera_button(self, key: str) -> None:
        self._active_camera_preset = key
        self._apply_camera_for_preset(key)

    def _replay_to(self, t_end: float) -> list[str]:
        result = self._result
        assert result is not None
        log_lines: list[str] = []
        result.replay_to(t_end, event_log=log_lines)
        if result.last_sim_profiler is not None:
            self._replay_step_count = result.last_sim_profiler.step_count
        return log_lines

    def _build_scene(self) -> None:
        result = self._result
        assert result is not None
        pv = self._pv
        config = result.config
        body_radius = config.satellite.body_radius
        ctx = self._squish_ctx()

        p = self.plotter
        p.set_background("white")
        p.add_axes()

        self._s1_body_actor = p.add_mesh(
            pv.Sphere(radius=body_radius, center=ctx.position(result.p1)),
            color="blue",
            label="S1",
        )
        self._s2_body_actor = p.add_mesh(
            pv.Sphere(radius=body_radius, center=ctx.position(result.pt)),
            color="red",
            label="S2",
        )

        for sat, color, label in (
            (result.s1, "lightgray", "S1 believed aim"),
            (result.s2, "silver", "S2 believed aim"),
        ):
            ray_len = ctx.length(result.boresight_ray_length(sat.name))
            apex = ctx.position(sat.position)
            aim = ctx.direction(sat.believed_boresight, toward_partner=sat.bench.toward_partner)
            end = apex + aim * ray_len
            p.add_mesh(
                pv.Line(apex, end),
                color=color,
                line_width=1,
                opacity=0.5,
                label=label,
            )

        if self._pending_camera_restore is not None:
            self._restore_camera_state(self._pending_camera_restore)
            self._pending_camera_restore = None
        else:
            self._apply_camera_for_preset(self._active_camera_preset)

    def _to_polydata(self, verts: np.ndarray, faces: np.ndarray):
        pv = self._pv
        if len(verts) == 0 or len(faces) == 0:
            return pv.PolyData()
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
        )
        return pv.PolyData(verts, faces_pv)

    def _empty_mesh(self):
        return self._pv.PolyData()

    def _hardware_state(self, satellite: str, t: float) -> tuple[bool, bool]:
        result = self._result
        assert result is not None
        scheduled, local_t = result.schedule.script_at(t)
        timeline = scheduled.script.s1 if satellite == "S1" else scheduled.script.s2
        return timeline.hardware_state_at(local_t)

    def _cone_mesh(self, satellite: str, t: float):
        beam_enabled, _ = self._hardware_state(satellite, t)
        if not beam_enabled:
            return self._empty_mesh()
        result = self._result
        assert result is not None
        config = result.config
        viz = config.three_d_viz
        ctx = self._squish_ctx()
        sat = result.s1 if satellite == "S1" else result.s2
        raw_aim = result.bench_aim(satellite, t)
        aim = ctx.direction(raw_aim, toward_partner=sat.bench.toward_partner)
        beam_len = ctx.length(result.beam_length(satellite))
        verts, faces = cone_mesh_for_aim(
            ctx.position(sat.position),
            aim,
            ctx.angle(config.satellite.alpha),
            beam_len,
            viz.cone_u_steps,
            viz.cone_v_steps,
        )
        return self._to_polydata(verts, faces)

    def _fov_circle(self, satellite: str, t: float):
        _, receiver_enabled = self._hardware_state(satellite, t)
        if not receiver_enabled:
            return self._empty_mesh()
        result = self._result
        assert result is not None
        config = result.config
        viz = config.three_d_viz
        ctx = self._squish_ctx()
        sat = result.s1 if satellite == "S1" else result.s2
        dish_fov = ctx.angle(sat.receiver.dish_fov)
        pv = self._pv
        rx = sat.receiver
        raw_aim = result.bench_aim(satellite, t)
        aim = ctx.direction(raw_aim, toward_partner=sat.bench.toward_partner)
        mount = ctx.position(rx.dish_mount_for_boresight(raw_aim))
        axis = normalize(aim)
        partner = ctx.position(sat.partner_actual)
        dist_along_aim = float(np.dot(partner - mount, axis))
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

    def _swept_area_mesh(self, t: float):
        return self._pv.PolyData()

    def _dish_mesh(self, satellite: str, t: float):
        result = self._result
        assert result is not None
        viz = result.config.three_d_viz
        ctx = self._squish_ctx()
        rx = result.s1.receiver if satellite == "S1" else result.s2.receiver
        boresight = result.dish_boresight_for_display(satellite, t)
        verts, faces = rx.dish_mesh_at(viz.cone_v_steps, boresight=boresight)
        verts = ctx.positions(verts)
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

        if line.n_points == 0:
            if actor is not None:
                actor.SetVisibility(0)
            return

        if poly is None or actor is None:
            setattr(self, poly_attr, line)
            kwargs: dict = {"color": color, "line_width": line_width}
            if label is not None:
                kwargs["label"] = label
            actor = self.plotter.add_mesh(line, **kwargs)
            setattr(self, actor_attr, actor)
            return

        actor.SetVisibility(1)
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
        t = self._current_t

        with self._profiler.measure("mesh_cone_s1"):
            s1_cone = self._cone_mesh("S1", t)
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
            s2_cone = self._cone_mesh("S2", t)
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
            s1_fov = self._fov_circle("S1", t)
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
            s2_fov = self._fov_circle("S2", t)
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
            swept_mesh = self._swept_area_mesh(t)
        with self._profiler.measure("actor_swept"):
            self._update_mesh_actor(
                swept_mesh,
                "_swept_poly",
                "_swept_actor",
                color="orange",
                opacity=0.55,
                label="swept area",
            )

        with self._profiler.measure("mutual_lock"):
            in_cone = result.mutual_lock(t)
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
            s1_dish = self._dish_mesh("S1", t)
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
            s2_dish = self._dish_mesh("S2", t)
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
            if self._btn_bar.isVisible():
                self._position_overlay()

    def on_tab_shown(self) -> None:
        if self._btn_bar.isVisible():
            self._position_overlay()
        if self._scene_built and self.plotter is not None:
            self.plotter.render()

    def capture_record_image(self) -> np.ndarray:
        """Capture the rendered 3D view (Qt grab cannot read the GL framebuffer)."""
        from visualize.record import ensure_rgb_uint8

        if not self._scene_built or self.plotter is None:
            raise RuntimeError("ThreeDPanel must be initialized before capture")

        self.plotter.render()
        rgb = self.plotter.screenshot(return_img=True)
        if rgb is None:
            raise RuntimeError("3D screenshot failed")
        return ensure_rgb_uint8(np.asarray(rgb))

    def set_overlay_visible(self, visible: bool) -> None:
        self._btn_bar.setVisible(visible)
        if visible:
            self._position_overlay()
