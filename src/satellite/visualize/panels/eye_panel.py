"""Angular eye view panel (no playback controls)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from satellite.mapviz.panel_widget import AngularMapPanel
from satellite.mapviz.scene import MapScene, build_scene
from satellite.sim.scenario import ScenarioResult
from satellite.visualize.diagnostics import FrameProfiler


@dataclass(frozen=True)
class EyeFrameInfo:
    capture_active: bool
    event_log: tuple[str, ...]


class EyePanel:
    """Embedded angular eye view; caller owns timeline scrubbing."""

    def __init__(self, parent) -> None:
        from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

        self._result: ScenarioResult | None = None

        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_host = QWidget(self._widget)
        canvas_host_layout = QHBoxLayout(self._canvas_host)
        canvas_host_layout.setContentsMargins(0, 0, 0, 0)

        self._panel_s1 = AngularMapPanel(axis_limit=1.0)
        self._panel_s2 = AngularMapPanel(axis_limit=1.0)
        canvas_host_layout.addWidget(self._panel_s1, stretch=1)
        canvas_host_layout.addWidget(self._panel_s2, stretch=1)

        layout.addWidget(self._canvas_host, stretch=1)

        self._profiler = FrameProfiler.from_env(False)
        self._sim_profile_enabled = False
        self._profiling_active = False
        self._replay_step_count = 0
        self._profile_callback = None
        self._initialized = False

    @property
    def widget(self):
        return self._widget

    def set_profile_callback(self, callback) -> None:
        self._profile_callback = callback

    @property
    def profiling_active(self) -> bool:
        return self._profiling_active

    def set_result(self, result: ScenarioResult) -> None:
        self._result = result
        eye_cfg = result.config.eye_viz
        self._panel_s1.set_axis_limit(eye_cfg.axis_limit)
        self._panel_s2.set_axis_limit(eye_cfg.axis_limit)
        self._profiler = FrameProfiler.from_env(eye_cfg.profile_frames)
        self._sim_profile_enabled = result.config.simulation.profile_replay
        self._profiling_active = (
            self._profiler.enabled or self._sim_profile_enabled
        )
        self._initialized = False

    def ensure_initialized(self) -> None:
        if self._initialized or self._result is None:
            return
        self._initialized = True
        self._result.ensure_replay_timeline()

    def apply_t(self, t: float) -> EyeFrameInfo:
        result = self._result
        if result is None:
            raise RuntimeError("EyePanel.set_result must be called first")
        self.ensure_initialized()

        total_t = result.playable_t_end
        t = float(np.clip(t, 0.0, total_t))

        if self._profiling_active:
            frame_start = time.perf_counter()
            with self._profiler.measure("replay"):
                event_log = self._replay_to(t)
            self._profiler.set_gauge(
                "replay_steps",
                float(self._replay_step_count),
            )
            with self._profiler.measure("build_scene"):
                scene = build_scene(result, t)
        else:
            event_log = self._replay_to(t)
            scene = build_scene(result, t)

        if self._profiling_active:
            t_paint = time.perf_counter()
            self._update_panels(scene)
            self._profiler.record("paint", time.perf_counter() - t_paint)
            self._profiler.record(
                "frame_total",
                time.perf_counter() - frame_start,
            )
            self._profiler.end_frame()
            self._emit_profile()
        else:
            self._update_panels(scene)

        return EyeFrameInfo(
            capture_active=scene.capture_active,
            event_log=tuple(event_log),
        )

    def close_panel(self) -> None:
        pass

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

    def _replay_to(self, t_end: float) -> list[str]:
        result = self._result
        assert result is not None
        log_lines: list[str] = []
        result.replay_to(t_end, event_log=log_lines)
        if self._profiling_active and result.last_sim_profiler is not None:
            self._replay_step_count = result.last_sim_profiler.step_count
        return log_lines

    def _update_panels(self, scene: MapScene) -> None:
        if self._panel_s1.set_panel(scene.s1, partner_label="S2"):
            self._panel_s1.repaint()
        if self._panel_s2.set_panel(scene.s2, partner_label="S1"):
            self._panel_s2.repaint()
