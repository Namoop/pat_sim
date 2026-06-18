"""Environment.toml types and loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SharedSatelliteConfig:
    """Shared spacecraft hardware ([satellite])."""

    body_radius: float
    dish_fov: float
    max_beam_speed: float
    max_fsm_speed: float
    max_fsm_radius: float
    beam_width_mrad: float
    k: float

    @property
    def alpha(self) -> float:
        """Transmitter cone half-angle in radians."""
        return self.beam_width_mrad * 1e-3


@dataclass(frozen=True)
class SimulationConfig:
    distance: float
    t_step: float
    beam_length: float | None
    boresight_extension: float
    max_search_radius: float
    profile_replay: bool
    timeout: float = 100.0
    enforce_speed_limit: bool = True


@dataclass(frozen=True)
class ThreeDVizConfig:
    cone_u_steps: int
    cone_v_steps: int
    spiral_trail_steps: int
    ribbon_v_steps: int
    profile_frames: bool


@dataclass(frozen=True)
class EyeVizConfig:
    axis_limit: float
    profile_frames: bool
    slider_debounce_ms: int


@dataclass(frozen=True)
class MagVizConfig:
    visual_limit_deg: float
    fov_cone_length: float = 1.0
    beam_cone_length: float = 1.0
    bottom_margin: float = 48.0


@dataclass(frozen=True)
class SimulationBundle:
    satellite: SharedSatelliteConfig
    simulation: SimulationConfig
    three_d_viz: ThreeDVizConfig
    eye_viz: EyeVizConfig
    mag_viz: MagVizConfig


def _load_shared_satellite(data: dict) -> SharedSatelliteConfig:
    return SharedSatelliteConfig(
        body_radius=float(data.get("body_radius", 0.5)),
        dish_fov=float(data.get("dish_fov", 0.002)) * 1e-3,
        max_beam_speed=float(data.get("max_beam_speed", 0.087)) * 1e-3,
        max_fsm_speed=float(data.get("max_fsm_speed", 1.0)) * 1e-3,
        max_fsm_radius=float(data.get("max_fsm_radius", 1.0)) * 1e-3,
        beam_width_mrad=float(data["beam_width"]),
        k=float(data.get("k", 10.0)),
    )


def _load_simulation_section(data: dict) -> SimulationConfig:
    beam_length = data.get("beam_length")
    if beam_length is not None:
        beam_length = float(beam_length)
    return SimulationConfig(
        distance=float(data["distance"]),
        t_step=float(data["t_step"]),
        beam_length=beam_length,
        boresight_extension=float(data.get("boresight_extension", 5.0)),
        max_search_radius=float(data.get("max_search_radius", 0.07)) * 1e-3,
        profile_replay=bool(data.get("profile_replay", False)),
        timeout=float(data.get("timeout", 100.0)),
        enforce_speed_limit=bool(data.get("enforce_speed_limit", True)),
    )


def _load_three_d_viz(data: dict) -> ThreeDVizConfig:
    return ThreeDVizConfig(
        cone_u_steps=int(data.get("cone_u_steps", 24)),
        cone_v_steps=int(data.get("cone_v_steps", 32)),
        spiral_trail_steps=int(data.get("spiral_trail_steps", 80)),
        ribbon_v_steps=int(data.get("ribbon_v_steps", 4)),
        profile_frames=bool(data.get("profile_frames", False)),
    )


def _load_eye_viz(data: dict) -> EyeVizConfig:
    return EyeVizConfig(
        axis_limit=float(data.get("axis_limit", 0.1)) * 1e-3,
        profile_frames=bool(data.get("profile_frames", False)),
        slider_debounce_ms=int(data.get("slider_debounce_ms", 16)),
    )


def _load_mag_viz(data: dict) -> MagVizConfig:
    return MagVizConfig(
        visual_limit_deg=float(data.get("visual_limit_deg", 25.0)),
        fov_cone_length=float(data.get("fov_cone_length", 1.0)),
        beam_cone_length=float(data.get("beam_cone_length", 1.0)),
        bottom_margin=float(data.get("bottom_margin", 48.0)),
    )


def load_simulation_config(path: str | Path) -> SimulationBundle:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    return SimulationBundle(
        satellite=_load_shared_satellite(data.get("satellite", {})),
        simulation=_load_simulation_section(data.get("simulation", {})),
        three_d_viz=_load_three_d_viz(data.get("3d_viz", {})),
        eye_viz=_load_eye_viz(data.get("eye_viz", {})),
        mag_viz=_load_mag_viz(data.get("mag_viz", {})),
    )
