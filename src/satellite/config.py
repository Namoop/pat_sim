"""Load scenario parameters from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from satellite.math3d import Vec3, as_vec3


@dataclass(frozen=True)
class SatelliteInstanceConfig:
    """Per-spacecraft pose and optical-bench offsets ([s1] / [s2])."""

    position: Vec3
    bench_theta_offset: float
    bench_phi_offset: float


@dataclass(frozen=True)
class SharedSatelliteConfig:
    """Shared spacecraft hardware ([satellite])."""

    body_radius: float
    dish_fov: float
    bench_slew_time: float
    fsm_settle_time: float
    beam_width_mrad: float

    @property
    def alpha(self) -> float:
        """Transmitter cone half-angle in radians."""
        return self.beam_width_mrad * 1e-3


@dataclass(frozen=True)
class SdaConfig:
    k: float
    gamma: float
    beta: float
    omega_r: float
    L_r: float


@dataclass(frozen=True)
class SimulationConfig:
    q_max: float
    q_step: float
    beam_length: float | None
    boresight_extension: float
    profile_replay: bool


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool
    cone_u_steps: int
    cone_v_steps: int
    spiral_trail_steps: int
    ribbon_v_steps: int
    profile_frames: bool


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    s1: SatelliteInstanceConfig
    s2: SatelliteInstanceConfig
    satellite: SharedSatelliteConfig
    sda: SdaConfig
    simulation: SimulationConfig
    visualization: VisualizationConfig


def _vec3_from_list(values: list[float], field: str) -> Vec3:
    if len(values) != 3:
        raise ValueError(f"{field} must have exactly 3 components")
    return as_vec3(values)


def _load_satellite_instance(
    data: dict,
    section: str,
) -> SatelliteInstanceConfig:
    return SatelliteInstanceConfig(
        position=_vec3_from_list(data["position"], f"{section}.position"),
        bench_theta_offset=float(data.get("bench_theta_offset", 0.0)),
        bench_phi_offset=float(data.get("bench_phi_offset", 0.0)),
    )


def load_config(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    scenario = data.get("scenario", {})
    s1 = data.get("s1", {})
    s2 = data.get("s2", {})
    satellite = data.get("satellite", {})
    sda = data.get("sda", {})
    simulation = data.get("simulation", {})
    visualization = data.get("visualization", {})

    beam_length = simulation.get("beam_length")
    if beam_length is not None:
        beam_length = float(beam_length)

    return ScenarioConfig(
        name=str(scenario.get("name", "unnamed")),
        s1=_load_satellite_instance(s1, "s1"),
        s2=_load_satellite_instance(s2, "s2"),
        satellite=SharedSatelliteConfig(
            body_radius=float(satellite.get("body_radius", 0.5)),
            dish_fov=float(satellite.get("dish_fov", 0.002)),
            bench_slew_time=float(
                satellite.get(
                    "bench_slew_time",
                    satellite.get("dish_slew_time", 0.3),
                )
            ),
            fsm_settle_time=float(satellite.get("fsm_settle_time", 0.0)),
            beam_width_mrad=float(satellite["beam_width"]),
        ),
        sda=SdaConfig(
            k=float(sda["k"]),
            gamma=float(sda["gamma"]),
            beta=float(sda["beta"]),
            omega_r=float(sda["omega_r"]),
            L_r=float(sda["L_r"]),
        ),
        simulation=SimulationConfig(
            q_max=float(simulation["q_max"]),
            q_step=float(simulation["q_step"]),
            beam_length=beam_length,
            boresight_extension=float(simulation.get("boresight_extension", 5.0)),
            profile_replay=bool(simulation.get("profile_replay", False)),
        ),
        visualization=VisualizationConfig(
            enabled=bool(visualization.get("enabled", False)),
            cone_u_steps=int(visualization.get("cone_u_steps", 24)),
            cone_v_steps=int(visualization.get("cone_v_steps", 32)),
            spiral_trail_steps=int(visualization.get("spiral_trail_steps", 80)),
            ribbon_v_steps=int(visualization.get("ribbon_v_steps", 4)),
            profile_frames=bool(visualization.get("profile_frames", False)),
        ),
    )


def default_beam_length(
    position: Vec3,
    partner: Vec3,
    simulation: SimulationConfig,
) -> float:
    """Default TX cone length: link range plus boresight extension."""
    from satellite.math3d import distance

    if simulation.beam_length is not None:
        return simulation.beam_length
    return distance(position, partner) + simulation.boresight_extension
