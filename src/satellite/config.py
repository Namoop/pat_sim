"""Load scenario parameters from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from satellite.math3d import Vec3, as_vec3


@dataclass(frozen=True)
class PositionsConfig:
    p1: Vec3
    p2: Vec3
    pt: Vec3 | None


@dataclass(frozen=True)
class OffsetsConfig:
    theta_jumble: float
    phi_jumble: float


@dataclass(frozen=True)
class SdaConfig:
    k: float
    alpha: float
    gamma: float
    beta: float
    omega_r: float
    L_r: float


@dataclass(frozen=True)
class SimulationConfig:
    q_max: float
    q_step: float
    beam_length: float | None


@dataclass(frozen=True)
class ReceiverConfig:
    dish_theta_offset: float
    dish_phi_offset: float
    body_radius: float
    dish_radius: float
    dish_depth: float
    dish_slew_time: float


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool
    cone_u_steps: int
    cone_v_steps: int
    spiral_trail_steps: int
    ribbon_v_steps: int


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    positions: PositionsConfig
    offsets: OffsetsConfig
    sda: SdaConfig
    simulation: SimulationConfig
    receiver: ReceiverConfig
    visualization: VisualizationConfig


def _vec3_from_list(values: list[float], field: str) -> Vec3:
    if len(values) != 3:
        raise ValueError(f"{field} must have exactly 3 components")
    return as_vec3(values)


def load_config(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    scenario = data.get("scenario", {})
    positions = data.get("positions", {})
    offsets = data.get("offsets", {})
    sda = data.get("sda", {})
    simulation = data.get("simulation", {})
    receiver = data.get("receiver", {})
    visualization = data.get("visualization", {})

    pt_raw = positions.get("pt")
    pt = _vec3_from_list(pt_raw, "positions.pt") if pt_raw is not None else None

    beam_length = simulation.get("beam_length")
    if beam_length is not None:
        beam_length = float(beam_length)

    return ScenarioConfig(
        name=str(scenario.get("name", "unnamed")),
        positions=PositionsConfig(
            p1=_vec3_from_list(positions["p1"], "positions.p1"),
            p2=_vec3_from_list(positions["p2"], "positions.p2"),
            pt=pt,
        ),
        offsets=OffsetsConfig(
            theta_jumble=float(offsets.get("theta_jumble", 0.0)),
            phi_jumble=float(offsets.get("phi_jumble", 0.0)),
        ),
        sda=SdaConfig(
            k=float(sda["k"]),
            alpha=float(sda["alpha"]),
            gamma=float(sda["gamma"]),
            beta=float(sda["beta"]),
            omega_r=float(sda["omega_r"]),
            L_r=float(sda["L_r"]),
        ),
        simulation=SimulationConfig(
            q_max=float(simulation["q_max"]),
            q_step=float(simulation["q_step"]),
            beam_length=beam_length,
        ),
        receiver=ReceiverConfig(
            dish_theta_offset=float(receiver.get("dish_theta_offset", 0.08)),
            dish_phi_offset=float(receiver.get("dish_phi_offset", 0.06)),
            body_radius=float(receiver.get("body_radius", 0.1)),
            dish_radius=float(receiver.get("dish_radius", 0.08)),
            dish_depth=float(receiver.get("dish_depth", 0.04)),
            dish_slew_time=float(receiver.get("dish_slew_time", 0.3)),
        ),
        visualization=VisualizationConfig(
            enabled=bool(visualization.get("enabled", False)),
            cone_u_steps=int(visualization.get("cone_u_steps", 24)),
            cone_v_steps=int(visualization.get("cone_v_steps", 32)),
            spiral_trail_steps=int(visualization.get("spiral_trail_steps", 80)),
            ribbon_v_steps=int(visualization.get("ribbon_v_steps", 4)),
        ),
    )


def resolve_actual_position(config: ScenarioConfig) -> Vec3:
    """Return P_t, computing from jumble offsets when not explicit in TOML."""
    if config.positions.pt is not None:
        return config.positions.pt
    from satellite.geometry import actual_position_from_jumble

    return actual_position_from_jumble(
        config.positions.p1,
        config.positions.p2,
        config.offsets.theta_jumble,
        config.offsets.phi_jumble,
    )
