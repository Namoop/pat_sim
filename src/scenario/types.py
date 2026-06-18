"""Single-scenario runtime types and assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from config import apply_dict_overrides
from satellite.config import (
    EyeVizConfig,
    MagVizConfig,
    SharedSatelliteConfig,
    SimulationBundle,
    SimulationConfig,
    ThreeDVizConfig,
)
from satellite.math.math3d import Vec3, as_vec3, distance

if TYPE_CHECKING:
    from strategy.config import StrategyConfig


@dataclass(frozen=True)
class BenchOffsetConfig:
    bench_theta_offset: float
    bench_phi_offset: float


@dataclass(frozen=True)
class SatelliteInstanceConfig:
    """Per-spacecraft pose and optical-bench offsets."""

    position: Vec3
    bench_theta_offset: float
    bench_phi_offset: float


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    s1: SatelliteInstanceConfig
    s2: SatelliteInstanceConfig
    satellite: SharedSatelliteConfig
    simulation: SimulationConfig
    three_d_viz: ThreeDVizConfig
    eye_viz: EyeVizConfig
    mag_viz: MagVizConfig
    strategy: StrategyConfig
    visualize: str | None = None


@dataclass(frozen=True)
class ScenarioInstance:
    name: str
    s1: BenchOffsetConfig
    s2: BenchOffsetConfig
    chain: tuple[str, ...]
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    simulation_path: Path | None = None
    visualize: str | None = None


def positions_for_distance(dist: float) -> tuple[Vec3, Vec3]:
    """S1 at origin, S2 on +x axis."""
    return as_vec3([0.0, 0.0, 0.0]), as_vec3([dist, 0.0, 0.0])


def build_scenario_config(
    sim: SimulationBundle,
    instance: ScenarioInstance,
    *,
    strategy: StrategyConfig,
) -> ScenarioConfig:
    overrides = dict(instance.overrides)

    satellite = apply_dict_overrides(sim.satellite, overrides.get("satellite", {}))
    simulation = apply_dict_overrides(sim.simulation, overrides.get("simulation", {}))

    three_d_viz_overrides = overrides.get("3d_viz", {})
    three_d_viz = apply_dict_overrides(sim.three_d_viz, three_d_viz_overrides)

    eye_viz_overrides = overrides.get("eye_viz", {})
    eye_viz = apply_dict_overrides(sim.eye_viz, eye_viz_overrides)

    mag_viz_overrides = overrides.get("mag_viz", {})
    mag_viz = apply_dict_overrides(sim.mag_viz, mag_viz_overrides)

    s1_pos, s2_pos = positions_for_distance(simulation.distance)

    return ScenarioConfig(
        name=instance.name,
        s1=SatelliteInstanceConfig(
            position=s1_pos,
            bench_theta_offset=instance.s1.bench_theta_offset,
            bench_phi_offset=instance.s1.bench_phi_offset,
        ),
        s2=SatelliteInstanceConfig(
            position=s2_pos,
            bench_theta_offset=instance.s2.bench_theta_offset,
            bench_phi_offset=instance.s2.bench_phi_offset,
        ),
        satellite=satellite,
        simulation=simulation,
        three_d_viz=three_d_viz,
        eye_viz=eye_viz,
        mag_viz=mag_viz,
        strategy=strategy,
        visualize=instance.visualize,
    )


def default_beam_length(
    position: Vec3,
    partner: Vec3,
    simulation: SimulationConfig,
) -> float:
    """Default TX cone length: link range plus boresight extension."""
    if simulation.beam_length is not None:
        return simulation.beam_length
    return distance(position, partner) + simulation.boresight_extension
