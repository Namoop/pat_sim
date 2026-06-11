"""Load simulation, scenario, and Monte Carlo parameters from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from satellite.math3d import Vec3, as_vec3


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
class SimulationConfig:
    distance: float
    t_step: float
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
class MapVisualizationConfig:
    axis_limit: float
    profile_frames: bool
    slider_debounce_ms: int


@dataclass(frozen=True)
class MinorOffsetStrategyConfig:
    duration: float
    max_spiral_radius: str | float
    spiral_speed: float


@dataclass(frozen=True)
class SingleMissStrategyConfig:
    phase1_duration: float
    a_spiral_radius: str | float
    reset_duration: float
    phase2_duration: float
    b_spiral_radius: str | float
    spiral_speed: float


@dataclass(frozen=True)
class AsymmetricProbeStrategyConfig:
    probe_duration: float
    spiral_radius: str | float
    spiral_speed: float = 1.0
    reset_duration: float = 0.0


@dataclass(frozen=True)
class StrategyConfig:
    k: float
    chain: tuple[str, ...]
    minor_offset: MinorOffsetStrategyConfig
    single_miss: SingleMissStrategyConfig
    asymmetric_probe: AsymmetricProbeStrategyConfig

    def spiral_w(self, satellite: SharedSatelliteConfig) -> float:
        return self.k * satellite.alpha / 3.141592653589793

    @staticmethod
    def resolve_radius(value: str | float, dish_fov: float) -> float:
        if isinstance(value, str) and value.lower() == "fov":
            return dish_fov
        return float(value)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    s1: SatelliteInstanceConfig
    s2: SatelliteInstanceConfig
    satellite: SharedSatelliteConfig
    simulation: SimulationConfig
    visualization: VisualizationConfig
    map_visualization: MapVisualizationConfig
    strategy: StrategyConfig


@dataclass(frozen=True)
class SimulationBundle:
    satellite: SharedSatelliteConfig
    simulation: SimulationConfig
    visualization: VisualizationConfig
    map_visualization: MapVisualizationConfig


@dataclass(frozen=True)
class ScenarioInstance:
    name: str
    distance: float | None
    s1: BenchOffsetConfig
    s2: BenchOffsetConfig


@dataclass(frozen=True)
class UniformErrorConfig:
    distribution: str
    theta_min: float
    theta_max: float
    phi_min: float
    phi_max: float


@dataclass(frozen=True)
class GaussianErrorConfig:
    distribution: str
    theta_mean: float
    theta_std: float
    phi_mean: float
    phi_std: float


ErrorDistributionConfig = UniformErrorConfig | GaussianErrorConfig


@dataclass(frozen=True)
class MonteCarloConfig:
    simulation_path: Path
    runs: int
    seed: int
    error: ErrorDistributionConfig
    strategy: StrategyConfig


def positions_for_distance(distance: float) -> tuple[Vec3, Vec3]:
    """S1 at origin, S2 on +x axis."""
    return as_vec3([0.0, 0.0, 0.0]), as_vec3([distance, 0.0, 0.0])


def _load_shared_satellite(data: dict) -> SharedSatelliteConfig:
    return SharedSatelliteConfig(
        body_radius=float(data.get("body_radius", 0.5)),
        dish_fov=float(data.get("dish_fov", 0.002)),
        bench_slew_time=float(
            data.get("bench_slew_time", data.get("dish_slew_time", 0.3))
        ),
        fsm_settle_time=float(data.get("fsm_settle_time", 0.0)),
        beam_width_mrad=float(data["beam_width"]),
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
        profile_replay=bool(data.get("profile_replay", False)),
    )


def _load_visualization(data: dict) -> VisualizationConfig:
    return VisualizationConfig(
        enabled=bool(data.get("enabled", False)),
        cone_u_steps=int(data.get("cone_u_steps", 24)),
        cone_v_steps=int(data.get("cone_v_steps", 32)),
        spiral_trail_steps=int(data.get("spiral_trail_steps", 80)),
        ribbon_v_steps=int(data.get("ribbon_v_steps", 4)),
        profile_frames=bool(data.get("profile_frames", False)),
    )


def _load_map_visualization(data: dict) -> MapVisualizationConfig:
    return MapVisualizationConfig(
        axis_limit=float(data.get("axis_limit", 0.1)),
        profile_frames=bool(data.get("profile_frames", False)),
        slider_debounce_ms=int(data.get("slider_debounce_ms", 16)),
    )


def _require_float(section: dict, key: str, context: str) -> float:
    if key not in section:
        raise ValueError(f"{context}.{key} is required")
    return float(section[key])


def _load_strategy(data: dict) -> StrategyConfig:
    minor_offset_cfg = data.get("minor_offset", {})
    single_miss_cfg = data.get("single_miss", {})
    if "k" not in data:
        raise ValueError("[strategy].k is required")
    if "duration" not in minor_offset_cfg:
        raise ValueError("[strategy.minor_offset].duration is required")
    if "phase1_duration" not in single_miss_cfg:
        raise ValueError("[strategy.single_miss].phase1_duration is required")
    if "phase2_duration" not in single_miss_cfg:
        raise ValueError("[strategy.single_miss].phase2_duration is required")

    asymmetric_probe_cfg = data.get("asymmetric_probe", {})

    return StrategyConfig(
        k=float(data["k"]),
        chain=tuple(data.get("chain", ["minor_offset", "single_miss"])),
        minor_offset=MinorOffsetStrategyConfig(
            duration=float(minor_offset_cfg["duration"]),
            max_spiral_radius=minor_offset_cfg.get("max_spiral_radius", "fov"),
            spiral_speed=float(minor_offset_cfg.get("spiral_speed", 1.0)),
        ),
        single_miss=SingleMissStrategyConfig(
            phase1_duration=float(single_miss_cfg["phase1_duration"]),
            a_spiral_radius=single_miss_cfg.get("a_spiral_radius", 0.05),
            reset_duration=float(single_miss_cfg.get("reset_duration", 0.0)),
            phase2_duration=float(single_miss_cfg["phase2_duration"]),
            b_spiral_radius=single_miss_cfg.get("b_spiral_radius", 0.05),
            spiral_speed=float(single_miss_cfg.get("spiral_speed", 1.0)),
        ),
        asymmetric_probe=AsymmetricProbeStrategyConfig(
            probe_duration=float(
                asymmetric_probe_cfg.get(
                    "probe_duration",
                    single_miss_cfg.get("phase1_duration", 3.15),
                )
            ),
            spiral_radius=asymmetric_probe_cfg.get(
                "spiral_radius",
                single_miss_cfg.get("a_spiral_radius", 0.05),
            ),
            spiral_speed=float(asymmetric_probe_cfg.get("spiral_speed", 1.0)),
            reset_duration=float(
                asymmetric_probe_cfg.get(
                    "reset_duration",
                    single_miss_cfg.get("reset_duration", 0.0),
                )
            ),
        ),
    )


def _load_error(data: dict) -> ErrorDistributionConfig:
    distribution = str(data.get("distribution", "uniform")).lower()
    if distribution == "uniform":
        return UniformErrorConfig(
            distribution="uniform",
            theta_min=_require_float(data, "theta_min", "[error]"),
            theta_max=_require_float(data, "theta_max", "[error]"),
            phi_min=_require_float(data, "phi_min", "[error]"),
            phi_max=_require_float(data, "phi_max", "[error]"),
        )
    if distribution == "gaussian":
        return GaussianErrorConfig(
            distribution="gaussian",
            theta_mean=float(data.get("theta_mean", 0.0)),
            theta_std=_require_float(data, "theta_std", "[error]"),
            phi_mean=float(data.get("phi_mean", 0.0)),
            phi_std=_require_float(data, "phi_std", "[error]"),
        )
    raise ValueError(f"Unsupported [error].distribution: {distribution!r}")


def load_simulation_config(path: str | Path) -> SimulationBundle:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    return SimulationBundle(
        satellite=_load_shared_satellite(data.get("satellite", {})),
        simulation=_load_simulation_section(data.get("simulation", {})),
        visualization=_load_visualization(data.get("visualization", {})),
        map_visualization=_load_map_visualization(data.get("map_visualization", {})),
    )


def load_scenario_config(path: str | Path) -> ScenarioInstance:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    scenario = data.get("scenario", {})
    s1 = data.get("s1", {})
    s2 = data.get("s2", {})
    distance = scenario.get("distance")
    return ScenarioInstance(
        name=str(scenario.get("name", "unnamed")),
        distance=float(distance) if distance is not None else None,
        s1=BenchOffsetConfig(
            bench_theta_offset=float(s1.get("bench_theta_offset", 0.0)),
            bench_phi_offset=float(s1.get("bench_phi_offset", 0.0)),
        ),
        s2=BenchOffsetConfig(
            bench_theta_offset=float(s2.get("bench_theta_offset", 0.0)),
            bench_phi_offset=float(s2.get("bench_phi_offset", 0.0)),
        ),
    )


def load_monte_carlo_config(path: str | Path) -> MonteCarloConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    mc = data.get("monte_carlo", {})
    simulation_rel = str(mc.get("simulation", "Simulation.toml"))
    simulation_path = (config_path.parent / simulation_rel).resolve()
    return MonteCarloConfig(
        simulation_path=simulation_path,
        runs=int(mc.get("runs", 1)),
        seed=int(mc.get("seed", 0)),
        error=_load_error(data.get("error", {})),
        strategy=_load_strategy(data.get("strategy", {})),
    )


def build_scenario_config(
    sim: SimulationBundle,
    instance: ScenarioInstance,
    *,
    strategy: StrategyConfig,
) -> ScenarioConfig:
    distance = (
        instance.distance
        if instance.distance is not None
        else sim.simulation.distance
    )
    s1_pos, s2_pos = positions_for_distance(distance)
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
        satellite=sim.satellite,
        simulation=sim.simulation,
        visualization=sim.visualization,
        map_visualization=sim.map_visualization,
        strategy=strategy,
    )


def load_single_scenario(
    scenario_path: str | Path,
    simulation_path: str | Path,
    strategy_path: str | Path,
) -> ScenarioConfig:
    """Merge scenario instance, simulation base, and strategy for a single run."""
    sim = load_simulation_config(simulation_path)
    instance = load_scenario_config(scenario_path)
    mc = load_monte_carlo_config(strategy_path)
    return build_scenario_config(sim, instance, strategy=mc.strategy)


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
