"""Load simulation, scenario, and Monte Carlo parameters from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace, is_dataclass
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


from typing import Any, Mapping


@dataclass(frozen=True)
class StrategyConfig:
    k: float
    chain: tuple[str, ...]
    params: Mapping[str, Any]

    def spiral_w(self, satellite: SharedSatelliteConfig) -> float:
        return self.k * satellite.alpha / 3.141592653589793

    def spiral_duration(
        self, radius: float, w: float, speed: float = 1.0
    ) -> float:
        """T = R / (w * speed)"""
        if w <= 0:
            return 0.0
        return radius / (w * speed)

    def reset_duration(
        self, radius: float, max_beam_speed: float
    ) -> float:
        """T_reset = R / max_beam_speed"""
        if max_beam_speed <= 0:
            return 0.0
        return radius / max_beam_speed

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
    s1: BenchOffsetConfig
    s2: BenchOffsetConfig
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


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
class MonteCarloChainConfig:
    runs: int
    chain: tuple[str, ...]


@dataclass(frozen=True)
class MonteCarloConfig:
    simulation_path: Path
    seed: int
    error: ErrorDistributionConfig
    strategy: StrategyConfig
    chains: tuple[MonteCarloChainConfig, ...]

    @property
    def runs(self) -> int:
        return sum(c.runs for c in self.chains)


def positions_for_distance(distance: float) -> tuple[Vec3, Vec3]:
    """S1 at origin, S2 on +x axis."""
    return as_vec3([0.0, 0.0, 0.0]), as_vec3([distance, 0.0, 0.0])


def _load_shared_satellite(data: dict) -> SharedSatelliteConfig:
    return SharedSatelliteConfig(
        body_radius=float(data.get("body_radius", 0.5)),
        dish_fov=float(data.get("dish_fov", 0.002)) * 1e-3,
        max_beam_speed=float(data.get("max_beam_speed", 0.087)) * 1e-3,  # ~5 deg/s
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
        axis_limit=float(data.get("axis_limit", 0.1)) * 1e-3,
        profile_frames=bool(data.get("profile_frames", False)),
        slider_debounce_ms=int(data.get("slider_debounce_ms", 16)),
    )


def _require_float(section: dict, key: str, context: str) -> float:
    if key not in section:
        raise ValueError(f"{context}.{key} is required")
    return float(section[key])


def _load_strategy(data: dict, chains_data: list | None = None, default_k: float = 10.0) -> StrategyConfig:
    from satellite.strategy.base import CONFIG_PARSERS

    global_chain = data.get("chain", ["minor_offset", "single_miss"])
    all_strategy_names = set(global_chain)
    if chains_data:
        for c in chains_data:
            all_strategy_names.update(c.get("chain", []))

    params = {}
    for name in all_strategy_names:
        section = data.get(name, {})
        if name in CONFIG_PARSERS:
            params[name] = CONFIG_PARSERS[name](section)
        else:
            # Fallback for strategies not yet updated/stubbed or without config
            params[name] = section

    return StrategyConfig(
        k=float(data.get("k", default_k)),
        chain=tuple(global_chain),
        params=params,
    )


def _load_error(data: dict) -> ErrorDistributionConfig:
    distribution = str(data.get("distribution", "uniform")).lower()
    if distribution == "uniform":
        return UniformErrorConfig(
            distribution="uniform",
            theta_min=_require_float(data, "theta_min", "[error]") * 1e-3,
            theta_max=_require_float(data, "theta_max", "[error]") * 1e-3,
            phi_min=_require_float(data, "phi_min", "[error]") * 1e-3,
            phi_max=_require_float(data, "phi_max", "[error]") * 1e-3,
        )
    if distribution == "gaussian":
        return GaussianErrorConfig(
            distribution="gaussian",
            theta_mean=float(data.get("theta_mean", 0.0)) * 1e-3,
            theta_std=_require_float(data, "theta_std", "[error]") * 1e-3,
            phi_mean=float(data.get("phi_mean", 0.0)) * 1e-3,
            phi_std=_require_float(data, "phi_std", "[error]") * 1e-3,
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
    
    overrides: dict[str, dict[str, Any]] = {}
    for key, value in scenario.items():
        if key == "name":
            continue
        
        if "." in key:
            parts = key.split(".", 1)
            section, prop = parts[0], parts[1]
            overrides.setdefault(section, {})[prop] = value
        elif isinstance(value, dict):
            for prop, val in value.items():
                overrides.setdefault(key, {})[prop] = val

    return ScenarioInstance(
        name=str(scenario.get("name", "unnamed")),
        s1=BenchOffsetConfig(
            bench_theta_offset=float(s1.get("bench_theta_offset", 0.0)) * 1e-3,
            bench_phi_offset=float(s1.get("bench_phi_offset", 0.0)) * 1e-3,
        ),
        s2=BenchOffsetConfig(
            bench_theta_offset=float(s2.get("bench_theta_offset", 0.0)) * 1e-3,
            bench_phi_offset=float(s2.get("bench_phi_offset", 0.0)) * 1e-3,
        ),
        overrides=overrides,
    )


def load_monte_carlo_config(path: str | Path) -> MonteCarloConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    mc = data.get("monte_carlo", {})
    simulation_rel = str(mc.get("simulation", "Simulation.toml"))
    simulation_path = (config_path.parent / simulation_rel).resolve()
    
    if not simulation_path.exists():
        for candidate in [
            config_path.parent.parent / simulation_rel,
            Path.cwd() / simulation_rel,
        ]:
            candidate = candidate.resolve()
            if candidate.exists():
                simulation_path = candidate
                break

    sim_bundle = load_simulation_config(simulation_path)
    
    chains_data = mc.get("chains") or data.get("chains")
    if not chains_data:
        raise ValueError("monte_carlo.chains is required")
    strategy = _load_strategy(data.get("strategy", {}), chains_data=chains_data, default_k=sim_bundle.satellite.k)
    
    chains = []
    for c in chains_data:
        chain_list = list(c.get("chain", []))
        if not chain_list:
            raise ValueError("each chain in monte_carlo.chains must specify a non-empty 'chain' list of strategy names")
        r = int(c.get("runs", 1))
        chains.append(MonteCarloChainConfig(runs=r, chain=tuple(chain_list)))

    return MonteCarloConfig(
        simulation_path=simulation_path,
        seed=int(mc.get("seed", 0)),
        error=_load_error(data.get("error", {})),
        strategy=strategy,
        chains=tuple(chains),
    )


SCALED_PROPERTIES = {
    "dish_fov",
    "max_beam_speed",
    "max_fsm_speed",
    "max_fsm_radius",
    "max_search_radius",
    "axis_limit",
}


def _apply_dict_overrides(obj, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return obj
    kwargs = {}
    for key, val in overrides.items():
        if hasattr(obj, key):
            current_val = getattr(obj, key)
            if is_dataclass(current_val) and isinstance(val, dict):
                kwargs[key] = _apply_dict_overrides(current_val, val)
            else:
                if key in SCALED_PROPERTIES and isinstance(val, (int, float)):
                    val = float(val) * 1e-3
                kwargs[key] = val
    return replace(obj, **kwargs)


def build_scenario_config(
    sim: SimulationBundle,
    instance: ScenarioInstance,
    *,
    strategy: StrategyConfig,
) -> ScenarioConfig:
    overrides = dict(instance.overrides)

    satellite = _apply_dict_overrides(sim.satellite, overrides.get("satellite", {}))
    simulation = _apply_dict_overrides(sim.simulation, overrides.get("simulation", {}))
    visualization = _apply_dict_overrides(sim.visualization, overrides.get("visualization", {}))
    map_visualization = _apply_dict_overrides(sim.map_visualization, overrides.get("map_visualization", {}))

    distance = simulation.distance
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
        satellite=satellite,
        simulation=simulation,
        visualization=visualization,
        map_visualization=map_visualization,
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
