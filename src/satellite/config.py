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


MapVizConfig = EyeVizConfig
DistVizConfig = MagVizConfig


from typing import Any, Mapping


@dataclass(frozen=True)
class StrategyConfig:
    k: float
    chain: tuple[str, ...]
    params: Mapping[str, Any]

    def spiral_w(self, satellite: SharedSatelliteConfig) -> float:
        return self.k * satellite.alpha / 3.141592653589793

    def spiral_duration(
        self, radius: float, w: float, max_beam_speed: float
    ) -> float:
        """T = s(R / w) / max_beam_speed where s is the arc length of the spiral path."""
        if w <= 0 or max_beam_speed <= 0:
            return 0.0
        k = self.k
        if k <= 0.0:
            return radius / (w * max_beam_speed)
        import math
        x = (k * radius) / w
        sqrt_term = math.sqrt(1.0 + x * x)
        arc_len = (w / (2.0 * k)) * (x * sqrt_term + math.log(x + sqrt_term))
        return arc_len / max_beam_speed

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
    three_d_viz: ThreeDVizConfig
    eye_viz: EyeVizConfig
    mag_viz: MagVizConfig
    strategy: StrategyConfig
    visualize: str | None = None


@dataclass(frozen=True)
class SimulationBundle:
    satellite: SharedSatelliteConfig
    simulation: SimulationConfig
    three_d_viz: ThreeDVizConfig
    eye_viz: EyeVizConfig
    mag_viz: MagVizConfig


@dataclass(frozen=True)
class ScenarioInstance:
    name: str
    s1: BenchOffsetConfig
    s2: BenchOffsetConfig
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy: StrategyConfig | None = None
    simulation_path: Path | None = None
    visualize: str | None = None


@dataclass(frozen=True)
class UniformErrorConfig:
    distribution: str
    min: float
    max: float


@dataclass(frozen=True)
class GaussianErrorConfig:
    distribution: str
    mean: float
    std: float


ErrorDistributionConfig = UniformErrorConfig | GaussianErrorConfig


@dataclass(frozen=True)
class MonteCarloConfig:
    simulation_path: Path
    seed: int
    error: ErrorDistributionConfig
    strategy: StrategyConfig
    runs: int
    chain: tuple[str, ...]
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    visual_limit_deg = float(data.get("visual_limit_deg", 25.0))
    fov_cone_length = float(data.get("fov_cone_length", 1.0))
    beam_cone_length = float(data.get("beam_cone_length", 1.0))
    bottom_margin = float(data.get("bottom_margin", 48.0))
    return MagVizConfig(
        visual_limit_deg=visual_limit_deg,
        fov_cone_length=fov_cone_length,
        beam_cone_length=beam_cone_length,
        bottom_margin=bottom_margin,
    )


def _require_float(section: dict, key: str, context: str) -> float:
    if key not in section:
        raise ValueError(f"{context}.{key} is required")
    return float(section[key])


def _load_strategy(data: dict, chain_list: list[str] | None = None, default_k: float = 10.0) -> StrategyConfig:
    from satellite.strategy.base import CONFIG_PARSERS

    global_chain = data.get("chain")
    if global_chain is None:
        if chain_list is not None:
            global_chain = chain_list
        else:
            raise ValueError("strategy.chain is required")

    all_strategy_names = set(global_chain)
    if chain_list:
        all_strategy_names.update(chain_list)

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


def _load_error(error_data: dict) -> ErrorDistributionConfig:
    distribution = str(error_data.get("distribution", "uniform")).lower()
    if distribution == "uniform":
        uniform_data = error_data.get("uniform", {})
        return UniformErrorConfig(
            distribution="uniform",
            min=float(uniform_data.get("min", 0.0)) * 1e-3,
            max=_require_float(uniform_data, "max", "[monte_carlo.error.uniform]") * 1e-3,
        )
    if distribution == "gaussian":
        gaussian_data = error_data.get("gaussian", {})
        return GaussianErrorConfig(
            distribution="gaussian",
            mean=float(gaussian_data.get("mean", 0.0)) * 1e-3,
            std=_require_float(gaussian_data, "std", "[monte_carlo.error.gaussian]") * 1e-3,
        )
    raise ValueError(f"Unsupported [monte_carlo.error].distribution: {distribution!r}")


def load_simulation_config(path: str | Path) -> SimulationBundle:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    return SimulationBundle(
        satellite=_load_shared_satellite(data.get("satellite", {})),
        simulation=_load_simulation_section(data.get("simulation", {})),
        three_d_viz=_load_three_d_viz(data.get("3d_viz", {})),
        eye_viz=_load_eye_viz(data.get("eye_viz", data.get("map_viz", {}))),
        mag_viz=_load_mag_viz(data.get("mag_viz", data.get("dist_viz", {}))),
    )


def load_scenario_config(path: str | Path) -> ScenarioInstance:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    scenario = data.get("scenario", {})
    s1 = data.get("s1", {})
    s2 = data.get("s2", {})
    
    scenario_chain = scenario.get("chain")
    if scenario_chain is None:
        raise ValueError(f"scenario.chain is required in scenario config: {path}")
        
    simulation_rel = scenario.get("environment")
    if simulation_rel is None:
        simulation_rel = scenario.get("simulation_file")
    if simulation_rel is None:
        simulation_val = scenario.get("simulation")
        if isinstance(simulation_val, (str, Path)):
            simulation_rel = simulation_val
            
    simulation_path = None
    if simulation_rel is not None and isinstance(simulation_rel, (str, Path)):
        simulation_path = (config_path.parent / simulation_rel).resolve()
        
    visualize = scenario.get("visualize")
    if visualize is not None:
        visualize = str(visualize).lower()
        if visualize == "map":
            visualize = "eye"
        elif visualize == "dist":
            visualize = "mag"
        if visualize not in ("3d", "eye", "mag"):
            raise ValueError(f"scenario.visualize must be '3d', 'eye', or 'mag': {path}")
    
    overrides: dict[str, dict[str, Any]] = {}
    for key, value in scenario.items():
        if key in ("name", "chain", "environment", "simulation_file", "visualize"):
            continue
        if key == "simulation" and isinstance(value, (str, Path)):
            continue
        
        if "." in key:
            parts = key.split(".", 1)
            section, prop = parts[0], parts[1]
            overrides.setdefault(section, {})[prop] = value
        elif isinstance(value, dict):
            for prop, val in value.items():
                overrides.setdefault(key, {})[prop] = val

    strategy_data = data.get("strategy", {})
    if "chain" not in strategy_data:
        strategy_data = dict(strategy_data)
        strategy_data["chain"] = scenario_chain
    strategy = _load_strategy(strategy_data)

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
        strategy=strategy,
        simulation_path=simulation_path,
        visualize=visualize,
    )


def load_monte_carlo_config(path: str | Path) -> MonteCarloConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    mc = data.get("monte_carlo", {})
    simulation_rel = mc.get("environment")
    if simulation_rel is None:
        simulation_rel = mc.get("simulation_file")
    if simulation_rel is None:
        simulation_rel = mc.get("simulation", "Environment.toml")
    simulation_rel = str(simulation_rel)
    simulation_path = (config_path.parent / simulation_rel).resolve()
    
    if not simulation_path.exists():
        for candidate in [
            config_path.parent.parent / simulation_rel,
            Path.cwd() / "config" / simulation_rel,
            Path.cwd() / simulation_rel,
        ]:
            candidate = candidate.resolve()
            if candidate.exists():
                simulation_path = candidate
                break

    sim_bundle = load_simulation_config(simulation_path)
    
    runs = int(mc.get("runs", 1))
    chain_list = list(mc.get("chain", []))

    overrides: dict[str, dict[str, Any]] = {}
    for key, value in mc.items():
        if key in ("runs", "seed", "chain", "environment", "simulation_file", "error"):
            continue
        if key == "simulation" and not isinstance(value, dict):
            continue
        if "." in key:
            parts = key.split(".", 1)
            section, prop = parts[0], parts[1]
            overrides.setdefault(section, {})[prop] = value
        elif isinstance(value, dict):
            for prop, val in value.items():
                overrides.setdefault(key, {})[prop] = val

    strategy = _load_strategy(data.get("strategy", {}), chain_list=chain_list, default_k=sim_bundle.satellite.k)

    return MonteCarloConfig(
        simulation_path=simulation_path,
        seed=int(mc.get("seed", 0)),
        error=_load_error(mc.get("error", {})),
        strategy=strategy,
        runs=runs,
        chain=tuple(chain_list),
        overrides=overrides,
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
    
    three_d_viz_overrides = overrides.get("3d_viz", overrides.get("three_d_viz", {}))
    three_d_viz = _apply_dict_overrides(sim.three_d_viz, three_d_viz_overrides)
    
    eye_viz_overrides = overrides.get("eye_viz", overrides.get("map_viz", overrides.get("map_visualization", {})))
    eye_viz = _apply_dict_overrides(sim.eye_viz, eye_viz_overrides)

    mag_viz_overrides = overrides.get("mag_viz", overrides.get("dist_viz", overrides.get("dist_visualization", {})))
    mag_viz = _apply_dict_overrides(sim.mag_viz, mag_viz_overrides)

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
        three_d_viz=three_d_viz,
        eye_viz=eye_viz,
        mag_viz=mag_viz,
        strategy=strategy,
        visualize=instance.visualize,
    )


def load_single_scenario(
    scenario_path: str | Path,
    simulation_path: str | Path | None = None,
) -> ScenarioConfig:
    """Merge scenario instance and simulation base, with strategy loaded from scenario."""
    instance = load_scenario_config(scenario_path)
    
    if simulation_path is None:
        simulation_path = instance.simulation_path
        
    if simulation_path is None:
        fallback_path = Path(scenario_path).parent / "Environment.toml"
        if not fallback_path.exists():
            fallback_path = Path("config/Environment.toml")
        simulation_path = fallback_path

    sim = load_simulation_config(simulation_path)
    
    strategy = instance.strategy
    if strategy is None or not strategy.chain:
        raise ValueError(f"Strategy chain must be specified in the scenario config: {scenario_path}")

    return build_scenario_config(sim, instance, strategy=strategy)


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
