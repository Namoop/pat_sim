"""Scenario instance TOML parser ([scenario], [s1], [s2])."""

from __future__ import annotations

from pathlib import Path

from config import collect_overrides, resolve_relative_path
from scenario.types import BenchOffsetConfig, ScenarioInstance

_SCENARIO_SKIP = frozenset({"name", "chain", "environment", "visualize"})


def parse(data: dict, *, path: str | Path) -> ScenarioInstance:
    config_path = Path(path)
    scenario = data.get("scenario", {})
    s1 = data.get("s1", {})
    s2 = data.get("s2", {})

    scenario_chain = scenario.get("chain")
    if scenario_chain is None:
        raise ValueError(f"scenario.chain is required in scenario config: {path}")

    simulation_rel = scenario.get("environment")

    simulation_path = None
    if simulation_rel is not None and isinstance(simulation_rel, (str, Path)):
        simulation_path = resolve_relative_path(simulation_rel, config_path)

    visualize = scenario.get("visualize")
    if visualize is not None:
        visualize = str(visualize).lower()
        if visualize not in ("3d", "eye", "mag"):
            raise ValueError(f"scenario.visualize must be '3d', 'eye', or 'mag': {path}")

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
        chain=tuple(scenario_chain),
        overrides=collect_overrides(scenario, skip_keys=_SCENARIO_SKIP),
        simulation_path=simulation_path,
        visualize=visualize,
    )


def load_scenario_config(path: str | Path) -> ScenarioInstance:
    config_path = Path(path)
    from config import load_toml

    return parse(load_toml(config_path), path=config_path)
