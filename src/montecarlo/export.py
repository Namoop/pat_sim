"""Export a Monte Carlo run as a replayable scenario TOML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from config import load_toml
from montecarlo.run import sample_offsets
from montecarlo.types import MonteCarloConfig
from scenario.types import BenchOffsetConfig


def require_monte_carlo_seed(mc_toml_path: Path) -> None:
    data = load_toml(mc_toml_path)
    if "seed" not in data.get("monte_carlo", {}):
        raise ValueError("A random seed ([monte_carlo].seed) is required for --export")


def monte_carlo_run_seed(mc: MonteCarloConfig, run_number: int) -> int:
    if run_number < 1 or run_number > mc.runs:
        raise ValueError(
            f"run_number must be between 1 and {mc.runs}, got {run_number}"
        )
    chain_rng = np.random.default_rng(mc.seed)
    seeds = chain_rng.integers(0, 2**32 - 1, size=mc.runs).tolist()
    return int(seeds[run_number - 1])


def monte_carlo_run_offsets(
    mc: MonteCarloConfig,
    run_number: int,
) -> tuple[BenchOffsetConfig, BenchOffsetConfig]:
    seed = monte_carlo_run_seed(mc, run_number)
    rng = np.random.default_rng(seed)
    return sample_offsets(mc.error, rng)


def default_export_file_stem(run_number: int) -> str:
    return f"run{run_number}"


def default_export_scenario_name(run_number: int) -> str:
    return f"run {run_number}"


def export_scenario_path(file_stem: str) -> Path:
    return Path("config") / f"_scenario_{file_stem}.toml"


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        inner = ", ".join(_format_toml_value(item) for item in value)
        return f"[{inner}]"
    raise TypeError(f"unsupported TOML value type: {type(value)!r}")


def _format_section(title: str, values: dict[str, Any]) -> list[str]:
    lines = [f"[{title}]"]
    for key, value in values.items():
        lines.append(f"{key} = {_format_toml_value(value)}")
    return lines


def _relative_environment_path(simulation_path: Path, export_path: Path) -> str:
    try:
        return os.path.relpath(simulation_path, export_path.parent)
    except ValueError:
        return simulation_path.name


def _scenario_override_entries(overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for section, props in overrides.items():
        for prop, val in props.items():
            entries[f"{section}.{prop}"] = val
    return entries


def build_scenario_export_text(
    *,
    mc: MonteCarloConfig,
    mc_toml_path: Path,
    run_number: int,
    scenario_name: str,
    export_path: Path,
    visualize: str | None,
) -> str:
    s1, s2 = monte_carlo_run_offsets(mc, run_number)
    data = load_toml(mc_toml_path)
    raw_strategy = data.get("strategy", {})

    blocks: list[str] = []
    scenario_entries: dict[str, Any] = {
        "name": scenario_name,
        "environment": _relative_environment_path(mc.simulation_path, export_path),
        "chain": list(mc.chain),
    }
    if visualize is not None:
        scenario_entries["visualize"] = visualize
    scenario_entries.update(_scenario_override_entries(mc.overrides))
    blocks.extend(_format_section("scenario", scenario_entries))
    blocks.append("")
    blocks.extend(
        _format_section(
            "s1",
            {
                "bench_theta_offset": s1.bench_theta_offset * 1e3,
                "bench_phi_offset": s1.bench_phi_offset * 1e3,
            },
        )
    )
    blocks.append("")
    blocks.extend(
        _format_section(
            "s2",
            {
                "bench_theta_offset": s2.bench_theta_offset * 1e3,
                "bench_phi_offset": s2.bench_phi_offset * 1e3,
            },
        )
    )

    strategy_root = {
        key: value
        for key, value in raw_strategy.items()
        if not isinstance(value, dict) and key != "chain"
    }
    if strategy_root:
        blocks.append("")
        blocks.extend(_format_section("strategy", strategy_root))

    for strat_name in mc.chain:
        blocks.append("")
        blocks.extend(
            _format_section(f"strategy.{strat_name}", raw_strategy.get(strat_name, {}))
        )

    return "\n".join(blocks).rstrip() + "\n"


def export_scenario_toml(
    *,
    mc: MonteCarloConfig,
    mc_toml_path: Path,
    run_number: int,
    file_stem: str | None = None,
    scenario_name: str | None = None,
    visualize: str | None = None,
) -> Path:
    resolved_file_stem = file_stem or default_export_file_stem(run_number)
    resolved_scenario_name = scenario_name or default_export_scenario_name(run_number)
    export_path = export_scenario_path(resolved_file_stem)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_scenario_export_text(
        mc=mc,
        mc_toml_path=mc_toml_path,
        run_number=run_number,
        scenario_name=resolved_scenario_name,
        export_path=export_path,
        visualize=visualize,
    )
    export_path.write_text(text, encoding="utf-8")
    return export_path
