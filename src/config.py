"""Shared TOML I/O, path resolution, and override merge helpers."""

from __future__ import annotations

import tomllib
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

SCALED_PROPERTIES = {
    "dish_fov",
    "max_beam_speed",
    "max_fsm_speed",
    "max_fsm_radius",
    "max_search_radius",
    "axis_limit",
}


def load_toml(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def collect_overrides(
    section: dict,
    *,
    skip_keys: frozenset[str],
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for key, value in section.items():
        if key in skip_keys:
            continue
        if key == "simulation" and isinstance(value, (str, Path)):
            continue
        if "." in key:
            part, prop = key.split(".", 1)
            overrides.setdefault(part, {})[prop] = value
        elif isinstance(value, dict):
            for prop, val in value.items():
                overrides.setdefault(key, {})[prop] = val
    return overrides


def resolve_relative_path(rel: str | Path, config_path: Path) -> Path:
    return (config_path.parent / rel).resolve()


def resolve_environment_path(rel: str | Path, config_path: Path) -> Path:
    simulation_path = resolve_relative_path(rel, config_path)
    if simulation_path.exists():
        return simulation_path
    rel_name = Path(rel).name
    for candidate in [
        config_path.parent.parent / rel,
        Path.cwd() / "config" / rel_name,
        Path.cwd() / rel_name,
    ]:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return simulation_path


def apply_dict_overrides(obj, overrides: dict[str, Any]):
    if not overrides:
        return obj
    kwargs = {}
    for key, val in overrides.items():
        if hasattr(obj, key):
            current_val = getattr(obj, key)
            if is_dataclass(current_val) and isinstance(val, dict):
                kwargs[key] = apply_dict_overrides(current_val, val)
            else:
                if key in SCALED_PROPERTIES and isinstance(val, (int, float)):
                    val = float(val) * 1e-3
                kwargs[key] = val
    return replace(obj, **kwargs)
