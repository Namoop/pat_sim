"""Write structured optimization result logs under optimize/logs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from montecarlo.types import GaussianErrorConfig, MonteCarloConfig, UniformErrorConfig
from satellite.config import SimulationBundle
from scenario.types import BenchOffsetConfig, ScenarioInstance, build_scenario_config


LOG_DIR = Path(__file__).resolve().parent / "logs"
CONFIG_HASH_PREFIX = "# config_hash: "
_CONFIG_HASH_RE = re.compile(r"^# config_hash:\s*([0-9a-f]{64})\s*$")

# Satellite fields: (config attr, toml key, scale_to_mrad)
_SATELLITE_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("max_beam_speed", "max_beam_speed", True),
    ("dish_fov", "dish_fov", True),
    ("max_fsm_speed", "max_fsm_speed", True),
    ("max_fsm_radius", "max_fsm_radius", True),
    ("beam_width_mrad", "beam_width", False),
    ("k", "k", False),
    ("scan_envelope_ramp", "scan_envelope_ramp", False),
    ("scan_envelope_profile", "scan_envelope_profile", False),
)
_SIMULATION_KEYS = (
    "distance",
    "t_step",
    "max_search_radius",
    "timeout",
    "enforce_speed_limit",
)

_MRAD_SIMULATION = frozenset({"max_search_radius"})


def build_config_fingerprint_text(
    *,
    optimize_section: dict[str, Any],
    mc_cfg: MonteCarloConfig,
    sim: SimulationBundle,
    strategy: str,
) -> str:
    """Canonical TOML for [optimize], [monte_carlo], [satellite], [simulation]."""
    blocks: list[str] = []
    blocks.extend(_format_section("optimize", optimize_section))
    blocks.append("")
    blocks.extend(_format_section("monte_carlo", _monte_carlo_section(mc_cfg, strategy=strategy)))
    blocks.append("")
    blocks.extend(_format_section("satellite", _satellite_section(sim, mc_cfg, strategy=strategy)))
    blocks.append("")
    blocks.extend(_format_section("simulation", _simulation_section(sim, mc_cfg, strategy=strategy)))
    return "\n".join(blocks) + "\n"


def config_log_hash(fingerprint: str) -> tuple[str, str]:
    """Return (full_hash, short_hash) where short_hash is the first 6 hex chars."""
    full_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
    return full_hash, full_hash[:6]


def log_path_for_config(strategy: str, short_hash: str) -> Path:
    return LOG_DIR / f"{strategy}_{short_hash}.log"


def read_config_hash_from_log(path: Path) -> str | None:
    """Read stored full config hash from the first lines of a result log."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines()[:10]:
        match = _CONFIG_HASH_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def assert_safe_overwrite(path: Path, full_hash: str) -> None:
    """Raise if an existing log at *path* belongs to a different config fingerprint."""
    if not path.exists():
        return
    stored = read_config_hash_from_log(path)
    if stored is None or stored == full_hash:
        return
    raise ValueError(
        f"Config hash prefix collision at {path}: existing log has "
        f"config_hash={stored}, new config_hash={full_hash}. "
        f"Rename or remove the existing file before writing."
    )


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


def _monte_carlo_section(mc_cfg: MonteCarloConfig, *, strategy: str) -> dict[str, Any]:
    chain = list(mc_cfg.chain) if mc_cfg.chain else [strategy]
    section: dict[str, Any] = {
        "seed": mc_cfg.seed,
        "runs": mc_cfg.runs,
        "chain": chain,
    }
    error = mc_cfg.error
    if isinstance(error, GaussianErrorConfig):
        section["error.distribution"] = "gaussian"
        section["error.gaussian.mean"] = error.mean * 1e3
        section["error.gaussian.limit"] = error.limit * 1e3
        section["error.gaussian.confidence"] = error.confidence
    elif isinstance(error, UniformErrorConfig):
        section["error.distribution"] = "uniform"
        section["error.uniform.min"] = error.min * 1e3
        section["error.uniform.max"] = error.max * 1e3
    return section


def _satellite_section(
    sim: SimulationBundle, mc_cfg: MonteCarloConfig, *, strategy: str
) -> dict[str, Any]:
    instance = ScenarioInstance(
        name="optimize_log",
        s1=BenchOffsetConfig(0.0, 0.0),
        s2=BenchOffsetConfig(0.0, 0.0),
        chain=mc_cfg.chain or (strategy,),
        overrides=mc_cfg.overrides,
    )
    cfg = build_scenario_config(sim, instance, strategy=mc_cfg.strategy)
    sat = cfg.satellite
    raw = asdict(sat) if is_dataclass(sat) else dict(sat.__dict__)
    section: dict[str, Any] = {}
    for attr, toml_key, scale_mrad in _SATELLITE_FIELDS:
        value = raw.get(attr)
        if value is None:
            continue
        if scale_mrad:
            value = value * 1e3
        section[toml_key] = value
    return section


def _simulation_section(
    sim: SimulationBundle, mc_cfg: MonteCarloConfig, *, strategy: str
) -> dict[str, Any]:
    instance = ScenarioInstance(
        name="optimize_log",
        s1=BenchOffsetConfig(0.0, 0.0),
        s2=BenchOffsetConfig(0.0, 0.0),
        chain=mc_cfg.chain or (strategy,),
        overrides=mc_cfg.overrides,
    )
    cfg = build_scenario_config(sim, instance, strategy=mc_cfg.strategy)
    sim_cfg = cfg.simulation
    raw = asdict(sim_cfg) if is_dataclass(sim_cfg) else dict(sim_cfg.__dict__)
    section: dict[str, Any] = {}
    for key in _SIMULATION_KEYS:
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if key in _MRAD_SIMULATION:
            value = value * 1e3
        section[key] = value
    return section


def format_best_eval_comments(
    *,
    runs: int,
    success_rate: float,
    mean_t: float | None,
    median_t: float | None,
) -> list[str]:
    successes = int(round(success_rate * runs))
    lines = [
        f"# Monte Carlo: {successes}/{runs} succeeded ({100.0 * success_rate:.1f}%)"
    ]
    if mean_t is not None and median_t is not None:
        lines.append(f"# Success sim-t: mean={mean_t:.3f}, median={median_t:.3f}")
    return lines


def build_log_header(
    *,
    interrupted: bool,
    elapsed: float,
    completed_trials: int,
    best_cost: float,
    best_params: dict[str, Any] | None,
    runs: int,
    success_rate: float | None,
    mean_t: float | None,
    median_t: float | None,
) -> list[str]:
    if interrupted:
        status = (
            f"--- Optimization interrupted after {elapsed:.1f}s "
            f"({completed_trials} trial(s) completed) ---"
        )
    else:
        status = f"--- Optimization completed in {elapsed:.1f}s ---"

    lines = [f"# {status}"]
    if best_params:
        lines.append(f"# Best objective cost score: {best_cost:.4f}")
        if success_rate is not None:
            lines.extend(
                format_best_eval_comments(
                    runs=runs,
                    success_rate=success_rate,
                    mean_t=mean_t,
                    median_t=median_t,
                )
            )
    lines.append("")
    lines.append("# Optimal parameters:")
    return lines


def build_result_log_text(
    *,
    optimize_section: dict[str, Any],
    mc_cfg: MonteCarloConfig,
    sim: SimulationBundle,
    strategy: str,
    best_params: dict[str, Any] | None,
    interrupted: bool,
    elapsed: float,
    completed_trials: int,
    best_cost: float,
    full_hash: str,
    success_rate: float | None = None,
    mean_t: float | None = None,
    median_t: float | None = None,
) -> str:
    blocks: list[str] = [f"{CONFIG_HASH_PREFIX}{full_hash}"]
    blocks.extend(
        build_log_header(
            interrupted=interrupted,
            elapsed=elapsed,
            completed_trials=completed_trials,
            best_cost=best_cost,
            best_params=best_params,
            runs=mc_cfg.runs,
            success_rate=success_rate,
            mean_t=mean_t,
            median_t=median_t,
        )
    )
    if best_params:
        blocks.extend(_format_section(f"strategy.{strategy}", best_params))
        blocks.append("")
    blocks.extend(_format_section("optimize", optimize_section))
    blocks.append("")
    blocks.extend(_format_section("monte_carlo", _monte_carlo_section(mc_cfg, strategy=strategy)))
    blocks.append("")
    blocks.extend(_format_section("satellite", _satellite_section(sim, mc_cfg, strategy=strategy)))
    blocks.append("")
    blocks.extend(_format_section("simulation", _simulation_section(sim, mc_cfg, strategy=strategy)))
    return "\n".join(blocks) + "\n"


def write_result_log(
    *,
    strategy: str,
    optimize_section: dict[str, Any],
    mc_cfg: MonteCarloConfig,
    sim: SimulationBundle,
    best_params: dict[str, Any] | None,
    interrupted: bool,
    elapsed: float,
    completed_trials: int,
    best_cost: float,
    success_rate: float | None = None,
    mean_t: float | None = None,
    median_t: float | None = None,
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = build_config_fingerprint_text(
        optimize_section=optimize_section,
        mc_cfg=mc_cfg,
        sim=sim,
        strategy=strategy,
    )
    full_hash, short_hash = config_log_hash(fingerprint)
    path = log_path_for_config(strategy, short_hash)
    assert_safe_overwrite(path, full_hash)
    text = build_result_log_text(
        optimize_section=optimize_section,
        mc_cfg=mc_cfg,
        sim=sim,
        strategy=strategy,
        best_params=best_params,
        interrupted=interrupted,
        elapsed=elapsed,
        completed_trials=completed_trials,
        best_cost=best_cost,
        full_hash=full_hash,
        success_rate=success_rate,
        mean_t=mean_t,
        median_t=median_t,
    )
    path.write_text(text, encoding="utf-8")
    return path


def build_summary_lines(
    *,
    interrupted: bool,
    elapsed: float,
    completed_trials: int,
    best_cost: float,
    best_params: dict[str, Any] | None,
) -> list[str]:
    lines: list[str] = []
    if interrupted:
        lines.append(
            f"--- Optimization interrupted after {elapsed:.1f}s "
            f"({completed_trials} trial(s) completed) ---"
        )
    else:
        lines.append(f"--- Optimization completed in {elapsed:.1f}s ---")

    if best_params:
        lines.append(f"Best objective cost score: {best_cost:.4f}")
        lines.append("Optimal Parameters:")
        for key, value in best_params.items():
            if isinstance(value, float):
                lines.append(f"  {key} = {value:.6f}")
            else:
                lines.append(f"  {key} = {value}")
    elif interrupted and completed_trials == 0:
        lines.append("Stopped before any trials finished.")
    return lines
