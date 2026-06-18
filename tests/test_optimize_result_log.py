"""Tests for optimize result log formatting."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from montecarlo.types import GaussianErrorConfig, MonteCarloConfig
from optimize import result_log
from optimize.result_log import (
    CONFIG_HASH_PREFIX,
    assert_safe_overwrite,
    build_config_fingerprint_text,
    build_log_header,
    build_result_log_text,
    config_log_hash,
    format_best_eval_comments,
    log_path_for_config,
    read_config_hash_from_log,
    write_result_log,
)
from satellite.config import load_simulation_config
from strategy.config import StrategyConfig


def _sample_mc_cfg() -> MonteCarloConfig:
    return MonteCarloConfig(
        simulation_path=Path("config/Environment.toml"),
        seed=42,
        error=GaussianErrorConfig(distribution="gaussian", mean=0.0, std=0.0015),
        strategy=StrategyConfig(k=10.0, chain=("lissajous_scan",), params={}),
        runs=30,
        chain=("lissajous_scan",),
    )


def _sample_optimize_section() -> dict:
    return {
        "strategy": "lissajous_scan",
        "method": "optuna",
        "trials": 100,
        "trial_seed": 101,
    }


def _fingerprint_and_hash(
    *,
    optimize_section: dict | None = None,
    mc_cfg: MonteCarloConfig | None = None,
    sim=None,
    strategy: str = "lissajous_scan",
) -> tuple[str, str, str]:
    sim = sim or load_simulation_config("config/Environment.toml")
    mc_cfg = mc_cfg or _sample_mc_cfg()
    optimize_section = optimize_section or _sample_optimize_section()
    fingerprint = build_config_fingerprint_text(
        optimize_section=optimize_section,
        mc_cfg=mc_cfg,
        sim=sim,
        strategy=strategy,
    )
    full_hash, short_hash = config_log_hash(fingerprint)
    return fingerprint, full_hash, short_hash


def test_build_result_log_contains_sections(tmp_path: Path):
    sim = load_simulation_config("config/Environment.toml")
    mc_cfg = _sample_mc_cfg()
    best_params = {"s1_wx": 1.5, "s1_wy": 2.0, "s2_wx": 3.0, "s2_wy": 4.0}
    _, full_hash, _ = _fingerprint_and_hash(sim=sim, mc_cfg=mc_cfg)
    text = build_result_log_text(
        optimize_section=_sample_optimize_section(),
        mc_cfg=mc_cfg,
        sim=sim,
        strategy="lissajous_scan",
        best_params=best_params,
        interrupted=False,
        elapsed=12.3,
        completed_trials=10,
        best_cost=6.93,
        full_hash=full_hash,
        success_rate=1.0,
        mean_t=7.65,
        median_t=7.65,
    )
    strategy_idx = text.index("[strategy.lissajous_scan]")
    optimize_idx = text.index("[optimize]")
    assert strategy_idx < optimize_idx
    assert text.startswith(f"{CONFIG_HASH_PREFIX}{full_hash}\n# --- Optimization completed in 12.3s ---")
    assert "# Best objective cost score: 6.9300" in text
    assert "# Monte Carlo: 30/30 succeeded (100.0%)" in text
    assert "# Success sim-t: mean=7.650, median=7.650" in text
    assert "# Optimal parameters:\n[strategy.lissajous_scan]" in text
    assert "[monte_carlo]" in text
    assert "[satellite]" in text
    assert "max_fsm_speed" in text
    assert "beam_width" in text
    assert "[simulation]" in text
    assert "max_search_radius" in text
    assert "scan_envelope_profile" in text
    assert "boresight_extension" not in text
    assert "environment" not in text
    assert "s1_wx = 1.5" in text


def test_build_log_header_without_best_params():
    lines = build_log_header(
        interrupted=True,
        elapsed=1.0,
        completed_trials=0,
        best_cost=float("inf"),
        best_params=None,
        runs=30,
        success_rate=None,
        mean_t=None,
        median_t=None,
    )
    assert lines[0].startswith("# --- Optimization interrupted")
    assert lines[-1] == "# Optimal parameters:"
    assert not any("Best objective" in line for line in lines)


def test_format_best_eval_comments_omits_sim_t_without_successes():
    lines = format_best_eval_comments(
        runs=30, success_rate=0.0, mean_t=None, median_t=None
    )
    assert len(lines) == 1
    assert "0/30 succeeded" in lines[0]


def test_config_fingerprint_is_stable():
    fp1, full1, short1 = _fingerprint_and_hash()
    fp2, full2, short2 = _fingerprint_and_hash()
    assert fp1 == fp2
    assert full1 == full2
    assert short1 == short2


def test_config_fingerprint_changes_with_trials():
    _, full_a, _ = _fingerprint_and_hash()
    section = _sample_optimize_section()
    section["trials"] = 200
    _, full_b, _ = _fingerprint_and_hash(optimize_section=section)
    assert full_a != full_b


def test_config_fingerprint_changes_with_max_search_radius():
    sim = load_simulation_config("config/Environment.toml")
    _, full_a, _ = _fingerprint_and_hash(sim=sim)

    mc_b = replace(
        _sample_mc_cfg(),
        overrides={"simulation": {"max_search_radius": 0.004}},
    )
    _, full_b, _ = _fingerprint_and_hash(sim=sim, mc_cfg=mc_b)
    assert full_a != full_b


def test_config_fingerprint_ignores_best_params():
    fp_a, full_a, _ = _fingerprint_and_hash()
    fp_b, full_b, _ = _fingerprint_and_hash()
    assert fp_a == fp_b
    assert full_a == full_b


def test_log_path_for_config_pattern():
    _, _, short_hash = _fingerprint_and_hash()
    path = log_path_for_config("lissajous_scan", short_hash)
    assert re.fullmatch(r"lissajous_scan_[0-9a-f]{6}\.log", path.name)


def test_write_result_log_path_and_config_hash(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(result_log, "LOG_DIR", tmp_path)
    sim = load_simulation_config("config/Environment.toml")
    mc_cfg = _sample_mc_cfg()
    path = write_result_log(
        strategy="lissajous_scan",
        optimize_section=_sample_optimize_section(),
        mc_cfg=mc_cfg,
        sim=sim,
        best_params={"s1_wx": 1.0},
        interrupted=False,
        elapsed=1.0,
        completed_trials=5,
        best_cost=1.0,
    )
    _, full_hash, short_hash = _fingerprint_and_hash(sim=sim, mc_cfg=mc_cfg)
    assert path == tmp_path / f"lissajous_scan_{short_hash}.log"
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"{CONFIG_HASH_PREFIX}{full_hash}\n")
    assert read_config_hash_from_log(path) == full_hash


def test_write_result_log_overwrites_same_config(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(result_log, "LOG_DIR", tmp_path)
    sim = load_simulation_config("config/Environment.toml")
    mc_cfg = _sample_mc_cfg()
    kwargs = dict(
        strategy="lissajous_scan",
        optimize_section=_sample_optimize_section(),
        mc_cfg=mc_cfg,
        sim=sim,
        best_params={"s1_wx": 1.0},
        interrupted=False,
        elapsed=1.0,
        completed_trials=5,
        best_cost=1.0,
    )
    path1 = write_result_log(**kwargs)
    kwargs["best_params"] = {"s1_wx": 9.9}
    kwargs["best_cost"] = 0.5
    path2 = write_result_log(**kwargs)
    assert path1 == path2
    assert "s1_wx = 9.9" in path2.read_text(encoding="utf-8")


def test_write_result_log_raises_on_hash_collision(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(result_log, "LOG_DIR", tmp_path)
    sim = load_simulation_config("config/Environment.toml")
    mc_cfg = _sample_mc_cfg()
    _, full_hash, short_hash = _fingerprint_and_hash(sim=sim, mc_cfg=mc_cfg)
    path = tmp_path / f"lissajous_scan_{short_hash}.log"
    other_hash = "a" * 64
    assert other_hash != full_hash
    path.write_text(f"{CONFIG_HASH_PREFIX}{other_hash}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config hash prefix collision"):
        write_result_log(
            strategy="lissajous_scan",
            optimize_section=_sample_optimize_section(),
            mc_cfg=mc_cfg,
            sim=sim,
            best_params={"s1_wx": 1.0},
            interrupted=False,
            elapsed=1.0,
            completed_trials=5,
            best_cost=1.0,
        )


def test_assert_safe_overwrite_allows_missing_hash(tmp_path: Path):
    path = tmp_path / "legacy.log"
    path.write_text("# --- Optimization completed ---\n", encoding="utf-8")
    assert_safe_overwrite(path, "b" * 64)
