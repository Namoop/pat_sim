"""Monte Carlo scenario export tests."""

from __future__ import annotations

from pathlib import Path

from dataclasses import replace

import numpy as np
import pytest

from montecarlo.export import (
    build_scenario_export_text,
    default_export_file_stem,
    default_export_scenario_name,
    export_scenario_path,
    export_scenario_toml,
    monte_carlo_run_offsets,
    require_monte_carlo_seed,
)
from montecarlo.run import load_monte_carlo_config, run_monte_carlo_single
from scenario.run import load_single_scenario
from satellite.config import load_simulation_config

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _mc_fixture():
    mc = load_monte_carlo_config(FIXTURES / "MonteCarlo_success.toml")
    return replace(mc, simulation_path=(REPO_ROOT / "config/Environment.toml").resolve())


def test_export_scenario_path_name():
    assert export_scenario_path("run3") == Path("config/_scenario_run3.toml")
    assert export_scenario_path("my-run") == Path("config/_scenario_my-run.toml")


def test_default_export_names():
    assert default_export_file_stem(5) == "run5"
    assert default_export_scenario_name(5) == "run 5"


def test_export_includes_only_chain_strategies():
    mc = _mc_fixture()
    mc_path = FIXTURES / "MonteCarlo_success.toml"
    text = build_scenario_export_text(
        mc=mc,
        mc_toml_path=mc_path,
        run_number=1,
        scenario_name="run 1",
        export_path=REPO_ROOT / "config/_scenario_run1.toml",
        visualize=None,
    )
    assert 'name = "run 1"' in text
    assert "[strategy.minor_offset]" in text
    assert "[strategy.single_miss]" in text
    assert "[strategy.asymmetric_swap]" not in text
    assert "chain = " in text


def test_export_includes_visualize_and_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = tmp_path / "Environment.toml"
    sim.write_text((REPO_ROOT / "config/Environment.toml").read_text())
    mc_toml = tmp_path / "MonteCarlo.toml"
    mc_toml.write_text(
        f"""
[monte_carlo]
environment = "{sim.name}"
seed = 7
runs = 3
chain = ["minor_offset"]
simulation.distance = 750.0

[monte_carlo.error]
distribution = "uniform"
uniform.max = 1.0

[strategy.minor_offset]
max_spiral_radius = "fov"

[strategy.single_miss]
a_spiral_radius = 50.0
b_spiral_radius = 50.0
"""
    )
    mc = load_monte_carlo_config(mc_toml)
    text = build_scenario_export_text(
        mc=mc,
        mc_toml_path=mc_toml,
        run_number=2,
        scenario_name="eye_case",
        export_path=export_scenario_path("eye_case"),
        visualize="eye",
    )
    assert 'visualize = "eye"' in text
    assert "simulation.distance = 750" in text
    assert "[strategy.single_miss]" not in text


def test_exported_scenario_matches_monte_carlo_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    sim = config_dir / "Environment.toml"
    sim.write_text((REPO_ROOT / "config/Environment.toml").read_text())
    mc_path = FIXTURES / "MonteCarlo_success.toml"
    mc = load_monte_carlo_config(mc_path)
    mc = replace(mc, simulation_path=sim.resolve())

    export_path = export_scenario_toml(
        mc=mc,
        mc_toml_path=mc_path,
        run_number=3,
    )
    assert export_path.name == "_scenario_run3.toml"
    exported = load_single_scenario(export_path, sim)
    sim_bundle = load_simulation_config(sim)
    seed_rng = np.random.default_rng(mc.seed)
    seeds = seed_rng.integers(0, 2**32 - 1, size=mc.runs).tolist()
    expected = run_monte_carlo_single(mc, sim_bundle, int(seeds[2]), 2)

    np.testing.assert_allclose(
        exported.s1.bench_theta_offset,
        expected.s1_theta,
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        exported.s2.bench_phi_offset,
        expected.s2_phi,
        rtol=0,
        atol=1e-9,
    )
    assert exported.strategy.chain == mc.chain
    assert exported.visualize is None


def test_monte_carlo_run_offsets_invalid_run():
    mc = _mc_fixture()
    with pytest.raises(ValueError, match="run_number must be between"):
        monte_carlo_run_offsets(mc, 0)
    with pytest.raises(ValueError, match="run_number must be between"):
        monte_carlo_run_offsets(mc, mc.runs + 1)


def test_require_monte_carlo_seed(tmp_path):
    mc_path = tmp_path / "MonteCarlo.toml"
    mc_path.write_text(
        """
[monte_carlo]
runs = 1
chain = ["minor_offset"]
"""
    )
    with pytest.raises(ValueError, match="monte_carlo.seed is required"):
        require_monte_carlo_seed(mc_path)

    mc_path.write_text(
        """
[monte_carlo]
seed = 0
runs = 1
chain = ["minor_offset"]
"""
    )
    require_monte_carlo_seed(mc_path)
