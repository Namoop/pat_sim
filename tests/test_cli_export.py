"""CLI --export tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from montecarlo.__main__ import main

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture_mc(tmp_path: Path) -> Path:
    sim = tmp_path / "Environment.toml"
    sim.write_text((REPO_ROOT / "config/Environment.toml").read_text())
    mc = tmp_path / "MonteCarlo.toml"
    mc.write_text(
        (FIXTURES / "MonteCarlo_success.toml")
        .read_text()
        .replace('environment = "Environment.toml"', f'environment = "{sim.name}"')
    )
    return mc


def test_export_writes_default_name_from_run_number(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    mc = _fixture_mc(tmp_path)
    code = main([str(mc), "--export"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Exported scenario to config/_scenario_run1.toml" in out
    path = tmp_path / "config/_scenario_run1.toml"
    assert path.is_file()
    assert 'name = "run 1"' in path.read_text()


def test_export_custom_name(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    mc = _fixture_mc(tmp_path)
    code = main([str(mc), "--export", "replay"])
    assert code == 0
    path = tmp_path / "config/_scenario_replay.toml"
    assert path.is_file()
    assert 'name = "replay"' in path.read_text()
    assert "config/_scenario_replay.toml" in capsys.readouterr().out


def test_export_with_run_number(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    mc = _fixture_mc(tmp_path)
    code = main([str(mc), "--run", "2", "--export"])
    assert code == 0
    path = tmp_path / "config/_scenario_run2.toml"
    assert path.is_file()
    assert 'name = "run 2"' in path.read_text()


def test_export_invalid_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    mc = _fixture_mc(tmp_path)
    with pytest.raises(SystemExit):
        main([str(mc), "--export", "../bad"])


def test_export_with_visualize_only_writes_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    mc = _fixture_mc(tmp_path)
    calls: list[str] = []

    def fake_run_visualizer(session, **kwargs):
        calls.append("viz")
        return 0

    def fail_batch(*args, **kwargs):
        raise AssertionError("batch should not run when exporting")

    monkeypatch.setattr("visualize.run_visualizer", fake_run_visualizer)
    monkeypatch.setattr("montecarlo.__main__.run_monte_carlo", fail_batch)
    code = main([str(mc), "--export", "viz_case", "--visualize", "mag"])
    assert code == 0
    path = tmp_path / "config/_scenario_viz_case.toml"
    assert path.is_file()
    text = path.read_text()
    assert 'visualize = "mag"' in text
    assert calls == []


def test_export_skips_batch_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    mc = _fixture_mc(tmp_path)

    def fail_batch(*args, **kwargs):
        raise AssertionError("batch should not run when only exporting")

    monkeypatch.setattr("montecarlo.__main__.run_monte_carlo", fail_batch)
    code = main([str(mc), "--export"])
    assert code == 0


def test_export_requires_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    sim = tmp_path / "Environment.toml"
    sim.write_text((REPO_ROOT / "config/Environment.toml").read_text())
    mc = tmp_path / "MonteCarlo.toml"
    mc.write_text(
        """
[monte_carlo]
environment = "Environment.toml"
runs = 3
chain = ["minor_offset"]

[monte_carlo.error]
distribution = "uniform"
uniform.max = 1.0

[strategy.minor_offset]
max_spiral_radius = "fov"
""".replace("Environment.toml", sim.name)
    )
    with pytest.raises(SystemExit):
        main([str(mc), "--export"])
