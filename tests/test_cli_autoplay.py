"""CLI --autoplay validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from montecarlo.__main__ import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_autoplay_requires_positive_speed():
    with pytest.raises(SystemExit):
        main(["config/_montecarlo_minor.toml", "--autoplay", "0"])


def test_autoplay_without_speed_defaults_to_one(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    captured: dict = {}

    def fake_run_visualizer(session, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("visualize.run_visualizer", fake_run_visualizer)
    main([str(repo / "config/_montecarlo_minor.toml"), "--visualize", "--autoplay"])
    assert captured["autoplay_speed"] == 1.0


def test_autoplay_requires_visualization(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sim = tmp_path / "Environment.toml"
    sim.write_text((repo / "config/Environment.toml").read_text())
    mc = tmp_path / "_montecarlo_minor.toml"
    mc.write_text(
        (FIXTURES / "MonteCarlo_success.toml")
        .read_text()
        .replace('environment = "Environment.toml"', f'environment = "{sim.name}"')
    )
    with pytest.raises(SystemExit):
        main([str(mc), "--autoplay", "1"])
