"""CLI --record tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenario.__main__ import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_record_requires_visualization(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sim = tmp_path / "Environment.toml"
    sim.write_text((repo / "config/Environment.toml").read_text())
    scenario = tmp_path / "Scenario.toml"
    scenario.write_text(
        f"""
[scenario]
name = "no_viz"
environment = "{sim.name}"
chain = ["minor_offset"]

[s1]
bench_theta_offset = 1.0
bench_phi_offset = 1.0

[s2]
bench_theta_offset = 2.0
bench_phi_offset = 1.5

[strategy.minor_offset]
max_spiral_radius = "fov"
"""
    )
    with pytest.raises(SystemExit):
        main([str(scenario), "--record"])


def test_record_must_be_positive():
    with pytest.raises(SystemExit):
        main([str(REPO_ROOT / "config/Scenario.toml"), "--visualize", "--record", "0"])


def test_record_passes_stride_to_visualizer(monkeypatch):
    captured: dict = {}

    def fake_run_visualizer(session, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("visualize.run_visualizer", fake_run_visualizer)
    monkeypatch.setattr(
        "scenario.__main__.run_scenario",
        lambda config: type(
            "R",
            (),
            {
                "success": True,
                "config": config,
                "playable_t_end": 1.0,
            },
        )(),
    )
    monkeypatch.setattr("scenario.__main__.format_summary", lambda r: "ok")
    main([str(REPO_ROOT / "config/Scenario.toml"), "--visualize", "--record", "3"])
    assert captured["record_stride"] == 3


def test_record_default_stride(monkeypatch):
    captured: dict = {}

    def fake_run_visualizer(session, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("visualize.run_visualizer", fake_run_visualizer)
    monkeypatch.setattr(
        "scenario.__main__.run_scenario",
        lambda config: type(
            "R",
            (),
            {
                "success": True,
                "config": config,
                "playable_t_end": 1.0,
            },
        )(),
    )
    monkeypatch.setattr("scenario.__main__.format_summary", lambda r: "ok")
    main([str(REPO_ROOT / "config/Scenario.toml"), "--visualize", "--record"])
    assert captured["record_stride"] == 2
