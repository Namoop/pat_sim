"""Benchmark mapviz QPainter render path (headless Qt)."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from satellite.config import load_single_scenario
from satellite.mapviz.panel_widget import AngularMapPanel
from satellite.mapviz.scene import build_scene
from satellite.scenario import run_scenario


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_benchmark(
    scenario_path: str,
    simulation_path: str | None = None,
    *,
    samples: int = 1000,
    seed: int = 0,
) -> int:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    config = load_single_scenario(scenario_path, simulation_path)
    result = run_scenario(config)
    result.ensure_replay_timeline()
    total_t = result.playable_t_end
    map_cfg = config.map_viz

    app = QApplication.instance() or QApplication([])
    panel_s1 = AngularMapPanel(axis_limit=map_cfg.axis_limit)
    panel_s2 = AngularMapPanel(axis_limit=map_cfg.axis_limit)
    for panel in (panel_s1, panel_s2):
        panel.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        panel.resize(520, 480)
    app.processEvents()

    rng = np.random.default_rng(seed)
    ts = rng.uniform(0.0, total_t, size=samples)

    replay_times: list[float] = []
    build_times: list[float] = []
    paint_times: list[float] = []
    frame_times: list[float] = []

    for t in ts:
        t = float(t)
        t0 = time.perf_counter()

        t_replay = time.perf_counter()
        result.replay_to(t)
        replay_times.append(time.perf_counter() - t_replay)

        t_build = time.perf_counter()
        scene = build_scene(result, t)
        build_times.append(time.perf_counter() - t_build)

        t_paint = time.perf_counter()
        if panel_s1.set_panel(scene.s1, partner_label="S2"):
            panel_s1.repaint()
        if panel_s2.set_panel(scene.s2, partner_label="S1"):
            panel_s2.repaint()
        paint_times.append(time.perf_counter() - t_paint)

        frame_times.append(time.perf_counter() - t0)

    def report(name: str, vals: list[float]) -> None:
        vals_ms = sorted(v * 1000.0 for v in vals)
        print(
            f"  {name}: p50={_percentile(vals_ms, 50):.2f} ms  "
            f"p95={_percentile(vals_ms, 95):.2f} ms  "
            f"max={vals_ms[-1]:.2f} ms"
        )

    print(f"Mapviz QPainter benchmark ({samples} samples, t in [0, {total_t:.2f}])")
    report("replay", replay_times)
    report("build_scene", build_times)
    report("paint", paint_times)
    report("frame_total", frame_times)

    p95_paint = _percentile(sorted(t * 1000 for t in paint_times), 95)
    if p95_paint > 2.0:
        print(f"WARN: paint p95 {p95_paint:.2f} ms exceeds 2 ms target", file=sys.stderr)
        return 1
    print("OK: paint p95 within 2 ms target")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark mapviz QPainter render.")
    parser.add_argument(
        "--scenario",
        default="config/Scenario.toml",
        help="Scenario instance TOML",
    )
    parser.add_argument(
        "--simulation",
        default=None,
        help="Simulation base TOML",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of random t samples",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run_benchmark(
        args.scenario,
        args.simulation,
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
