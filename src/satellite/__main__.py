"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from satellite.config import load_single_scenario
from satellite.monte_carlo import format_monte_carlo_summary, run_monte_carlo_from_path
from satellite.scenario import format_summary, run_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a satellite SDA communication scenario.",
    )
    parser.add_argument(
        "--monte-carlo",
        type=Path,
        default=None,
        help="Run Monte Carlo batch from MonteCarlo.toml",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("default.toml"),
        help="Scenario instance TOML (bench offsets, optional distance override)",
    )
    parser.add_argument(
        "--simulation",
        type=Path,
        default=Path("Simulation.toml"),
        help="Simulation base TOML (physics, timing, visualization)",
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        default=Path("MonteCarlo.toml"),
        help="Strategy chain TOML (strategy section; default: MonteCarlo.toml)",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const="3d",
        choices=("3d", "map"),
        default=None,
        help=(
            "Open interactive visualization: 3d (PyVista) or map (angular θ/φ view). "
            "Use --visualize or --visualize 3d for the 3D window."
        ),
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.0,
        help="Starting time q for visualization (default: 0)",
    )
    args = parser.parse_args(argv)

    if args.monte_carlo is not None:
        if not args.monte_carlo.exists():
            print(f"Monte Carlo config not found: {args.monte_carlo}", file=sys.stderr)
            return 1
        summary = run_monte_carlo_from_path(str(args.monte_carlo))
        print(format_monte_carlo_summary(summary))
        return 0

    for label, path in (
        ("Scenario", args.scenario),
        ("Simulation", args.simulation),
        ("Strategy", args.strategy),
    ):
        if not path.exists():
            print(f"{label} config not found: {path}", file=sys.stderr)
            return 1

    config = load_single_scenario(args.scenario, args.simulation, args.strategy)
    result = run_scenario(config)
    print(format_summary(result))

    exit_code = 0 if result.success else 1

    viz_mode = args.visualize
    if viz_mode is None and config.visualization.enabled:
        viz_mode = "3d"

    if viz_mode == "3d":
        from satellite.visualize import run_visualizer

        try:
            run_visualizer(result, start_q=args.q)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130
    elif viz_mode == "map":
        from satellite.mapviz import run_map_visualizer

        try:
            run_map_visualizer(result, start_q=args.q)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
