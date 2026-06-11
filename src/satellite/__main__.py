"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from satellite.config import (
    load_monte_carlo_config,
    load_simulation_config,
    load_single_scenario,
)
from satellite.monte_carlo import format_monte_carlo_summary, run_monte_carlo
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
            "Open unified visualization window; optional 3d or map picks the initial tab. "
            "Use --visualize or --visualize 3d for the 3D tab first."
        ),
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.0,
        help="Starting time q for visualization (default: 0)",
    )
    parser.add_argument(
        "--autoplay",
        nargs="?",
        const=1.0,
        type=float,
        default=None,
        metavar="SPEED",
        help=(
            "Monte Carlo visualization: auto-scrub at play speed × SPEED "
            "(default 1), advance to the next run when each finishes; "
            "stops on Pause or when done"
        ),
    )
    args = parser.parse_args(argv)

    if args.autoplay is not None:
        if args.monte_carlo is None:
            parser.error("--autoplay requires --monte-carlo")
        if args.autoplay <= 0:
            parser.error("--autoplay SPEED must be positive")

    if args.monte_carlo is not None:
        if not args.monte_carlo.exists():
            print(f"Monte Carlo config not found: {args.monte_carlo}", file=sys.stderr)
            return 1
        mc = load_monte_carlo_config(args.monte_carlo)
        sim = load_simulation_config(mc.simulation_path)
        viz_mode = args.visualize
        if args.autoplay is not None and viz_mode is None and not sim.visualization.enabled:
            parser.error(
                "--autoplay requires visualization (--visualize or visualization.enabled)"
            )
        if viz_mode is not None or sim.visualization.enabled:
            from satellite.visualize import run_visualizer
            from satellite.visualize.session import MonteCarloVizSession

            session = MonteCarloVizSession(mc)
            try:
                run_visualizer(
                    session,
                    default_tab=viz_mode or "3d",
                    start_q=args.q,
                    autoplay_speed=args.autoplay,
                )
            except KeyboardInterrupt:
                print("Interrupted.", file=sys.stderr)
                return 130
            return 0

        summary = run_monte_carlo(mc)
        print(format_monte_carlo_summary(summary))
        if summary.interrupted:
            print("Interrupted.", file=sys.stderr)
            return 130
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
    if viz_mode is not None or config.visualization.enabled:
        from satellite.visualize import run_visualizer
        from satellite.visualize.session import SingleResultSession

        try:
            run_visualizer(
                SingleResultSession(result),
                default_tab=viz_mode or "3d",
                start_q=args.q,
            )
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
