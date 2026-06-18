"""CLI entry point for Monte Carlo batch runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import satellite  # noqa: F401 — CUDA bootstrap

from satellite.sim.config import load_monte_carlo_config
from satellite.sim.monte_carlo import format_monte_carlo_summary, run_monte_carlo


def _normalize_viz_mode(viz_mode: str | None) -> str | None:
    if viz_mode == "map":
        return "eye"
    if viz_mode == "dist":
        return "mag"
    return viz_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Monte Carlo batch of satellite SDA communication scenarios.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("config/MonteCarlo.toml"),
        help="Monte Carlo batch TOML (default: config/MonteCarlo.toml)",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const="3d",
        choices=("3d", "eye", "mag", "map", "dist"),
        default=None,
        help=(
            "Open unified visualization window; optional 3d, eye, or mag picks the initial tab. "
            "Use --visualize or --visualize 3d for the 3D tab first."
        ),
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.0,
        help="Starting time t for visualization (default: 0)",
    )
    parser.add_argument(
        "--autoplay",
        nargs="?",
        const=1.0,
        type=float,
        default=None,
        metavar="SPEED",
        help=(
            "Visualization: auto-scrub at play speed × SPEED "
            "(default 1), advance to the next run when each finishes; "
            "stops on Pause or when done"
        ),
    )
    args = parser.parse_args(argv)

    if args.autoplay is not None and args.autoplay <= 0:
        parser.error("--autoplay SPEED must be positive")

    if not args.config.exists():
        print(f"Monte Carlo config not found: {args.config}", file=sys.stderr)
        return 1

    mc = load_monte_carlo_config(args.config)

    viz_mode = _normalize_viz_mode(args.visualize)
    if args.autoplay is not None and viz_mode is None:
        parser.error("--autoplay requires visualization (--visualize)")

    if viz_mode is not None:
        from satellite.visualize import run_visualizer
        from satellite.visualize.session import MonteCarloVizSession

        session = MonteCarloVizSession(mc)
        try:
            run_visualizer(
                session,
                default_tab=viz_mode,
                start_t=args.t,
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


if __name__ == "__main__":
    raise SystemExit(main())
