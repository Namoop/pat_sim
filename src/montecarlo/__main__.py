"""CLI entry point for Monte Carlo batch runs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import satellite  # noqa: F401 — CUDA bootstrap

from montecarlo.export import export_scenario_toml, require_monte_carlo_seed
from montecarlo.run import load_monte_carlo_config
from montecarlo.run import format_monte_carlo_summary, run_monte_carlo

_EXPORT_NAME_RE = re.compile(r"^[\w.-]+$")


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
        choices=("3d", "eye", "mag"),
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
        "--run",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Starting run number for visualization (1-based, default: 1)",
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
    parser.add_argument(
        "--export",
        nargs="?",
        const="",
        default=None,
        metavar="NAME",
        help=(
            "Export the selected --run as config/_scenario_NAME.toml "
            "(default: config/_scenario_runN.toml); does not run the simulation"
        ),
    )
    args = parser.parse_args(argv)

    if args.autoplay is not None and args.autoplay <= 0:
        parser.error("--autoplay SPEED must be positive")

    if args.run is not None and args.run < 1:
        parser.error("--run NUMBER must be at least 1")

    if not args.config.exists():
        print(f"Monte Carlo config not found: {args.config}", file=sys.stderr)
        return 1

    mc = load_monte_carlo_config(args.config)

    viz_mode = args.visualize
    export_run = args.run if args.run is not None else 1

    if args.autoplay is not None and viz_mode is None:
        parser.error("--autoplay requires visualization (--visualize)")
    if args.run is not None and viz_mode is None and args.export is None:
        parser.error("--run requires visualization (--visualize) or export (--export)")

    if args.export is not None:
        try:
            require_monte_carlo_seed(args.config)
        except ValueError as exc:
            parser.error(str(exc))
        if export_run > mc.runs:
            parser.error(f"--run {export_run} exceeds configured runs ({mc.runs})")
        file_stem: str | None = None
        scenario_name: str | None = None
        if args.export != "":
            if not _EXPORT_NAME_RE.fullmatch(args.export):
                parser.error(
                    f"invalid export name {args.export!r}; "
                    "use letters, digits, underscores, dots, or hyphens"
                )
            file_stem = args.export
            scenario_name = args.export
        export_path = export_scenario_toml(
            mc=mc,
            mc_toml_path=args.config,
            run_number=export_run,
            file_stem=file_stem,
            scenario_name=scenario_name,
            visualize=viz_mode,
        )
        print(f"Exported scenario to {export_path}")
        return 0

    if viz_mode is not None:
        from visualize import run_visualizer
        from visualize.session import MonteCarloVizSession

        start_run = export_run
        if start_run > mc.runs:
            parser.error(f"--run {start_run} exceeds configured runs ({mc.runs})")

        session = MonteCarloVizSession(mc, start_run=start_run)
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
