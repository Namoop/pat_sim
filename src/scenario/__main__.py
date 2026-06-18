"""CLI entry point for single scenario runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import satellite  # noqa: F401 — CUDA bootstrap

from scenario.run import load_single_scenario
from scenario.run import format_summary, run_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a satellite SDA communication scenario.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("config/Scenario.toml"),
        help="Scenario instance TOML (default: config/Scenario.toml)",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=None,
        help="Environment base TOML (physics, timing, visualization; default: config/Environment.toml or scenario-defined)",
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
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Scenario config not found: {args.config}", file=sys.stderr)
        return 1

    if args.environment is not None and not args.environment.exists():
        print(f"Environment config not found: {args.environment}", file=sys.stderr)
        return 1

    config = load_single_scenario(args.config, args.environment)
    result = run_scenario(config)
    print(format_summary(result))

    exit_code = 0 if result.success else 1

    viz_mode = args.visualize
    if viz_mode is None:
        viz_mode = config.visualize

    if viz_mode is not None:
        from visualize import run_visualizer
        from visualize.session import SingleResultSession

        try:
            run_visualizer(
                SingleResultSession(result),
                default_tab=viz_mode,
                start_t=args.t,
            )
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
