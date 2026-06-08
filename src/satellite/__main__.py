"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from satellite.config import load_config
from satellite.scenario import format_summary, run_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a satellite SDA communication scenario.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scenario.toml"),
        help="Path to scenario TOML file (default: scenario.toml)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open interactive 3D visualization window",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.0,
        help="Starting time q for visualization (default: 0)",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    result = run_scenario(config)
    print(format_summary(result))

    visualize = args.visualize or config.visualization.enabled
    if visualize:
        from satellite.visualize import run_visualizer

        try:
            run_visualizer(result, start_q=args.q)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
