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

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    result = run_scenario(config)
    print(format_summary(result))

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
