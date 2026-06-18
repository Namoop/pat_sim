"""Deprecated CLI entry point — use scenario or montecarlo instead."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "The satellite CLI was split into separate entry points:\n"
        "  python -m scenario   — single scenario run\n"
        "  python -m montecarlo — Monte Carlo batch\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
