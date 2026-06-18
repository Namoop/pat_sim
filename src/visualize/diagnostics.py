"""Temporary per-frame timing helpers for visualizer profiling."""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class FrameProfiler:
    """Collect last-frame and EMA timings for viz hot paths."""

    enabled: bool = True
    print_every: int = 30
    ema_alpha: float = 0.25
    _frame: int = 0
    _last: dict[str, float] = field(default_factory=dict)
    _ema: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @classmethod
    def from_env(cls, config_enabled: bool) -> FrameProfiler:
        env = os.environ.get("SATELLITE_VIZ_PROFILE", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            enabled = False
        elif env in ("1", "true", "yes", "on"):
            enabled = True
        else:
            enabled = config_enabled
        every = int(os.environ.get("SATELLITE_VIZ_PROFILE_EVERY", "30"))
        return cls(enabled=enabled, print_every=max(1, every))

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._last[name] = elapsed
            if name in self._ema:
                a = self.ema_alpha
                self._ema[name] = a * elapsed + (1.0 - a) * self._ema[name]
            else:
                self._ema[name] = elapsed

    def set_gauge(self, name: str, value: float) -> None:
        """Record a scalar gauge (e.g. replay step count), not a duration."""
        if not self.enabled:
            return
        self._last[name] = value
        self._ema[name] = value

    def increment(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self._counts[name] += amount

    def record(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        self._last[name] = seconds
        if name in self._ema:
            a = self.ema_alpha
            self._ema[name] = a * seconds + (1.0 - a) * self._ema[name]
        else:
            self._ema[name] = seconds

    def end_frame(self) -> None:
        if not self.enabled:
            return
        self._frame += 1
        if self.print_every > 0 and self._frame % self.print_every == 0:
            print(self.format_summary(), file=sys.stderr)

    def format_overlay(self) -> str:
        if not self.enabled or not self._last:
            return ""
        total = self._last.get("frame_total", 0.0)
        lines = [f"Frame {self._frame}  total {total * 1000:.1f} ms"]
        for name, _ in sorted(self._ema.items(), key=lambda item: -item[1]):
            if name in ("frame_total", "replay_steps"):
                continue
            last_ms = self._last.get(name, 0.0) * 1000
            ema_ms = self._ema[name] * 1000
            lines.append(f"  {name}: {last_ms:.1f} ms  (ema {ema_ms:.1f})")
        steps = int(self._last.get("replay_steps", 0))
        if steps:
            lines.append(f"  replay_steps: {steps}")
        return "\n".join(lines)

    def format_summary(self) -> str:
        if not self._last:
            return "[viz profile] (no samples)"
        total = self._last.get("frame_total", 0.0) * 1000
        parts = [
            f"[viz profile frame {self._frame}] total={total:.1f}ms",
        ]
        for name, ema in sorted(self._ema.items(), key=lambda item: -item[1]):
            if name in ("frame_total", "replay_steps"):
                continue
            parts.append(f"{name}={ema * 1000:.1f}ms")
        steps = int(self._last.get("replay_steps", 0))
        if steps:
            parts.append(f"replay_steps={steps}")
        return "  ".join(parts)
