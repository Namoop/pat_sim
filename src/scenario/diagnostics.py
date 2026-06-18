"""Temporary replay/simulation profiling (remove after optimization)."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class SimReplayProfiler:
    """Cumulative timings across replay steps; reports per-step averages."""

    enabled: bool = False
    step_count: int = 0
    totals: dict[str, float] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, profile_replay: bool) -> SimReplayProfiler:
        env = os.environ.get("SATELLITE_SIM_PROFILE", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            enabled = False
        elif env in ("1", "true", "yes", "on"):
            enabled = True
        else:
            enabled = profile_replay
        return cls(enabled=enabled)

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
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.call_counts[name] = self.call_counts.get(name, 0) + 1

    def report(self, label: str = "replay") -> None:
        if not self.enabled:
            return
        print(self.format_summary(label), file=sys.stderr)

    def format_summary(self, label: str = "replay") -> str:
        if not self.enabled or not self.totals:
            return f"[sim profile {label}] (no samples)"
        total_ms = self.totals.get("replay_total", 0.0) * 1000
        header = f"[sim profile {label}] steps={self.step_count} replay_total={total_ms:.2f}ms"
        lines = [header]
        skip = {"replay_total"}
        for name in sorted(self.totals, key=lambda n: -self.totals[n]):
            if name in skip:
                continue
            total_s = self.totals[name]
            calls = self.call_counts[name]
            total_name_ms = total_s * 1000
            per_call = total_name_ms / calls if calls else 0.0
            per_step = total_name_ms / self.step_count if self.step_count else 0.0
            lines.append(
                f"  {name}: total={total_name_ms:.2f}ms "
                f"per_call={per_call:.4f}ms per_step={per_step:.4f}ms calls={calls}"
            )
        return "\n".join(lines)

    def format_overlay(self, label: str = "sim") -> str:
        if not self.enabled or not self.totals:
            return ""
        total_ms = self.totals.get("replay_total", 0.0) * 1000
        lines = [f"{label}  {self.step_count} steps  total {total_ms:.1f} ms"]
        skip = {"replay_total", "event_log"}
        for name in sorted(self.totals, key=lambda n: -self.totals[n])[:8]:
            if name in skip:
                continue
            per_step = (self.totals[name] / self.step_count * 1000) if self.step_count else 0.0
            lines.append(f"  {name}: {per_step:.3f} ms/step")
        return "\n".join(lines)
