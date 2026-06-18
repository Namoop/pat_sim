"""Receiver acquisition state for bench + FSM tracking."""

from __future__ import annotations

from dataclasses import dataclass

from satellite.math.math3d import Vec3


@dataclass
class AcquisitionState:
    has_seen_beam: bool = False
    incident_angle: float | None = None
    track_target: Vec3 | None = None
    bench_slew_rate: float | None = None
    fsm_locked: bool = False
    slew_complete: bool = False
