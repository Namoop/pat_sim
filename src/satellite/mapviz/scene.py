"""Build angular map panels from ScenarioResult at a given q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from satellite.mapviz.frames import (
    direction_to_tangent_angles,
    point_in_disc,
    spiral_trail_in_map,
)
from satellite.math3d import normalize
from satellite.schedule import SearchPhase, SearchSchedule
from satellite.sda.satellite import Satellite
from satellite.sda.transmitter import TransmitterSDA

if TYPE_CHECKING:
    from satellite.scenario import ScenarioResult


SatelliteName = Literal["S1", "S2"]


@dataclass(frozen=True)
class MapDisc:
    center_theta: float
    center_phi: float
    radius: float


@dataclass(frozen=True)
class MapPanel:
    satellite: SatelliteName
    partner: tuple[float, float]
    beam: MapDisc | None
    fov: MapDisc | None
    spiral_trail: np.ndarray
    partner_in_beam: bool
    partner_in_fov: bool
    is_transmitting: bool
    phase_label: str


@dataclass(frozen=True)
class MapScene:
    q: float
    s1: MapPanel
    s2: MapPanel
    capture_active: bool


def _satellite(result: ScenarioResult, name: SatelliteName) -> Satellite:
    return result.s1 if name == "S1" else result.s2


def _other_satellite(result: ScenarioResult, name: SatelliteName) -> Satellite:
    return result.s2 if name == "S1" else result.s1


def _partner_direction(viewer: Satellite, other: Satellite) -> np.ndarray:
    return normalize(other.position - viewer.position)


def build_panel(
    result: ScenarioResult,
    satellite: SatelliteName,
    q: float,
    *,
    spiral_trail_steps: int,
) -> MapPanel:
    sat = _satellite(result, satellite)
    phase, _local_q = result.schedule.phase_at(q)
    origin = sat.bench.toward_partner

    other = _other_satellite(result, satellite)
    partner_dir = _partner_direction(sat, other)
    partner = direction_to_tangent_angles(origin, partner_dir)

    is_tx = SearchSchedule.transmitting_satellite(phase) == satellite
    tx: TransmitterSDA | None = None
    tx_local_q = 0.0
    if is_tx:
        tx = result.active_transmitter(q)
        tx_local_q = result.local_q(q)
    else:
        tx = sat.transmitter

    beam: MapDisc | None = None
    spiral = np.empty((0, 2), dtype=np.float64)
    if tx is not None:
        local_for_beam = tx_local_q if is_tx else 0.0
        beam_dir = tx.boresight_at(local_for_beam)
        beam_theta, beam_phi = direction_to_tangent_angles(origin, beam_dir)
        beam = MapDisc(
            beam_theta,
            beam_phi,
            result.config.satellite.alpha,
        )
        if is_tx:
            spiral = spiral_trail_in_map(
                origin,
                tx,
                tx_local_q,
                spiral_trail_steps,
            )

    fov_dir = sat.receiver.dish_boresight
    fov_theta, fov_phi = direction_to_tangent_angles(origin, fov_dir)
    fov = MapDisc(
        fov_theta,
        fov_phi,
        result.config.satellite.dish_fov,
    )

    partner_in_beam = beam is not None and point_in_disc(
        partner,
        (beam.center_theta, beam.center_phi),
        beam.radius,
    )
    partner_in_fov = point_in_disc(
        partner,
        (fov.center_theta, fov.center_phi),
        fov.radius,
    )

    phase_label = "Phase 1" if phase is SearchPhase.S1_TRANSMIT else "Phase 2"

    return MapPanel(
        satellite=satellite,
        partner=partner,
        beam=beam,
        fov=fov,
        spiral_trail=spiral,
        partner_in_beam=partner_in_beam,
        partner_in_fov=partner_in_fov,
        is_transmitting=is_tx,
        phase_label=phase_label,
    )


def build_scene(
    result: ScenarioResult,
    q: float,
    *,
    spiral_trail_steps: int,
) -> MapScene:
    s1 = build_panel(result, "S1", q, spiral_trail_steps=spiral_trail_steps)
    s2 = build_panel(result, "S2", q, spiral_trail_steps=spiral_trail_steps)

    phase, _ = result.schedule.phase_at(q)
    if phase is SearchPhase.S1_TRANSMIT:
        capture = s1.partner_in_beam and s2.partner_in_fov
    else:
        capture = s2.partner_in_beam and s1.partner_in_fov

    return MapScene(q=q, s1=s1, s2=s2, capture_active=capture)
