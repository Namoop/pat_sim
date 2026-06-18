"""Build eye-view scene data from ScenarioResult at a given t."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from satellite.math.math3d import normalize
from satellite.physics.satellite import Satellite
from visualize.frames import direction_to_tangent_angles, point_in_disc

if TYPE_CHECKING:
    from scenario.run import ScenarioResult


SatelliteName = Literal["S1", "S2"]


@dataclass(frozen=True)
class EyeDisc:
    center_theta: float
    center_phi: float
    radius: float


@dataclass(frozen=True)
class EyeView:
    satellite: SatelliteName
    partner: tuple[float, float]
    beam: EyeDisc | None
    fov: EyeDisc | None
    partner_in_beam: bool
    partner_in_fov: bool
    is_transmitting: bool
    beam_director: tuple[float, float] | None = None


@dataclass(frozen=True)
class EyeScene:
    t: float
    s1: EyeView
    s2: EyeView
    capture_active: bool


def _satellite(result: ScenarioResult, name: SatelliteName) -> Satellite:
    return result.s1 if name == "S1" else result.s2


def _other_satellite(result: ScenarioResult, name: SatelliteName) -> Satellite:
    return result.s2 if name == "S1" else result.s1


def _partner_direction(viewer: Satellite, other: Satellite) -> object:
    return normalize(other.position - viewer.position)


def _eye_origin(sat: Satellite) -> object:
    return sat.bench.initial_boresight


def build_view(
    result: ScenarioResult,
    satellite: SatelliteName,
    t: float,
) -> EyeView:
    result.replay_to(t)
    sat = _satellite(result, satellite)
    origin = _eye_origin(sat)

    other = _other_satellite(result, satellite)
    partner_dir = _partner_direction(sat, other)
    partner = direction_to_tangent_angles(origin, partner_dir)

    fsm = sat.receiver.fsm
    effective_aim = fsm.effective_receive_boresight(sat.bench.bench_boresight)
    center_theta, center_phi = direction_to_tangent_angles(origin, effective_aim)
    alpha = result.config.satellite.alpha
    dish_fov = result.config.satellite.dish_fov

    beam = EyeDisc(center_theta, center_phi, alpha)
    fov = EyeDisc(center_theta, center_phi, dish_fov)

    partner_in_beam = point_in_disc(
        partner,
        (center_theta, center_phi),
        alpha,
    )
    partner_in_fov = point_in_disc(
        partner,
        (center_theta, center_phi),
        dish_fov,
    )

    scheduled, local_t = result.schedule.script_at(t)
    timeline = scheduled.script.s1 if satellite == "S1" else scheduled.script.s2
    beam_enabled, receiver_enabled = timeline.hardware_state_at(local_t)

    if not fsm.is_neutral():
        bd_theta, bd_phi = direction_to_tangent_angles(origin, sat.bench.bench_boresight)
        beam_director = (bd_theta, bd_phi)
    else:
        beam_director = None

    return EyeView(
        satellite=satellite,
        partner=partner,
        beam=beam if beam_enabled else None,
        fov=fov if receiver_enabled else None,
        partner_in_beam=partner_in_beam if beam_enabled else False,
        partner_in_fov=partner_in_fov if receiver_enabled else False,
        is_transmitting=beam_enabled,
        beam_director=beam_director,
    )


def build_scene(result: ScenarioResult, t: float) -> EyeScene:
    result.replay_to(t)
    s1 = build_view(result, "S1", t)
    s2 = build_view(result, "S2", t)

    capture = result.mutual_lock(t)

    return EyeScene(t=t, s1=s1, s2=s2, capture_active=capture)
