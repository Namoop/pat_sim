"""Post-hoc geometry metrics (not used for runtime strategy decisions)."""

from __future__ import annotations

from satellite.math3d import angle_between
from satellite.sda.satellite import Satellite


def partner_in_fov(satellite: Satellite) -> bool:
    """True when true partner lies within dish FOV of initial boresight."""
    toward = satellite.bench.toward_partner
    initial = satellite.bench.initial_boresight
    return angle_between(initial, toward) <= satellite.receiver.dish_fov


def pointing_offset(satellite: Satellite) -> float:
    """Angular offset (rad) from initial boresight to true partner."""
    return angle_between(
        satellite.bench.toward_partner,
        satellite.bench.initial_boresight,
    )
