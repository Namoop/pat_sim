"""Composite satellite with transmitter and receiver subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field

from satellite.config import ScenarioConfig, SatelliteInstanceConfig, default_beam_length
from satellite.math3d import Vec3
from satellite.sda.body import SpacecraftBody
from satellite.sda.receiver import ReceiverSDA
from satellite.sda.transmitter import TransmitterSDA


@dataclass
class Satellite:
    name: str
    position: Vec3
    partner_actual: Vec3
    body: SpacecraftBody
    transmitter: TransmitterSDA
    receiver: ReceiverSDA
    phase2_transmitter: TransmitterSDA | None = field(default=None)

    @property
    def believed_boresight(self) -> Vec3:
        return self.body.initial_beam_boresight

    @staticmethod
    def build(
        name: str,
        instance: SatelliteInstanceConfig,
        partner_actual: Vec3,
        config: ScenarioConfig,
    ) -> Satellite:
        position = instance.position
        hw = config.satellite
        body = SpacecraftBody(
            position=position,
            partner_position=partner_actual,
            body_theta_offset=instance.body_theta_offset,
            body_phi_offset=instance.body_phi_offset,
            dish_theta_offset=instance.dish_theta_offset,
            dish_phi_offset=instance.dish_phi_offset,
            beam_theta_offset=instance.beam_theta_offset,
            beam_phi_offset=instance.beam_phi_offset,
            body_slew_time=hw.dish_slew_time,
        )
        beam_length = default_beam_length(position, partner_actual, config.simulation)
        transmitter = TransmitterSDA.from_boresight(
            position,
            body.initial_beam_boresight,
            partner_actual,
            config.sda.k,
            hw.alpha,
            beam_length,
            boresight_extension=config.simulation.boresight_extension,
        )
        receiver = ReceiverSDA(
            body=body,
            gamma=config.sda.gamma,
            beta=config.sda.beta,
            omega_r=config.sda.omega_r,
            l_r=config.sda.L_r,
            body_radius=hw.body_radius,
            dish_fov=hw.dish_fov,
        )
        return Satellite(
            name=name,
            position=position,
            partner_actual=partner_actual,
            body=body,
            transmitter=transmitter,
            receiver=receiver,
        )


def build_phase2_transmitter(
    s2: Satellite,
    s1: Satellite,
    beam_boresight_end: Vec3,
    config: ScenarioConfig,
) -> TransmitterSDA:
    """S2 phase-2 spiral centered on end-of-phase-1 beam boresight."""
    beam_length = default_beam_length(s2.position, s1.position, config.simulation)
    return TransmitterSDA.from_boresight(
        s2.position,
        beam_boresight_end,
        s1.position,
        config.sda.k,
        config.satellite.alpha,
        beam_length,
        boresight_extension=config.simulation.boresight_extension,
    )
