"""Composite satellite with transmitter and receiver subsystems."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from scenario.types import ScenarioConfig, SatelliteInstanceConfig, default_beam_length
from satellite.math.math3d import Vec3
from satellite.physics.bench import OpticalBench
from satellite.physics.fsm import FastSteeringMirror
from satellite.physics.receiver import ReceiverSDA
from satellite.physics.transmitter import TransmitterSDA


@dataclass
class Satellite:
    name: str
    position: Vec3
    partner_actual: Vec3
    bench: OpticalBench
    transmitter: TransmitterSDA
    receiver: ReceiverSDA
    cos_alpha: float
    beam_length: float
    phase2_transmitter: TransmitterSDA | None = field(default=None)

    @property
    def believed_boresight(self) -> Vec3:
        return self.bench.initial_beam_boresight

    @staticmethod
    def build(
        name: str,
        instance: SatelliteInstanceConfig,
        partner_actual: Vec3,
        config: ScenarioConfig,
    ) -> Satellite:
        position = instance.position
        hw = config.satellite
        bench = OpticalBench(
            position=position,
            partner_position=partner_actual,
            bench_theta_offset=instance.bench_theta_offset,
            bench_phi_offset=instance.bench_phi_offset,
            max_beam_speed=hw.max_beam_speed,
            max_fsm_speed=hw.max_fsm_speed,
            max_fsm_radius=hw.max_fsm_radius,
            scan_envelope_ramp=hw.scan_envelope_ramp,
            scan_envelope_profile_id=hw.scan_envelope_profile_id,
        )
        fsm = FastSteeringMirror()
        beam_length = default_beam_length(position, partner_actual, config.simulation)
        transmitter = TransmitterSDA.from_boresight(
            position,
            bench.initial_beam_boresight,
            partner_actual,
            config.strategy.k,
            hw.alpha,
            beam_length,
            boresight_extension=config.simulation.boresight_extension,
        )
        receiver = ReceiverSDA(
            bench=bench,
            fsm=fsm,
            body_radius=hw.body_radius,
            dish_fov=hw.dish_fov,
        )
        return Satellite(
            name=name,
            position=position,
            partner_actual=partner_actual,
            bench=bench,
            transmitter=transmitter,
            receiver=receiver,
            cos_alpha=float(np.cos(hw.alpha)),
            beam_length=beam_length,
        )
