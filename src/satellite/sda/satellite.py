"""Composite satellite with transmitter and receiver subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field

from satellite.config import ScenarioConfig, SatelliteInstanceConfig, default_beam_length
from satellite.math3d import Vec3
from satellite.sda.bench import OpticalBench
from satellite.sda.fsm import FastSteeringMirror
from satellite.sda.receiver import ReceiverSDA
from satellite.sda.transmitter import TransmitterSDA


@dataclass
class Satellite:
    name: str
    position: Vec3
    partner_actual: Vec3
    bench: OpticalBench
    transmitter: TransmitterSDA
    receiver: ReceiverSDA
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
            bench_slew_time=hw.bench_slew_time,
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
        )
