"""Scenario orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from satellite.config import ScenarioConfig, resolve_actual_position
from satellite.detection import (
    alignment_dot,
    beam_hits_dish_at_q,
    scan_dish_hits_up_to,
)
from satellite.geometry import actual_target_direction, believed_direction
from satellite.math3d import Vec3, distance, norm
from satellite.sda.receiver import ReceiverSDA
from satellite.sda.transmitter import TransmitterSDA


def replay_dish_tracking(
    receiver: ReceiverSDA,
    transmitter: TransmitterSDA,
    in_cone_at_q: Callable[[float], bool],
    q_end: float,
    q_step: float,
) -> None:
    """Replay dish orientation from q=0 through q_end."""
    receiver.reset_dish_tracking()
    q = 0.0
    while q <= q_end + 1e-12:
        receiver.observe_beam(
            in_cone_at_q(q),
            transmitter.boresight_at(q),
            q_step,
        )
        q += q_step


@dataclass
class ScenarioResult:
    config: ScenarioConfig
    p1: Vec3
    p2: Vec3
    pt: Vec3
    transmitter: TransmitterSDA
    receiver: ReceiverSDA
    target_direction: Vec3
    hit: bool
    hit_at_q: float | None
    alignment_at_q_max: float
    in_cone_at_q: Callable[[float], bool]

    @property
    def believed_distance(self) -> float:
        return distance(self.p1, self.p2)

    @property
    def believed_direction(self) -> Vec3:
        return believed_direction(self.p1, self.p2)


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Run headless SDA scan; no mesh allocation."""
    p1 = config.positions.p1
    p2 = config.positions.p2
    pt = resolve_actual_position(config)

    transmitter = TransmitterSDA(
        p1,
        p2,
        pt,
        config.sda.k,
        config.sda.alpha,
        config.simulation.beam_length,
    )
    receiver = ReceiverSDA(
        p1,
        pt,
        config.sda.gamma,
        config.sda.beta,
        config.sda.omega_r,
        config.sda.L_r,
        dish_theta_offset=config.receiver.dish_theta_offset,
        dish_phi_offset=config.receiver.dish_phi_offset,
        body_radius=config.receiver.body_radius,
        dish_fov=config.receiver.dish_fov,
        dish_slew_time=config.receiver.dish_slew_time,
    )

    target_direction = actual_target_direction(p1, pt)
    alpha = config.sda.alpha
    beam_length = transmitter.beam_length
    dish_fov = config.receiver.dish_fov

    def check_dish_hit(q: float) -> bool:
        return beam_hits_dish_at_q(
            q,
            p1,
            receiver.dish_mount,
            receiver.dish.boresight,
            dish_fov,
            transmitter.boresight_at,
            alpha,
            beam_length,
        )

    hit, hit_at_q = scan_dish_hits_up_to(
        config.simulation.q_max,
        config.simulation.q_step,
        p1,
        receiver.initial_dish_mount,
        receiver.initial_dish_boresight,
        dish_fov,
        transmitter.boresight_at,
        alpha,
        beam_length,
    )

    alignment_at_q_max = alignment_dot(
        target_direction,
        transmitter.boresight_at(config.simulation.q_max),
    )

    replay_dish_tracking(
        receiver,
        transmitter,
        check_dish_hit,
        config.simulation.q_max,
        config.simulation.q_step,
    )

    return ScenarioResult(
        config=config,
        p1=p1,
        p2=p2,
        pt=pt,
        transmitter=transmitter,
        receiver=receiver,
        target_direction=target_direction,
        hit=hit,
        hit_at_q=hit_at_q,
        alignment_at_q_max=alignment_at_q_max,
        in_cone_at_q=check_dish_hit,
    )


def format_summary(result: ScenarioResult) -> str:
    """Human-readable headless run summary."""
    tx = result.transmitter
    lines = [
        f"Scenario: {result.config.name}",
        f"Believed distance d = {result.believed_distance:.6f}",
        f"Believed direction D = {tuple(round(float(x), 8) for x in result.believed_direction)}",
        f"w = {tx.w:.10f}",
        f"U_t = {tuple(round(float(x), 8) for x in result.target_direction)}",
        f"|P_t - P_1| = {norm(result.pt - result.p1):.6f}",
        f"Hit: {'yes' if result.hit else 'no'}",
    ]
    if result.hit_at_q is not None:
        lines.append(f"Hit at q = {result.hit_at_q:.4f}")
    if result.receiver.dish.has_seen_beam and result.receiver.dish.incident_angle is not None:
        lines.append(
            f"Dish incident angle at first detection = "
            f"{np.degrees(result.receiver.dish.incident_angle):.3f} deg"
        )
    lines.append(
        f"Alignment at q_max = {result.alignment_at_q_max:.6f} "
        f"(threshold cos(alpha) = {np.cos(result.config.sda.alpha):.6f})"
    )
    q_max = result.config.simulation.q_max
    lines.append(
        f"Offsets at q_max: theta_off = {tx.theta_offset(q_max):.6f}, "
        f"phi_off = {tx.phi_offset(q_max):.6f}"
    )
    return "\n".join(lines)
