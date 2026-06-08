"""Scenario orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from satellite.config import ScenarioConfig, resolve_actual_position
from satellite.detection import alignment_dot, in_cone_at_q, scan_hits_up_to
from satellite.geometry import actual_target_direction, believed_direction
from satellite.math3d import Vec3, distance, norm
from satellite.sda.receiver import ReceiverSDA
from satellite.sda.transmitter import TransmitterSDA


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
    )

    target_direction = actual_target_direction(p1, pt)
    alpha = config.sda.alpha

    def check_in_cone(q: float) -> bool:
        return in_cone_at_q(q, target_direction, transmitter.boresight_at, alpha)

    hit, hit_at_q = scan_hits_up_to(
        config.simulation.q_max,
        config.simulation.q_step,
        target_direction,
        transmitter.boresight_at,
        alpha,
    )

    alignment_at_q_max = alignment_dot(
        target_direction,
        transmitter.boresight_at(config.simulation.q_max),
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
        in_cone_at_q=check_in_cone,
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
    lines.append(
        f"Alignment at q_max = {result.alignment_at_q_max:.6f} "
        f"(threshold cos(alpha) = {__import__('numpy').cos(result.config.sda.alpha):.6f})"
    )
    q_max = result.config.simulation.q_max
    lines.append(
        f"Offsets at q_max: theta_off = {tx.theta_offset(q_max):.6f}, "
        f"phi_off = {tx.phi_offset(q_max):.6f}"
    )
    return "\n".join(lines)
