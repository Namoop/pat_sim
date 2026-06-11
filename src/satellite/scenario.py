"""Scenario orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from satellite.config import ScenarioConfig, default_beam_length
from satellite.geometry import actual_target_direction
from satellite.math3d import Vec3, angle_between, distance, normalize
from satellite.strategy.base import StrategyContext, link_established
from satellite.strategy.meta import MetaStrategy, MetaStrategyResult
from satellite.strategy.movements import Reset, build_aim_context
from satellite.strategy.runner import FrameRunner
from satellite.strategy.schedule import LegSchedule

if TYPE_CHECKING:
    from satellite.diagnostics import SimReplayProfiler
    from satellite.sda.receiver import ReceiverSDA
    from satellite.sda.transmitter import TransmitterSDA


@dataclass
class ScenarioResult:
    config: ScenarioConfig
    s1: Satellite
    s2: Satellite
    meta: MetaStrategyResult
    phase1_target_direction: Vec3
    phase2_target_direction: Vec3
    _replay_timeline: object | None = field(default=None, repr=False)
    _stepper: object | None = field(default=None, repr=False)
    last_sim_profiler: SimReplayProfiler | None = field(default=None, repr=False)

    @property
    def schedule(self) -> LegSchedule:
        return self.meta.schedule

    @property
    def success(self) -> bool:
        return self.meta.success

    @property
    def strategy_name(self) -> str | None:
        return self.meta.winning_strategy

    @property
    def hit(self) -> bool:
        return self.meta.success

    @property
    def hit_at_q(self) -> float | None:
        return self.meta.hit_at_q

    @property
    def playable_q_end(self) -> float:
        """Last q for replay and visualization (lock time, or full schedule if no lock)."""
        hit = self.hit_at_q
        if hit is not None:
            return hit
        return self.schedule.total_duration

    @property
    def phase2_spiral_center_source(self) -> str:
        if self.s2.receiver.has_seen_beam:
            return "locked"
        return "initial_aim"

    @property
    def boresight_end(self) -> Vec3:
        return self.s2.bench.beam_boresight_inertial().copy()

    @boresight_end.setter
    def boresight_end(self, value: Vec3) -> None:
        self.s2.bench.set_bench_aim(value)

    @property
    def p1(self) -> Vec3:
        return self.s1.position

    @property
    def pt(self) -> Vec3:
        return self.s2.position

    @property
    def transmitter(self) -> TransmitterSDA:
        return self.s1.transmitter

    @property
    def receiver(self) -> ReceiverSDA:
        return self.s2.receiver

    @property
    def target_direction(self) -> Vec3:
        return self.phase1_target_direction

    @property
    def believed_boresight(self) -> Vec3:
        return self.s1.believed_boresight

    @property
    def believed_distance(self) -> float:
        return distance(self.s1.position, self.s2.position)

    def local_q(self, q: float) -> float:
        _, local = self.schedule.step_at(q)
        return local

    def bench_aim(self, satellite: str, q: float) -> Vec3:
        self.replay_to(q)
        sat = self.s1 if satellite == "S1" else self.s2
        return sat.bench.bench_boresight.copy()

    def beam_length(self, satellite: str) -> float:
        sat = self.s1 if satellite == "S1" else self.s2
        return default_beam_length(
            sat.position,
            sat.partner_actual,
            self.config.simulation,
        )

    def bidirectional_lock(self, q: float) -> tuple[bool, bool]:
        if self._stepper is None or abs(self._stepper.global_q - q) > 1e-9:
            self.replay_to(q)
        aim1 = self.s1.bench.bench_boresight
        aim2 = self.s2.bench.bench_boresight
        scheduled, local_t = self.schedule.script_at(q)
        s1_beam, s1_receiver = scheduled.script.s1.hardware_state_at(local_t)
        s2_beam, s2_receiver = scheduled.script.s2.hardware_state_at(local_t)
        return (
            s1_beam
            and s2_receiver
            and link_established(self.s1, self.s2, aim1, self.config),
            s2_beam
            and s1_receiver
            and link_established(self.s2, self.s1, aim2, self.config),
        )

    def mutual_lock(self, q: float) -> bool:
        hit_12, hit_21 = self.bidirectional_lock(q)
        return hit_12 and hit_21

    def in_cone_at_q(self, q: float) -> bool:
        return self.mutual_lock(q)

    def check_dish_hit(self, q: float) -> bool:
        return self.mutual_lock(q)

    def dish_boresight_for_display(self, satellite: str, q: float) -> Vec3:
        return self.bench_aim(satellite, q)

    def boresight_ray_length(self, satellite: str) -> float:
        sat = self.s1 if satellite == "S1" else self.s2
        return (
            distance(sat.position, sat.partner_actual)
            + self.config.simulation.boresight_extension
        )

    def ensure_replay_timeline(self):
        if self._replay_timeline is None:
            from satellite.replay_timeline import build_replay_timeline

            self._replay_timeline = build_replay_timeline(self)
        return self._replay_timeline

    def replay_to(
        self,
        q_end: float,
        *,
        event_log: list[str] | None = None,
    ) -> None:
        from satellite.replay_timeline import replay_to_q

        replay_to_q(self, q_end, event_log=event_log)


# Late import to avoid circular dependency
from satellite.sda.satellite import Satellite  # noqa: E402


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Run headless meta-strategy search."""
    p1 = config.s1.position
    pt = config.s2.position

    s1 = Satellite.build("S1", config.s1, pt, config)
    s2 = Satellite.build("S2", config.s2, p1, config)

    ctx = StrategyContext(s1=s1, s2=s2, config=config)
    meta = MetaStrategy.from_config(config).run(ctx)

    s1 = meta.s1
    s2 = meta.s2

    return ScenarioResult(
        config=config,
        s1=s1,
        s2=s2,
        meta=meta,
        phase1_target_direction=actual_target_direction(p1, pt),
        phase2_target_direction=actual_target_direction(pt, p1),
    )


def format_summary(result: ScenarioResult) -> str:
    """Human-readable headless run summary."""
    cfg = result.config
    lines = [
        f"Scenario: {cfg.name}",
        f"Strategy chain: {', '.join(cfg.strategy.chain)}",
        f"Schedule duration: {result.schedule.total_duration:.3f} sim-q",
        f"Success: {'yes' if result.success else 'no'}",
    ]
    if result.strategy_name:
        lines.append(f"Winning strategy: {result.strategy_name}")
    if result.hit_at_q is not None:
        lines.append(f"Hit at q = {result.hit_at_q:.4f}")
    if result.meta.lock_direction:
        lines.append(f"Lock direction: {result.meta.lock_direction}")

    for attempt in result.meta.attempts:
        status = "success" if attempt.success else "no lock"
        lines.append(
            f"  Attempt {attempt.strategy_name}: {status} "
            f"(duration {attempt.elapsed_q:.3f})"
        )
        if attempt.skipped_reason:
            lines.append(f"    skipped: {attempt.skipped_reason}")

    for name, sat in (("S1", result.s1), ("S2", result.s2)):
        lines.append(
            f"{name} pointing offset = "
            f"{np.degrees(angle_between(sat.bench.toward_partner, sat.bench.initial_boresight)):.3f} deg"
        )
        if sat.receiver.has_seen_beam and sat.bench.acquisition.incident_angle is not None:
            lines.append(
                f"{name} dish incident at detection = "
                f"{np.degrees(sat.bench.acquisition.incident_angle):.3f} deg "
                f"(FOV {np.degrees(sat.receiver.dish_fov):.3f} deg)"
            )

    return "\n".join(lines)
