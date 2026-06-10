"""Compiled epoch schedule for replay and visualization."""

from __future__ import annotations

from dataclasses import dataclass

from satellite.strategy.actions import ActionScript, Epoch, SatelliteAction


@dataclass(frozen=True)
class ScheduledEpoch:
    q_start: float
    duration: float
    s1_action: SatelliteAction
    s2_action: SatelliteAction
    strategy_name: str = ""
    epoch_index: int = 0
    label: str = ""


@dataclass(frozen=True)
class LegSchedule:
    epochs: tuple[ScheduledEpoch, ...]

    @property
    def total_duration(self) -> float:
        if not self.epochs:
            return 0.0
        last = self.epochs[-1]
        return last.q_start + last.duration

    def epoch_at(self, q: float) -> tuple[ScheduledEpoch, float]:
        if not self.epochs:
            raise RuntimeError("empty LegSchedule")
        q = max(0.0, q)
        for epoch in self.epochs:
            if q < epoch.q_start + epoch.duration - 1e-12:
                return epoch, q - epoch.q_start
        last = self.epochs[-1]
        return last, last.duration

    def epoch_index_at(self, q: float) -> int:
        epoch, _ = self.epoch_at(q)
        return epoch.epoch_index

    def strategy_at(self, q: float) -> str:
        epoch, _ = self.epoch_at(q)
        return epoch.strategy_name


def compile_script(
    script: ActionScript,
    *,
    q_start: float = 0.0,
) -> tuple[ScheduledEpoch, ...]:
    scheduled: list[ScheduledEpoch] = []
    cursor = q_start
    for index, epoch in enumerate(script.epochs):
        scheduled.append(
            ScheduledEpoch(
                q_start=cursor,
                duration=epoch.duration,
                s1_action=epoch.s1,
                s2_action=epoch.s2,
                strategy_name=script.strategy_name,
                epoch_index=index,
                label=epoch.label,
            )
        )
        cursor += epoch.duration
    return tuple(scheduled)


def compile_trace(scripts: list[ActionScript]) -> LegSchedule:
    """Concatenate multiple strategy scripts into a full attempt trace."""
    epochs: list[ScheduledEpoch] = []
    cursor = 0.0
    for script in scripts:
        for index, epoch in enumerate(script.epochs):
            epochs.append(
                ScheduledEpoch(
                    q_start=cursor,
                    duration=epoch.duration,
                    s1_action=epoch.s1,
                    s2_action=epoch.s2,
                    strategy_name=script.strategy_name,
                    epoch_index=index,
                    label=epoch.label,
                )
            )
            cursor += epoch.duration
    return LegSchedule(epochs=tuple(epochs))
