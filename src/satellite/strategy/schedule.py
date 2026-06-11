"""Compiled timeline schedule for replay and visualization."""

from __future__ import annotations

from dataclasses import dataclass

from satellite.strategy.actions import MovementStep, StrategyScript


@dataclass(frozen=True)
class ScheduledScript:
    q_start: float
    script: StrategyScript
    attempt_index: int = 0

    @property
    def duration(self) -> float:
        return self.script.total_duration

    @property
    def q_end(self) -> float:
        return self.q_start + self.duration

    @property
    def strategy_name(self) -> str:
        return self.script.strategy_name


@dataclass(frozen=True)
class ScheduledStep:
    q_start: float
    duration: float
    strategy_name: str = ""
    step_index: int = 0
    label: str = ""
    s1_step: MovementStep | None = None
    s2_step: MovementStep | None = None
    script: StrategyScript | None = None


@dataclass(frozen=True)
class LegSchedule:
    scripts: tuple[ScheduledScript, ...]

    @property
    def steps(self) -> tuple[ScheduledStep, ...]:
        return tuple(self.step_at(script.q_start)[0] for script in self.scripts)

    @property
    def total_duration(self) -> float:
        if not self.scripts:
            return 0.0
        return self.scripts[-1].q_end

    def script_at(self, q: float) -> tuple[ScheduledScript, float]:
        if not self.scripts:
            raise RuntimeError("empty LegSchedule")
        q = max(0.0, q)
        for script in self.scripts:
            if q < script.q_end - 1e-12:
                return script, q - script.q_start
        last = self.scripts[-1]
        return last, last.duration

    def step_at(self, q: float) -> tuple[ScheduledStep, float]:
        scheduled, local_t = self.script_at(q)
        s1_step, s1_local = scheduled.script.s1.movement_at(local_t)
        s2_step, _ = scheduled.script.s2.movement_at(local_t)
        label = " / ".join(
            part for part in (s1_step.label, s2_step.label) if part
        )
        step_start = scheduled.q_start + min(s1_step.start, s2_step.start)
        step_end = scheduled.q_start + max(s1_step.end, s2_step.end)
        step = ScheduledStep(
            q_start=step_start,
            duration=step_end - step_start,
            strategy_name=scheduled.strategy_name,
            step_index=max(s1_step.index, s2_step.index),
            label=label,
            s1_step=s1_step,
            s2_step=s2_step,
            script=scheduled.script,
        )
        return step, s1_local

    def step_index_at(self, q: float) -> int:
        step, _ = self.step_at(q)
        return step.step_index

    def strategy_at(self, q: float) -> str:
        script, _ = self.script_at(q)
        return script.strategy_name


def compile_script(
    script: StrategyScript,
    *,
    q_start: float = 0.0,
    attempt_index: int = 0,
) -> ScheduledScript:
    return ScheduledScript(
        q_start=q_start,
        script=script,
        attempt_index=attempt_index,
    )


def compile_trace(scripts: list[StrategyScript]) -> LegSchedule:
    """Concatenate multiple strategy scripts into a full attempt trace."""
    scheduled: list[ScheduledScript] = []
    cursor = 0.0
    for index, script in enumerate(scripts):
        scheduled.append(compile_script(script, q_start=cursor, attempt_index=index))
        cursor += script.total_duration
    return LegSchedule(scripts=tuple(scheduled))
