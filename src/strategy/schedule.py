"""Compiled timeline schedule for replay and visualization."""

from __future__ import annotations

from dataclasses import dataclass

from strategy.actions import MovementStep, StrategyScript


@dataclass(frozen=True)
class ScheduledScript:
    t_start: float
    script: StrategyScript
    attempt_index: int = 0

    @property
    def duration(self) -> float:
        return self.script.total_duration

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration

    @property
    def strategy_name(self) -> str:
        return self.script.strategy_name


@dataclass(frozen=True)
class ScheduledStep:
    t_start: float
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
        return tuple(self.step_at(script.t_start)[0] for script in self.scripts)

    @property
    def total_duration(self) -> float:
        if not self.scripts:
            return 0.0
        return self.scripts[-1].t_end

    def script_at(self, t: float) -> tuple[ScheduledScript, float]:
        if not self.scripts:
            raise RuntimeError("empty LegSchedule")
        t = max(0.0, t)
        for script in self.scripts:
            if t < script.t_end - 1e-12:
                return script, t - script.t_start
        last = self.scripts[-1]
        return last, last.duration

    def step_at(self, t: float) -> tuple[ScheduledStep, float]:
        scheduled, local_t = self.script_at(t)
        s1_step, s1_local = scheduled.script.s1.movement_at(local_t)
        s2_step, _ = scheduled.script.s2.movement_at(local_t)
        label = " / ".join(
            part for part in (s1_step.label, s2_step.label) if part
        )
        step_start = scheduled.t_start + min(s1_step.start, s2_step.start)
        step_end = scheduled.t_start + max(s1_step.end, s2_step.end)
        step = ScheduledStep(
            t_start=step_start,
            duration=step_end - step_start,
            strategy_name=scheduled.strategy_name,
            step_index=max(s1_step.index, s2_step.index),
            label=label,
            s1_step=s1_step,
            s2_step=s2_step,
            script=scheduled.script,
        )
        return step, s1_local

    def step_index_at(self, t: float) -> int:
        step, _ = self.step_at(t)
        return step.step_index

    def strategy_at(self, t: float) -> str:
        script, _ = self.script_at(t)
        return script.strategy_name


def compile_script(
    script: StrategyScript,
    *,
    t_start: float = 0.0,
    attempt_index: int = 0,
) -> ScheduledScript:
    return ScheduledScript(
        t_start=t_start,
        script=script,
        attempt_index=attempt_index,
    )


def compile_trace(scripts: list[StrategyScript]) -> LegSchedule:
    """Concatenate multiple strategy scripts into a full attempt trace."""
    scheduled: list[ScheduledScript] = []
    cursor = 0.0
    for index, script in enumerate(scripts):
        scheduled.append(compile_script(script, t_start=cursor, attempt_index=index))
        cursor += script.total_duration
    return LegSchedule(scripts=tuple(scheduled))
