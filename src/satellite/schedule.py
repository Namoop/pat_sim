"""Clock-driven two-phase search schedule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SearchPhase(Enum):
    S1_TRANSMIT = "S1_TRANSMIT"
    S2_TRANSMIT = "S2_TRANSMIT"


@dataclass(frozen=True)
class SearchSchedule:
    """Immutable schedule: phase 1 [0, q_max), phase 2 [q_max, 2·q_max)."""

    phase_duration: float

    @property
    def total_duration(self) -> float:
        return 2.0 * self.phase_duration

    def phase_at(self, q: float) -> tuple[SearchPhase, float]:
        if q < self.phase_duration:
            return SearchPhase.S1_TRANSMIT, q
        return SearchPhase.S2_TRANSMIT, q - self.phase_duration

    @staticmethod
    def transmitting_satellite(phase: SearchPhase) -> str:
        return "S1" if phase is SearchPhase.S1_TRANSMIT else "S2"

    @staticmethod
    def receiving_satellite(phase: SearchPhase) -> str:
        return "S2" if phase is SearchPhase.S1_TRANSMIT else "S1"
