"""Precomputed transmitter spiral boresight lookup."""

from __future__ import annotations

import numpy as np

from satellite.math3d import Vec3


class BoresightTable:
    """Precomputed boresight directions for q in [0, q_max] at fixed q_step."""

    def __init__(self, q_values: np.ndarray, boresights: np.ndarray) -> None:
        self.q_values = q_values
        self.boresights = boresights

    @classmethod
    def build(
        cls,
        boresight_fn,
        q_max: float,
        q_step: float,
    ) -> BoresightTable:
        n = max(1, int(np.ceil(q_max / q_step - 1e-12)) + 1)
        q_values = np.linspace(0.0, q_max, n)
        boresights = np.array([boresight_fn(float(q)) for q in q_values])
        return cls(q_values, boresights)

    def at(self, q: float) -> Vec3:
        """Nearest-index lookup for local spiral parameter q."""
        if len(self.q_values) == 1:
            return self.boresights[0].copy()
        step = float(self.q_values[1] - self.q_values[0])
        idx = int(np.round(q / step))
        idx = int(np.clip(idx, 0, len(self.q_values) - 1))
        return self.boresights[idx].copy()
