"""Strategy 9: Stochastic Center Re-bias — Stochastic curve with center-heavy pull."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from satellite.math.math3d import Vec3, normalize
from strategy.actions import beam, receiver, strategy
from strategy.base import SearchStrategy, StrategyContext, register_strategy
from strategy.movements import AimContext, MovementPattern

if TYPE_CHECKING:
    from scenario.types import ScenarioConfig


@dataclass(frozen=True)
class CenterRebiasConfig:
    velocity_a: float = 0.01
    velocity_ratio: float = 1.41421356
    drift_sigma: float = 0.1
    max_turn_radius: float = 0.5
    bias_strength: float = 0.2
    seed: int = 123


def parse_center_rebias_config(data: dict) -> CenterRebiasConfig:
    return CenterRebiasConfig(
        velocity_a=float(data.get("velocity_a", data.get("velocity", 10.0))) * 1e-3,
        velocity_ratio=float(data.get("velocity_ratio", 1.41421356)),
        drift_sigma=float(data.get("drift_sigma", 100.0)) * 1e-3,
        max_turn_radius=float(data.get("max_turn_radius", 500.0)) * 1e-3,
        bias_strength=float(data.get("bias_strength", 0.2)),
        seed=int(data.get("seed", 123)),
    )


@dataclass(frozen=True)
class CenterRebiasPattern(MovementPattern):
    velocity: float
    drift_sigma: float
    max_turn_radius: float
    bias_strength: float
    seed: int
    radius_limit: float
    dt_sim: float = 0.01
    _cache: list[tuple[float, float]] = field(
        default_factory=list, init=False, hash=False, compare=False
    )

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        if not self._cache:
            rng = random.Random(self.seed)
            
            curr_u, curr_v = 0.0, 0.0
            heading = rng.uniform(0, 2 * math.pi)
            current_dir = 0.0  # steering wheel angle
            
            # Simple integration up to duration
            t = 0.0
            while t <= duration + 1e-9:
                self._cache.append((curr_u, curr_v))
                
                # Drift the steering wheel angle
                current_dir += rng.gauss(0, self.drift_sigma) * math.sqrt(self.dt_sim)
                current_dir = max(-self.max_turn_radius, min(self.max_turn_radius, current_dir))
                
                # Heading updates based on steering angle
                heading += current_dir * self.dt_sim
                
                # Position updates stochastically
                curr_u += self.velocity * math.cos(heading) * self.dt_sim
                curr_v += self.velocity * math.sin(heading) * self.dt_sim
                
                # Gravitational center pull (bias_strength)
                curr_u *= (1.0 - self.bias_strength * self.dt_sim)
                curr_v *= (1.0 - self.bias_strength * self.dt_sim)
                
                # Boundary reflection
                dist = math.sqrt(curr_u**2 + curr_v**2)
                if dist > self.radius_limit:
                    angle_to_center = math.atan2(-curr_v, -curr_u)
                    heading = angle_to_center + rng.uniform(-math.pi/4, math.pi/4)
                    current_dir = 0.0
                    curr_u *= self.radius_limit / dist
                    curr_v *= self.radius_limit / dist
                t += self.dt_sim
            
        # O(1) lookup
        idx = min(int(local_t / self.dt_sim), len(self._cache) - 1)
        curr_u, curr_v = self._cache[idx]
        return normalize(ctx.u_z + curr_u * ctx.u_x + curr_v * ctx.u_y)


@register_strategy("center_rebias", parse_center_rebias_config)
class CenterRebiasStrategy(SearchStrategy):
    def __init__(self, config: CenterRebiasConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> CenterRebiasStrategy:
        return cls(config=config.strategy.params.get("center_rebias", CenterRebiasConfig()))

    def build_script(self, ctx: StrategyContext):
        max_radius = ctx.config.simulation.max_search_radius
        timeout = ctx.config.simulation.timeout
        velocity_a = self.config.velocity_a
        velocity_b = velocity_a * self.config.velocity_ratio
        
        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                CenterRebiasPattern(
                    velocity=velocity_a,
                    drift_sigma=self.config.drift_sigma,
                    max_turn_radius=self.config.max_turn_radius,
                    bias_strength=self.config.bias_strength,
                    seed=self.config.seed,
                    radius_limit=max_radius
                ),
                duration=timeout,
                label="S1 center rebias"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                CenterRebiasPattern(
                    velocity=velocity_b,
                    drift_sigma=self.config.drift_sigma,
                    max_turn_radius=self.config.max_turn_radius,
                    bias_strength=self.config.bias_strength,
                    seed=self.config.seed + 1,
                    radius_limit=max_radius
                ),
                duration=timeout,
                label="S2 center rebias"
            )
        return script.build()
