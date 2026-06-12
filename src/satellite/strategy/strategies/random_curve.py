"""Strategy 12: Random Curve — Continuous stochastic search with heading drift."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satellite.math3d import Vec3, normalize
from satellite.strategy.actions import beam, receiver, strategy
from satellite.strategy.base import SearchStrategy, StrategyContext, register_strategy
from satellite.strategy.movements import AimContext, MovementPattern

if TYPE_CHECKING:
    from satellite.config import ScenarioConfig


@dataclass(frozen=True)
class RandomCurveConfig:
    velocity_a: float = 0.01
    velocity_ratio: float = 1.41421356
    drift_sigma: float = 0.1
    seed: int = 42


def parse_random_curve_config(data: dict) -> RandomCurveConfig:
    return RandomCurveConfig(
        velocity_a=float(data.get("velocity_a", data.get("velocity", 0.01))),
        velocity_ratio=float(data.get("velocity_ratio", 1.41421356)),
        drift_sigma=float(data.get("drift_sigma", 0.1)),
        seed=int(data.get("seed", 42)),
    )


@dataclass(frozen=True)
class RandomCurvePattern(MovementPattern):
    velocity: float
    drift_sigma: float
    seed: int
    radius_limit: float
    dt_sim: float = 0.01  # Simulation step for internal integration

    def aim_at(self, local_t: float, duration: float, ctx: AimContext) -> Vec3:
        # Stateless-ish integration (seeded by seed)
        # For a real implementation, we might want a more efficient noise function
        rng = random.Random(self.seed)
        
        curr_u, curr_v = 0.0, 0.0
        heading = rng.uniform(0, 2 * math.pi)
        
        # Simple integration up to local_t
        t = 0.0
        while t < local_t:
            step = min(self.dt_sim, local_t - t)
            heading += rng.gauss(0, self.drift_sigma) * math.sqrt(step)
            curr_u += self.velocity * math.cos(heading) * step
            curr_v += self.velocity * math.sin(heading) * step
            
            # Boundary reflection
            dist = math.sqrt(curr_u**2 + curr_v**2)
            if dist > self.radius_limit:
                # Reflect heading back toward center
                angle_to_center = math.atan2(-curr_v, -curr_u)
                heading = angle_to_center + rng.uniform(-math.pi/4, math.pi/4)
                # Snap back
                curr_u *= self.radius_limit / dist
                curr_v *= self.radius_limit / dist
            t += step
            
        return normalize(ctx.u_z + curr_u * ctx.u_x + curr_v * ctx.u_y)


@register_strategy("random_curve", parse_random_curve_config)
class RandomCurveStrategy(SearchStrategy):
    def __init__(self, config: RandomCurveConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> RandomCurveStrategy:
        return cls(config=config.strategy.params.get("random_curve", RandomCurveConfig()))

    def build_script(self, ctx: StrategyContext):
        max_radius = ctx.config.simulation.max_search_radius
        timeout = ctx.config.simulation.timeout
        velocity_a = self.config.velocity_a
        velocity_b = velocity_a * self.config.velocity_ratio
        
        script = strategy(self.name)
        with script.satellite("S1"):
            beam.enable(); receiver.enable()
            script._builders["S1"].movement(
                RandomCurvePattern(
                    velocity=velocity_a,
                    drift_sigma=self.config.drift_sigma,
                    seed=self.config.seed,
                    radius_limit=max_radius
                ),
                duration=timeout,
                label="S1 random curve"
            )
        with script.satellite("S2"):
            beam.enable(); receiver.enable()
            script._builders["S2"].movement(
                RandomCurvePattern(
                    velocity=velocity_b,
                    drift_sigma=self.config.drift_sigma,
                    seed=self.config.seed + 1,
                    radius_limit=max_radius
                ),
                duration=timeout,
                label="S2 random curve"
            )
        return script.build()
