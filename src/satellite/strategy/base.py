"""Strategy ABC, context, and link detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from satellite.config import ScenarioConfig, default_beam_length
from satellite.detection import beam_hits_dish
from satellite.detection_fast import beam_hits_dish_fast
from satellite.math3d import Vec3
from satellite.strategy.actions import StrategyScript, validate_movement_durations

if TYPE_CHECKING:
    from satellite.sda.satellite import Satellite

T = TypeVar("T", bound="SearchStrategy")
ConfigParser = Callable[[dict[str, Any]], Any]

STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {}
CONFIG_PARSERS: dict[str, ConfigParser] = {}


def register_strategy(
    name: str,
    parser: ConfigParser | None = None,
) -> Callable[[type[T]], type[T]]:
    def wrapper(cls: type[T]) -> type[T]:
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        if parser is not None:
            CONFIG_PARSERS[name] = parser
        return cls

    return wrapper


@dataclass
class StrategyContext:
    s1: Satellite
    s2: Satellite
    config: ScenarioConfig
    s1_frozen_hardware: tuple[bool, bool] | None = None
    s2_frozen_hardware: tuple[bool, bool] | None = None

    @property
    def t_step(self) -> float:
        return self.config.simulation.t_step

    def reset_satellites(self) -> None:
        self.s1.receiver.reset_dish_tracking()
        self.s2.receiver.reset_dish_tracking()

    def clone_fresh(self) -> StrategyContext:
        from satellite.sda.satellite import Satellite

        s1 = Satellite.build("S1", self.config.s1, self.config.s2.position, self.config)
        s2 = Satellite.build("S2", self.config.s2, self.config.s1.position, self.config)
        return StrategyContext(s1=s1, s2=s2, config=self.config)

    def clone_for_next_attempt(
        self,
        *,
        end_hardware: dict[str, bool] | None = None,
    ) -> StrategyContext:
        """Fresh satellites for naive sides; preserve acquired satellites and hardware."""
        from satellite.sda.satellite import Satellite

        s1 = (
            self.s1
            if self.s1.receiver.has_seen_beam
            else Satellite.build(
                "S1", self.config.s1, self.config.s2.position, self.config
            )
        )
        s2 = (
            self.s2
            if self.s2.receiver.has_seen_beam
            else Satellite.build(
                "S2", self.config.s2, self.config.s1.position, self.config
            )
        )
        s1_hw = self.s1_frozen_hardware
        s2_hw = self.s2_frozen_hardware
        if end_hardware is not None:
            if self.s1.receiver.has_seen_beam:
                s1_hw = (end_hardware["s1_beam"], end_hardware["s1_receiver"])
            if self.s2.receiver.has_seen_beam:
                s2_hw = (end_hardware["s2_beam"], end_hardware["s2_receiver"])
        return StrategyContext(
            s1=s1,
            s2=s2,
            config=self.config,
            s1_frozen_hardware=s1_hw,
            s2_frozen_hardware=s2_hw,
        )


@dataclass
class StrategyResult:
    success: bool
    strategy_name: str
    hit_at_t: float | None
    script: StrategyScript
    elapsed_t: float
    skipped_reason: str | None = None
    metadata: dict = field(default_factory=dict)


class SearchStrategy(ABC):
    name: str

    @classmethod
    @abstractmethod
    def from_config(cls, config: ScenarioConfig) -> SearchStrategy:
        """Instantiate from scenario configuration."""

    @abstractmethod
    def build_script(self, ctx: StrategyContext) -> StrategyScript:
        """Return independent per-satellite timelines."""

    def try_run(self, ctx: StrategyContext, global_t_start: float) -> StrategyResult:
        from satellite.strategy.runner import FrameRunner

        script = self.build_script(ctx)
        validate_movement_durations(ctx, script)
        script.validate_slew_speed(ctx)
        run = FrameRunner(ctx).execute(script, global_t_start=global_t_start)
        return StrategyResult(
            success=run.success,
            strategy_name=self.name,
            hit_at_t=run.hit_at_t,
            script=script,
            elapsed_t=script.total_duration,
            metadata=run.metadata,
        )


def beam_length_for(tx: Satellite, config: ScenarioConfig) -> float:
    return default_beam_length(tx.position, tx.partner_actual, config.simulation)


def link_established(
    tx_sat: Satellite,
    rx_sat: Satellite,
    beam_axis: Vec3,
    config: ScenarioConfig,
) -> bool:
    # Optimization: Use pre-calculated cosines and bypass geometry snapshot
    dish_boresight = rx_sat.bench.dish_boresight_inertial()
    mount = rx_sat.bench.dish_mount_for_boresight(
        dish_boresight, rx_sat.receiver.body_radius
    )

    return beam_hits_dish_fast(
        tx_sat.position,
        mount,
        dish_boresight,
        rx_sat.receiver.cos_dish_fov,
        beam_axis,
        tx_sat.cos_alpha,
        tx_sat.beam_length,
    )
