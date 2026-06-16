from satellite.config import build_scenario_config
from satellite.scenario import run_scenario

def evaluate_single_instance(sim_cfg, instance, strategy) -> tuple[bool, float | None]:
    """Runs a single scenario simulation. Runs in a process pool."""
    config = build_scenario_config(sim_cfg, instance, strategy=strategy)
    res = run_scenario(config)
    return res.success, res.hit_at_t
