# Scenario Configuration (`Scenario.toml`)

This document describes the parameters defined in `Scenario.toml` (e.g. `config/Scenario.toml`), which configures a single execution run with fixed pointing offsets.

Scenario sections are parsed by [`scenario/config.py`](../src/scenario/config.py) (`load_scenario_config` / `parse`). For a full runnable config, [`load_single_scenario`](../src/scenario/run.py) in `scenario/run.py` loads the scenario TOML, parses `[scenario]` / `[s1]` / `[s2]` and `[strategy]`, loads the referenced `Environment.toml`, and merges them via `build_scenario_config` in [`scenario/types.py`](../src/scenario/types.py). All angular values in scenario configurations are defined in **milliradians**.

---

## Configuration Blocks

### `[scenario]`

- `name` (string): Unique identifier for the scenario instance.
- `chain` (array of strings, required): Ordered list of search strategies to run in sequence (e.g., `["minor_offset", "single_miss"]`).
- `environment` (string, optional): Path to the associated environment config TOML file (e.g. `"Environment.toml"`, resolved relative to the scenario file).
- `visualize` (string, optional): Automatically opens the visualizer after a single run if set to `"3d"` or `"eye"`.
- **Arbitrary Property Overrides**: Any simulation or satellite property can be overridden for the specific scenario using dotted keys or nested sub-tables under `[scenario]`.
  - *Dotted Keys Example*: `simulation.distance = 500` or `satellite.max_fsm_radius = 0.5`.
  - *Nested Tables Example*: `[scenario.simulation] distance = 500`

### `[s1]` and `[s2]` (Spacecraft Bench Offsets)

- `bench_theta_offset` (float, milliradians): Initial local theta pointing offset for the spacecraft's optical bench.
- `bench_phi_offset` (float, milliradians): Initial local phi pointing offset for the spacecraft's optical bench.

### `[strategy.<strategy_name>]` (Strategy Parameters)

Scenario configuration files contain parameters to configure individual strategies in the strategy chain. For a complete list of parameters available for each strategy, see [docs/strategies.md](strategies.md).

*Example:*

```toml
[strategy.minor_offset]
max_spiral_radius = "fov"
spiral_speed = 0.4

[strategy.single_miss]
a_spiral_radius = 50.0
b_spiral_radius = 50.0
spiral_speed = 0.05
```

