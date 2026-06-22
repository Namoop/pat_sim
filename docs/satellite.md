# Satellite Communication Simulation Model

This document outlines the underlying physical and logical models used in the simulation, as well as the repository layout.

For a detailed walkthrough of the physical system model (satellites, optical benches, FSMs, and offset bleeding), please refer to the **System Architecture & Model** section in the main [README.md](../README.md).

---

## Model Summary

- **Detection:** A transmission cone hits the target satellite's dish mount; the incoming direction from the transmitter body must fall within the receiver satellite's `dish_fov`.
- **Search:** Independent per-satellite timelines (hold, spiral, reset, etc.) with optional beam/receiver enable states; the FSM snaps and tracking starts upon acquisition.
- **Lock:** Mutual optical lock is achieved when both satellites are transmitting and receiving simultaneously, both bench slews are complete, and there is simultaneous bidirectional visibility (`visible_12 ∧ visible_21`).
- **Partial Acquisition:** If one satellite acquires the other before a strategy times out, the acquiring satellite keeps tracking and ignores later scripted search; the non-acquired satellite continues its strategy chain normally.
- **Replay:** The headless runner and visualizer share the same coupled replay timeline, which is capped at the lock time on successful runs.

---

## CLI Usage Reference

This section provides a detailed reference for running the satellite simulation and related command-line utilities.

### Scenario Command

```bash
python -m scenario [config] [options]
```

Runs a single satellite SDA communication scenario. Optional positional `config` defaults to `config/Scenario.toml`.

#### Scenario Options

- `--visualize [{3d,eye,mag}]`  
Opens the unified visualization window after the run. The optional choice `3d`, `eye`, or `mag` picks the **initial tab** (3D PyVista view, angular θ/φ eye map, or 2D magnitude alignment). `--visualize` alone is equivalent to `--visualize 3d`. If omitted, the window opens when `[scenario].visualize` is specified as `"3d"`, `"eye"`, or `"mag"` in the scenario config.

> - **Eye View (`eye`)**: Named "eye" because you are seeing the "eye" of each satellite, and since there are two satellites total, it looks kind of like eyes.
> - **Mag View (`mag`)**: Short for "magnitude" since all you see is in 2D representing the total magnitude offset.

- `--environment ENVIRONMENT`  
Path to the Environment base TOML containing hardware, timing, distance, `t_step`, and visualization defaults.
- `--t T` (default: `0`)  
Starting time `t` when opening a visualizer.

### Monte Carlo Command

```bash
python -m montecarlo [config] [options]
```

Runs a Monte Carlo batch from the specified TOML file (default: `config/MonteCarlo.toml`). Without `--visualize`, runs the full batch headlessly and prints a summary. With visualization enabled (via `--visualize`), runs one scenario at a time in the visualizer; use **Next** to advance. On successful runs the timeline ends at mutual lock — replay cache, slider, and playback cannot scrub past that point. Playback controls sit above the view; the **event log** (system, S1, S2) is in a three-column strip at the bottom.

#### Monte Carlo Options

- `--visualize [{3d,eye,mag}]` — same as scenario; opens interactive step-through mode.
- `--t T` (default: `0`) — starting time when opening a visualizer.
- `--run NUMBER` (default: `1`) — starting run number (1-based) when opening a visualizer.
- `--export [NAME]` — export the selected run to `config/_scenario_NAME.toml` (default name: `run N`); does not run the simulation. Requires `monte_carlo.seed` in the config.
- `--autoplay [SPEED]` — auto-scrub visualization; advance to the next run when each finishes.

### Examples

```bash
# Run the default scenario
python -m scenario

# Visualize with specific starting time and eye tab initially active
python -m scenario --visualize eye --t 2.5

# Run Monte Carlo simulation batch
python -m montecarlo config/MonteCarlo.toml
```

---

### Eye View Render Benchmark

```bash
python -m visualize.bench_eye_render [options]
```

Benchmark the eye-view QPainter render path (headless Qt). Requires the `[viz]` extra dependency (PyQt6).

### Benchmark Options

- `--scenario SCENARIO` (default: `config/Scenario.toml`)  
Scenario instance TOML.
- `--environment ENVIRONMENT` (default: `config/Environment.toml`)  
Environment base TOML.
- `--samples N` (default: `1000`)  
Number of random `t` samples.
- `--seed N` (default: `0`)  
RNG seed for sample times.

---

### Linux/Wayland Visualization Troubleshooting

On Linux with Wayland, Qt windows (visualizer) may fail to open or render incorrectly. Force the X11 backend via `QT_QPA_PLATFORM=xcb`:

```bash
QT_QPA_PLATFORM=xcb python -m scenario [...]
QT_QPA_PLATFORM=xcb python -m montecarlo [...]
```

---

## Configuration Architecture

Configuration is split across per-module parsers; runners orchestrate loading and merging:


| Module                                                | Role                                                             |
| ----------------------------------------------------- | ---------------------------------------------------------------- |
| `[config.py](../src/config.py)`                       | `load_toml`, path resolution, override merge helpers             |
| `[scenario/types.py](../src/scenario/types.py)`       | `ScenarioConfig`, `ScenarioInstance`, `build_scenario_config`    |
| `[montecarlo/types.py](../src/montecarlo/types.py)`   | `MonteCarloConfig`, error distribution types                     |
| `[satellite/config.py](../src/satellite/config.py)`   | `Environment.toml` types and `load_simulation_config`            |
| `[scenario/config.py](../src/scenario/config.py)`     | Parses `[scenario]`, `[s1]`, `[s2]` → `ScenarioInstance`         |
| `[strategy/config.py](../src/strategy/config.py)`     | Parses `[strategy]` and per-strategy sections → `StrategyConfig` |
| `[montecarlo/config.py](../src/montecarlo/config.py)` | Parses `[monte_carlo]` and error distribution                    |
| `[optimize/config.py](../src/optimize/config.py)`     | Parses `[optimize]`                                              |


Orchestration entry points:

- **Single scenario:** `load_single_scenario` in `[scenario/run.py](../src/scenario/run.py)`
- **Monte Carlo batch:** `load_monte_carlo_config` in `[montecarlo/run.py](../src/montecarlo/run.py)`

See [environment.md](environment.md), [scenario.md](scenario.md), and [monte_carlo.md](monte_carlo.md) for TOML field reference.