# Satellite Communication Simulation Model

This document outlines the underlying physical and logical models used in the simulation, as well as the repository layout.

## Detailed Simulation Guide

### 1. Physical Setup
The simulation models two satellites, **S1** (placed at the origin) and **S2** (placed on the positive x-axis at `distance`).
* **Optical Bench (Beam Director):** Each satellite has an optical bench that acts as a coarse-pointing system. It has a wide range of motion but slews slowly, limited by `max_beam_speed`. Initial pointing errors (launch mispoints) are modeled as angular theta/phi offsets of this bench.
* **Fast Steering Mirror (FSM):** Mounted on the optical bench is a fine-pointing FSM. The FSM is extremely fast (`max_fsm_speed`) and precise, but has a very narrow angular range of motion, limited by `max_fsm_radius`.
* **Transmitter & Receiver:** The transmitter narrow beam (half-angle `beam_width`) and the receiver dish (field-of-view `dish_fov`) share the bench boresight and are steered dynamically by the FSM.

### 2. General Strategy Concept
Because of the initial pointing offsets, the satellites' default boresights do not align. To establish a link, they must execute coordinated search strategies (e.g., spirals, rasters, Lissajous scans) to sweep their beams and receiver fields of view across the uncertainty region.
A strategy chain orchestrates this search, escalating to wider or more robust strategies if a lock is not established within a given strategy's timeout.

### 3. FSM Snapping and Offset Bleeding
When a satellite (e.g., S2) detects an incoming signal from S1 (meaning S1's beam hits S2 and falls within S2's receiver FOV):
1. **FSM Snap (Acquisition):** S2's FSM can move extremely quickly compared to the heavy optical bench. It immediately deflects ("snaps") to point S2's receiver and transmitter beam directly at S1. This establishes an immediate, temporary lock.
2. **Offset Bleeding:** While the FSM holds the connection, the slow optical bench begins slewing toward S1. As the bench rotates closer to S1's true position, the required FSM deflection angle decreases. This offset is "bled" off the FSM and back to the beam director, centering the FSM so it has its full range of motion available to track high-frequency movement.

### 4. Lock Condition
The ultimate goal is a stable **mutual lock**. A mutual lock is declared when:
* Both satellites are simultaneously transmitting to and receiving from one another.
* Both satellites have completed their bench slews (the pointing offset is fully bled, and the optical benches are aligned).
* There is bidirectional line-of-sight visibility.

---

## Model Summary

* **Detection:** A transmission cone hits the target satellite's dish mount; the incoming direction from the transmitter body must fall within the receiver satellite's `dish_fov`.
* **Search:** Independent per-satellite timelines (hold, spiral, reset, etc.) with optional beam/receiver enable states; the FSM snaps and tracking starts upon acquisition.
* **Lock:** Mutual optical lock is achieved when both satellites are transmitting and receiving simultaneously, both bench slews are complete, and there is simultaneous bidirectional visibility (`visible_12 ∧ visible_21`).
* **Partial Acquisition:** If one satellite acquires the other before a strategy times out, the acquiring satellite keeps tracking and ignores later scripted search; the non-acquired satellite continues its strategy chain normally.
* **Replay:** The headless runner and visualizer share the same coupled replay timeline, which is capped at the lock time on successful runs.

---

## Project Layout

The repository is structured as a Python package containing the following modules:

```
src/satellite/
  config.py       — Simulation / scenario / MC loaders
  monte_carlo.py  — error sampling and batch runner
  strategy/
    actions.py    — timeline DSL and StrategyScript
    runner.py     — frame runner
    schedule.py   — compiled timeline for replay
    meta.py       — strategy chain orchestrator
    strategies/   — built-in strategy implementations
  sda/            — bench, transmitter, receiver
  scenario.py     — orchestration and replay
  visualize/      — unified 3D + map visualizer
  mapviz/         — QPainter angular map panels
```

---

## CLI Usage Reference

This section provides a detailed reference for running the satellite simulation and related command-line utilities.

### Main Command

```bash
python -m satellite [options]
```

Runs a satellite SDA communication scenario.

### Command-line Options

* **`--visualize [{3d,map}]`**  
  Opens the unified visualization window after the run. The optional choice `3d` or `map` picks the **initial tab** (3D PyVista view or angular θ/φ map). `--visualize` alone is equivalent to `--visualize 3d`. If omitted, the window opens when `[scenario].visualize` is specified as `"3d"` or `"map"` in the scenario config. With `--monte-carlo`, opens an interactive step-through mode: **Next** runs the next sampled scenario (or closes on the last run / single scenario). On successful runs the timeline ends at mutual lock — replay cache, slider, and playback cannot scrub past that point. Playback controls sit above the view; the **event log** (system, S1, S2) is in a three-column strip at the bottom.

* **`--monte-carlo [MONTE_CARLO]`** (default if flag present but path omitted: `config/MonteCarlo.toml`)  
  Runs a Monte Carlo batch from the specified TOML file. Without `--visualize`, runs the full batch headlessly and prints a summary. With visualization enabled (via `--visualize`), runs one scenario at a time in the visualizer; use **Next** to advance.

* **`--scenario [SCENARIO]`** (default: `config/Scenario.toml`)  
  Path to the Scenario instance TOML containing bench offsets, strategy chain, and optional overrides. (Also defaults to `config/Scenario.toml` if the flag is provided without a path).

* **`--environment ENVIRONMENT`** (default: `config/Environment.toml`)  
  Path to the Environment base TOML containing hardware, timing, distance, `t_step`, and visualization defaults.

* **`--t T`** (default: `0`)  
  Starting time `t` when opening a visualizer.

### Examples

```bash
# Run the default scenario
python -m satellite

# Run Monte Carlo simulation batch
python -m satellite --monte-carlo config/MonteCarlo.toml

# Visualize with specific starting time and map tab initially active
python -m satellite --visualize map --t 2.5
```

---

### Map Render Benchmark

```bash
python -m satellite.mapviz.bench_render [options]
```

Benchmark the mapviz QPainter render path (headless Qt). Requires the `[viz]` extra dependency (PyQt6).

### Benchmark Options

* **`--scenario SCENARIO`** (default: `config/Scenario.toml`)  
  Scenario instance TOML.

* **`--environment ENVIRONMENT`** (default: `config/Environment.toml`)  
  Environment base TOML.

* **`--samples N`** (default: `1000`)  
  Number of random `t` samples.

* **`--seed N`** (default: `0`)  
  RNG seed for sample times.

---

### Linux/Wayland Visualization Troubleshooting

On Linux with Wayland, Qt windows (map visualizer) may fail to open or render incorrectly. Force the X11 backend via `QT_QPA_PLATFORM=xcb`:

```bash
QT_QPA_PLATFORM=xcb python -m satellite [...]
```

---

### Running Tests

To run the test suite, notably in automated environments:

1. **Install Test Dependencies**: If `pytest` is not already installed in your virtual environment:
   ```bash
   .venv/bin/pip install pytest
   ```

2. **Run Pytest with Environment Variables**: The project uses an auto-bootstrap mechanism in `src/satellite/__init__.py` that can cause issues or unexpected argument stripping if re-executed.
   To bypass this auto-bootstrap and run the tests correctly, set `SATELLITE_NO_GPU=1` and ensure the project path is in your `PYTHONPATH`:
   ```bash
   PYTHONPATH=src:. SATELLITE_NO_GPU=1 .venv/bin/pytest
   ```
