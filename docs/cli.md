# CLI Usage Reference

This document provides a detailed reference for running the satellite simulation and related command-line utilities.

## Main Command

```bash
python -m satellite [options]
```

Runs a satellite SDA communication scenario.

### Command-line Options

* **`--visualize [{3d,map}]`**  
  Opens the unified visualization window after the run. The optional choice `3d` or `map` picks the **initial tab** (3D PyVista view or angular θ/φ map). `--visualize` alone is equivalent to `--visualize 3d`. If omitted, the window opens when `[scenario].visualize` is specified as `"3d"` or `"map"` in the scenario config. With `--monte-carlo`, opens an interactive step-through mode: **Next** runs the next sampled scenario (or closes on the last run / single scenario). On successful runs the timeline ends at mutual lock — replay cache, slider, and playback cannot scrub past that point. Playback controls sit above the view; the **event log** (system, S1, S2) is in a three-column strip at the bottom.

* **`--monte-carlo MONTE_CARLO`**  
  Runs a Monte Carlo batch from the specified TOML file (e.g. `config/MonteCarlo.toml` or `config/MC_basic.toml`). Without `--visualize`, runs the full batch headlessly and prints a summary. With visualization enabled (via `--visualize`), runs one scenario at a time in the visualizer; use **Next** to advance.

* **`--scenario SCENARIO`** (default: `config/default.toml`)  
  Path to the Scenario instance TOML containing bench offsets, strategy chain, and optional overrides.

* **`--simulation SIMULATION`** (default: `config/Simulation.toml`)  
  Path to the Simulation base TOML containing hardware, timing, distance, `t_step`, and visualization defaults.

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

## Map Render Benchmark

```bash
python -m satellite.mapviz.bench_render [options]
```

Benchmark the mapviz QPainter render path (headless Qt). Requires the `[viz]` extra dependency (PyQt6).

### Benchmark Options

* **`--scenario SCENARIO`** (default: `config/default.toml`)  
  Scenario instance TOML.

* **`--simulation SIMULATION`** (default: `config/Simulation.toml`)  
  Simulation base TOML.

* **`--samples N`** (default: `1000`)  
  Number of random `t` samples.

* **`--seed N`** (default: `0`)  
  RNG seed for sample times.

---

## Linux/Wayland Visualization Troubleshooting

On Linux with Wayland, Qt windows (map visualizer) may fail to open or render incorrectly. Force the X11 backend via `QT_QPA_PLATFORM=xcb`:

```bash
QT_QPA_PLATFORM=xcb python -m satellite [...]
```

---

## Running Tests

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
