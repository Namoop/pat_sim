# Satellite Communication — SDA Meta-Strategy Search

Monte Carlo satellite link-establishment simulation. Two satellites search using a **meta-strategy chain** (`minor_offset`, `asymmetric_probe`, `single_miss`, …). Success requires **mutual lock**: both beams enabled, both receivers enabled, both bench slews complete, and simultaneous bidirectional visibility.

The spacecraft body is assumed correctly pointed. Launch mispoint is modeled as **optical-bench rotation**; dish and TX beam share the bench boresight.

## Install

```bash
pip install -e .
pip install -e ".[perf]"    # optional Numba-accelerated detection
pip install -e ".[viz]"      # interactive 3D + angular map (PyVista, PyQt6)
```

## Run

```bash
python -m satellite [options]
```

Run a satellite SDA communication scenario. Pass `--help` for the same option list in the terminal.

**Options**

`--visualize [{3d,map}]`  
Open unified visualization window after the run. Optional `3d` or `map` picks the **initial tab** (3D PyVista view or angular θ/φ map). `--visualize` alone is equivalent to `--visualize 3d`. If omitted, the window opens when `[visualization].enabled` is true in the simulation config. With `--monte-carlo`, opens an interactive step-through mode: **Next** runs the next sampled scenario (or closes on the last run / single scenario). On successful runs the timeline ends at mutual lock — replay cache, slider, and playback cannot scrub past that point. Playback controls sit above the view; the **event log** (system, S1, S2) is in a three-column strip at the bottom.

`--monte-carlo MONTE_CARLO`  
Run Monte Carlo from `MonteCarlo.toml`. Without `--visualize` (and with `[visualization].enabled` false), runs the full batch headlessly and prints a summary. With visualization enabled, runs one scenario at a time in the visualizer; use **Next** to advance.

`--scenario SCENARIO` (default: `default.toml`)  
Scenario instance TOML: bench offsets and optional `[scenario].distance` override.

`--simulation SIMULATION` (default: `Simulation.toml`)  
Simulation base TOML: hardware, timing, distance, `t_step`, and visualization defaults.

`--strategy STRATEGY` (default: `MonteCarlo.toml`)  
Strategy chain TOML. The `[strategy]` section and per-strategy parameter tables are read from this file.

`--t T` (default: `0`)  
Starting time `t` when opening a visualizer.

**Examples**

```bash
python -m satellite
python -m satellite --monte-carlo MonteCarlo.toml
python -m satellite --visualize map --t 2.5
```

### Map render benchmark

```bash
python -m satellite.mapviz.bench_render [options]
```

Benchmark mapviz QPainter render path (headless Qt). Requires `[viz]` (PyQt6). Pass `--help` for options.

**Options**

`--scenario SCENARIO` (default: `default.toml`)  
Scenario instance TOML.

`--simulation SIMULATION` (default: `Simulation.toml`)  
Simulation base TOML.

`--strategy STRATEGY` (default: `MonteCarlo.toml`)  
Strategy chain TOML.

`--samples N` (default: `1000`)  
Number of random `t` samples.

`--seed N` (default: `0`)  
RNG seed for sample times.

### Linux+Wayland visualization

On Linux with Wayland, Qt windows (map visualizer) may fail to open or render incorrectly. Force the X11 backend via `QT_QPA_PLATFORM=xcb`:

```bash
QT_QPA_PLATFORM=xcb python -m satellite [...]
```

## Configuration


| File                                 | Purpose                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| `[Simulation.toml](Simulation.toml)` | Hardware, `distance`, `t_step`, visualization                  |
| `[default.toml](default.toml)`       | Per-run bench offsets; optional `[scenario].distance` override |
| `[MonteCarlo.toml](MonteCarlo.toml)` | MC runs, error distribution, strategy chain                    |


Satellites are placed on the **x axis**: S1 at origin, S2 at `[distance, 0, 0]`.

Strategy step durations are explicit in `[strategy.*]` tables (total sim time = sum of attempt script durations in the chain). Movement `duration=0` is only valid for a no-op bench reset when already at the initial boresight; future work will auto-compute durations from beam-director max slew rate.

`beam_width` is the transmitter cone half-angle in **milliradians** (e.g. `5.0` → α = 0.005 rad).

### Monte Carlo Multi-Chain Configuration

You can run multiple independent strategy chains in a single Monte Carlo run. Each chain has its own sequence of strategies and number of runs. The pseudorandom seed generator resets for each chain to ensure that the error distribution of runs is exactly reproducible, independent of the order or number of other chains in the file.

Example configuration in `MonteCarlo.toml`:

```toml
[monte_carlo]
simulation = "Simulation.toml"
seed = 42

[[monte_carlo.chains]]
runs = 500
chain = ["minor_offset", "single_miss"]

[[monte_carlo.chains]]
runs = 300
chain = ["asymmetric_swap"]
```

### Built-in strategies


| Name                | Behavior                                                                  |
| ------------------- | ------------------------------------------------------------------------- |
| `minor_offset`      | Both TX/RX on; S1 FOV spiral, S2 holds                                    |
| `single_miss`       | Alternating wide spirals with bench reset between phases                  |
| `asymmetric_swap`   | S1 probes with RX off and resets, swap roles to establish reciprocal lock |
| `dual_spiral`       | Both satellites execute spirals simultaneously                            |
| `dual_raster`       | Satellites perform orthogonal raster scans (one horizontal, one vertical) |
| `lissajous_scan`    | Continuous Lissajous figure scan                                          |
| `rosette_scan`      | Rosette-pattern scan from center boresight                                |
| `center_rebias`     | Stochastic search with periodic center resets                             |
| `concentric_shells` | Progressive depth concentric circle scans                                 |
| `random_walk`       | Stochastic step-by-step random walk                                       |
| `random_curve`      | Smooth random walk in angle space                                         |
| `nested_spiral`     | Concentric Archimedean spirals                                            |


Custom strategies use the Python DSL in `strategy/actions.py`; TOML configures built-in chain parameters only.

## Project layout

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

## Model summary

- **Detection:** transmit cone hits dish mount; incoming direction from transmitter body must fall within `dish_fov`.
- **Search:** independent per-satellite timelines (hold, spiral, reset, …) with optional beam/receiver enable states; FSM snaps on acquisition.
- **Lock:** both satellites transmitting and receiving, both slews complete, simultaneous `visible_12 ∧ visible_21`.
- **Partial acquisition:** if one satellite acquires the other before a strategy times out, the acquirer keeps tracking and ignores later scripted search; the non-acquired satellite continues the strategy chain normally.
- **Replay:** headless and visualizer share one coupled replay timeline, capped at lock time on successful runs.

## Running Tests

To run the test suite, notably in automated environments:

1. **Install Test Dependencies**: If `pytest` is not already installed in your virtual environment:
  ```bash
   .venv/bin/pip install pytest
  ```
2. **Run Pytest with Environment Variables**: The project uses an auto-bootstrap mechanism in `[src/satellite/__init__.py](file:///home/theodore/Documents/satellite/src/satellite/__init__.py)` that can cause issues or unexpected argument stripping if re-executed.
  To bypass this auto-bootstrap and run the tests correctly, set `SATELLITE_NO_GPU=1` and ensure the project path is in your `PYTHONPATH`:

