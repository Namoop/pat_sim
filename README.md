# Satellite Communication — SDA Meta-Strategy Search

Monte Carlo satellite link-establishment simulation. Two satellites search using a **meta-strategy chain** (`minor_offset`, `single_miss`, …). Each frame, both directions are checked for bidirectional lock; the first hit stops the run.

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
Open unified visualization window after the run. Optional `3d` or `map` picks the **initial tab** (3D PyVista view or angular θ/φ map). `--visualize` alone is equivalent to `--visualize 3d`. If omitted, the window opens when `[visualization].enabled` is true in the simulation config. With `--monte-carlo`, opens an interactive step-through mode: **Next** runs the next sampled scenario (or closes on the last run / single scenario).

`--monte-carlo MONTE_CARLO`  
Run Monte Carlo from `MonteCarlo.toml`. Without `--visualize` (and with `[visualization].enabled` false), runs the full batch headlessly and prints a summary. With visualization enabled, runs one scenario at a time in the visualizer; use **Next** to advance.

`--scenario SCENARIO` (default: `default.toml`)  
Scenario instance TOML: bench offsets and optional `[scenario].distance` override.

`--simulation SIMULATION` (default: `Simulation.toml`)  
Simulation base TOML: hardware, timing, distance, `q_step`, and visualization defaults.

`--strategy STRATEGY` (default: `MonteCarlo.toml`)  
Strategy chain TOML. The `[strategy]` section (and nested epoch tables) is read from this file.

`--q Q` (default: `0`)  
Starting time `q` when opening a visualizer.

**Examples**

```bash
python -m satellite
python -m satellite --monte-carlo MonteCarlo.toml
python -m satellite --visualize map --q 2.5
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
Number of random `q` samples.

`--seed N` (default: `0`)  
RNG seed for sample times.

### Linux+Wayland visualization

On Linux with Wayland, Qt windows (map visualizer) may fail to open or render incorrectly. Force the X11 backend via `QT_QPA_PLATFORM=xcb`:

```bash
QT_QPA_PLATFORM=xcb python -m satellite [...]
```

## Configuration

| File | Purpose |
|------|---------|
| [`Simulation.toml`](Simulation.toml) | Hardware, `distance`, `q_step`, visualization |
| [`default.toml`](default.toml) | Per-run bench offsets; optional `[scenario].distance` override |
| [`MonteCarlo.toml`](MonteCarlo.toml) | MC runs, error distribution, strategy chain |

Satellites are placed on the **x axis**: S1 at origin, S2 at `[distance, 0, 0]`.

Strategy epoch durations are explicit in `[strategy.*]` (total sim time = sum of epoch durations in the winning attempt chain).

`beam_width` is the transmitter cone half-angle in **milliradians** (e.g. `5.0` → α = 0.005 rad).

## Project layout

```
src/satellite/
  config.py       — Simulation / scenario / MC loaders
  monte_carlo.py  — error sampling and batch runner
  strategy/       — meta-strategy, movements, frame runner
  sda/            — bench, transmitter, receiver
  scenario.py     — orchestration and replay
  visualize/      — unified 3D + map visualizer
  mapviz/         — QPainter angular map panels
```

## Model summary

- **Detection:** transmit cone hits dish mount; incoming direction from transmitter body must fall within `dish_fov`.
- **Search:** movement patterns set shared bench aim (dish == beam); FSM snaps on acquisition.
- **Replay:** headless and visualizer share one coupled replay timeline.
