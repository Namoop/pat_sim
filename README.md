# Satellite Communication — SDA Meta-Strategy Search

Monte Carlo satellite link-establishment simulation. Two satellites search using a **meta-strategy chain** (`minor_offset`, `single_miss`, …). Each frame, both directions are checked for bidirectional lock; the first hit stops the run.

The spacecraft body is assumed correctly pointed. Launch mispoint is modeled as **optical-bench rotation**; dish and TX beam share the bench boresight.

## Install

```bash
pip install -e .
pip install -e ".[perf]"    # optional Numba-accelerated detection
pip install -e ".[viz]"      # interactive 3D (PyVista)
pip install -e ".[mapviz]"   # angular map (PyQt6)
```

## Run

```bash
# Single scenario
python -m satellite --scenario default.toml --simulation Simulation.toml

# Monte Carlo batch
python -m satellite --monte-carlo MonteCarlo.toml

# Visualization
python -m satellite --scenario default.toml --visualize map
python -m satellite --scenario default.toml --visualize 3d --q 2.5

# Map render benchmark
python -m satellite.mapviz.bench_render --scenario default.toml
```

Defaults: `--scenario default.toml`, `--simulation Simulation.toml`, `--strategy MonteCarlo.toml` (strategy chain loaded from the strategy section of MonteCarlo.toml).

### Linux visualization

```bash
QT_QPA_PLATFORM=xcb python -m satellite --scenario default.toml --visualize 3d
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
  visualize/      — PyVista 3D
  mapviz/         — QPainter angular map
```

## Model summary

- **Detection:** transmit cone hits dish mount; incoming direction from transmitter body must fall within `dish_fov`.
- **Search:** movement patterns set shared bench aim (dish == beam); FSM snaps on acquisition.
- **Replay:** headless and visualizer share one coupled replay timeline.
