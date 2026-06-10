# Satellite Communication — SDA Two-Phase Search

Monte Carlo satellite link-establishment simulation. Two satellites alternate transmitting a spiral search cone on a shared clock:

| Global time `q` | Transmitter | Receiver |
|-----------------|-------------|----------|
| `[0, q_max)` | S1 | S2 |
| `[q_max, 2·q_max)` | S2 | S1 |

Each satellite carries both a transmitter and receiver. Satellites never communicate; they follow the synchronized schedule only. Phase 2 for S2 spirals around the **bench boresight** at the end of phase 1 (locked direction if the beam was acquired, otherwise initial mispoint).

The spacecraft body is assumed correctly pointed. Launch mispoint is modeled as **optical-bench rotation**; dish and TX beam share the bench boresight.

## Install

```bash
# Headless (fast batch runs)
pip install -e .

# Optional Numba-accelerated detection
pip install -e ".[perf]"

# Interactive 3D visualization
pip install -e ".[viz]"

# Angular map visualization (QPainter θ/φ view, PyQt6 only)
pip install -e ".[mapviz]"
```

## Run

```bash
# Fast headless — prints hit summary
python -m satellite --config scenario.toml

# Interactive 3D window (orbit/pan, time slider, Play/Pause)
python -m satellite --config scenario.toml --visualize 3d
# or shorthand:
python -m satellite --config scenario.toml --visualize

# Angular map view (two side-by-side θ/φ panels)
python -m satellite --config scenario.toml --visualize map

# Benchmark map render path (QPainter p50/p95 timings)
python -m satellite.mapviz.bench_render --config scenario.toml

# Start visualization at a specific time
python -m satellite --config scenario.toml --visualize 3d --q 2.5
```

### Linux visualization troubleshooting

If you see `BadWindow` or `vtkXOpenGLRenderWindow` errors (common on **Wayland**), force Qt to use X11:

```bash
QT_QPA_PLATFORM=xcb python -m satellite --config scenario.toml --visualize 3d
QT_QPA_PLATFORM=xcb python -m satellite --config scenario.toml --visualize map
```

The visualizer also sets `QT_QPA_PLATFORM=xcb` automatically on Linux when the variable is unset.

## Configuration

Edit [`scenario.toml`](scenario.toml):

| Section | Key fields |
|---------|------------|
| `s1`, `s2` | `position`, `bench_theta_offset`, `bench_phi_offset` |
| `satellite` | `body_radius`, `dish_fov`, `bench_slew_time`, `fsm_settle_time`, `beam_width` (milliradians) |
| `sda` | `k`, `gamma`, `beta`, `omega_r`, `L_r` |
| `simulation` | `q_max` (one phase), `q_step`, `boresight_extension` (default 5), optional `beam_length` |
| `visualization` | `enabled`, mesh resolution settings (3D PyVista) |
| `map_visualization` | `axis_limit`, `disc_segments`, `spiral_trail_steps` (angular map) |

`q_max` is the duration of **one** spiral phase; the full search runs for `2 * q_max`.

`beam_width` is the transmitter cone half-angle in **milliradians** (e.g. `5.0` → α = 0.005 rad).

Default beam length and boresight ray length = link range + `boresight_extension` (5 units unless overridden).

## Project layout

```
src/satellite/
  math3d.py       — vector helpers
  geometry.py     — frames, cone surfaces, spiral trail
  detection.py    — dish FOV hit test
  schedule.py     — two-phase SearchSchedule
  sda/            — TransmitterSDA, ReceiverSDA, Satellite
  scenario.py     — orchestration and coupled replay
  visualize/      — PyVista + Qt 3D (lazy-loaded)
  mapviz/         — QPainter + Qt angular map (lazy-loaded)
```

## Model summary

- **Body:** assumed correctly pointed at the partner (no body slew).
- **Optical bench:** per-satellite `bench_theta/phi_offset` is the sole launch mispoint; dish and TX beam are co-aligned on the bench (spiral center).
- **Acquisition:** on detect, the **FSM** (fast steering mirror) snaps to center the beam on the camera; the **bench** then slews slowly to recenter the FSM.
- **Detection:** transmitter cone illuminates dish mount; incoming angle must be within `dish_fov`.
- **Phase 2 handoff:** S2 builds a new spiral centered on `boresight_end` (end-of-phase-1 bench boresight).
- **Replay:** headless and visualizer share one coupled replay loop — no separate static-dish scan.
- **Performance:** spiral boresights are precomputed per phase; optional Numba kernel for detection (`pip install -e ".[perf]"`).
