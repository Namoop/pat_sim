# Satellite Communication — SDA Two-Phase Search

Monte Carlo satellite link-establishment simulation. Two satellites alternate transmitting a spiral search cone on a shared clock:

| Global time `q` | Transmitter | Receiver |
|-----------------|-------------|----------|
| `[0, q_max)` | S1 | S2 |
| `[q_max, 2·q_max)` | S2 | S1 |

Each satellite carries both a transmitter and receiver. Satellites never communicate; they follow the synchronized schedule only. Phase 2 for S2 spirals around the dish boresight at the end of phase 1 (locked direction if the beam was acquired, otherwise initial mispoint).

Each satellite's spiral center uses **beam θ/φ offsets** applied to the true line-of-sight toward its partner.

## Install

```bash
# Headless (fast batch runs)
pip install -e .

# Interactive 3D visualization
pip install -e ".[viz]"
```

## Run

```bash
# Fast headless — prints hit summary
python -m satellite --config scenario.toml

# Interactive 3D window (orbit/pan, time slider, Play/Pause, Next scenario stub)
python -m satellite --config scenario.toml --visualize

# Start visualization at a specific time
python -m satellite --config scenario.toml --visualize --q 2.5
```

### Linux visualization troubleshooting

If you see `BadWindow` or `vtkXOpenGLRenderWindow` errors (common on **Wayland**), force Qt to use X11:

```bash
QT_QPA_PLATFORM=xcb python -m satellite --config scenario.toml --visualize
```

The visualizer also sets `QT_QPA_PLATFORM=xcb` automatically on Linux when the variable is unset.

## Configuration

Edit [`scenario.toml`](scenario.toml):

| Section | Key fields |
|---------|------------|
| `s1`, `s2` | `position`, `body_theta_offset`, `body_phi_offset`, `beam_theta_offset`, `beam_phi_offset`, `dish_theta_offset`, `dish_phi_offset` |
| `satellite` | `body_radius`, `dish_fov`, `dish_slew_time`, `beam_width` (milliradians) |
| `sda` | `k`, `gamma`, `beta`, `omega_r`, `L_r` |
| `simulation` | `q_max` (one phase), `q_step`, `boresight_extension` (default 5), optional `beam_length` |
| `visualization` | `enabled`, mesh resolution settings |

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
  visualize/      — PyVista + Qt (lazy-loaded)
```

## Model summary

- **Body aim:** per-satellite `body_theta/phi_offset` jumbles the entire spacecraft relative to the partner.
- **Beam aim:** `beam_theta/phi_offset` relative to the body frame (spiral center).
- **Dish aim:** `dish_theta/phi_offset` relative to the body frame (gimbal mispoint).
- **Acquisition slew:** rotates the **entire body** toward the incoming beam; dish and beam follow rigidly and may retain residual error.
- **Detection:** transmitter cone illuminates dish mount; incoming angle must be within `dish_fov`.
- **Phase 2 handoff:** S2 builds a new spiral centered on `boresight_end` from full phase-1 replay.
- **Replay:** headless and visualizer share one coupled replay loop — no separate static-dish scan.
