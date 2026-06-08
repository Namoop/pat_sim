# Satellite Communication — SDA Skeleton

Monte Carlo satellite link-establishment simulation. A transmitter at `P_1` spirals a laser cone around where it **believes** the receiver is (`P_2`), while the receiver **actually** sits at `P_t`.

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

# Interactive 3D window (orbit/pan, time slider, Next button)
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
| `positions` | `p1` transmitter, `p2` believed receiver, `pt` actual receiver |
| `offsets` | `theta_jumble`, `phi_jumble` — used to compute `pt` when omitted |
| `sda` | `k`, `alpha`, `gamma`, `beta`, `omega_r`, `L_r` |
| `simulation` | `q_max`, `q_step`, optional `beam_length` |
| `visualization` | `enabled`, mesh resolution settings |

## Project layout

```
src/satellite/
  math3d.py       — vector helpers
  geometry.py     — frames, cone surfaces, spiral trail
  detection.py    — in-cone hit test
  sda/            — transmitter & receiver strategies
  scenario.py     — orchestration
  visualize/      — PyVista + Qt (lazy-loaded)
```

## Desmos reference values

With default `scenario.toml`:

- `d ≈ 5.099`, `D ≈ (0, 0.196, 0.981)`
- `w ≈ 0.0533`
- `U_t` from `P_t - P_1`
- At `q = 5`: `theta_off ≈ 0.267`, `phi_off = 25`
