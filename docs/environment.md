# Hardware and Environment Configuration (`Environment.toml`)

This document describes the parameters defined in `Environment.toml`, which represent the spacecraft hardware constraints, simulation environment constants, and default visualizer settings. 

The configuration file is loaded by [`load_simulation_config`](../src/satellite/config.py) in `satellite/config.py`. All angular values, pointing offsets, and angular speeds in the environment file are defined in **milliradians** or **milliradians per second** (except where explicitly marked as physical meters or counts).

---

## Configuration Blocks

### `[satellite]` (Spacecraft Hardware Limits)

- `body_radius` (float, meters): Physical radius of the spacecraft body.
- `dish_fov` (float, milliradians): Full field-of-view of the receiver dish.
- `s2_fov_mod` (float): Modifier scaling factor for S2 receiver field-of-view (defaults to `1.0` if omitted). E.g. if `dish_fov = 2.0` and `s2_fov_mod = 0.5`, S2 receiver FOV will be `1.0`.
- `max_beam_speed` (float, milliradians/second): Maximum optical bench/beam slew speed.
- `max_fsm_speed` (float, milliradians/second): Maximum Fast Steering Mirror deflection speed.
- `beam_width` (float, milliradians): Full spread of the transmitter beam.
- `k` (float): Density multiplier for spiral strategies.
- `scan_envelope_ramp` (float, seconds): Duration over which offset-based scan patterns (Lissajous, rosette, raster, etc.) ramp amplitude from zero at nominal boresight to full strength. Set to `0` to disable (patterns begin at full amplitude immediately, which may start off-center). Default: `2.0`.
- `scan_envelope_profile` (string): Shape of the amplitude ramp. One of:
  - `smooth` (default): Smoothstep \(f(t) = 3(t/T)^2 - 2(t/T)^3\) — zero slope at start and end of the ramp.
  - `linear`: \(f(t) = t/T\) — constant growth rate.
  - `cosine`: \(f(t) = \tfrac{1}{2}(1 - \cos(\pi t/T))\) — cosine ease-in/out with zero slope at endpoints.

### `[simulation]` (Environment Constants & Solvers)

- `distance` (float, meters): Nominal distance between satellites (acts as fallback if scenario distance is not set).
- `t_step` (float, seconds): Simulation time step size.
- `beam_length` (float or null, meters): Hardcoded length of the transmit beam. If null, computed automatically as distance plus `boresight_extension`.
- `boresight_extension` (float, meters): Distance the beam extends past the target plane.
- `max_search_radius` (float, milliradians): Maximum search radius for strategies and boundary reflections.
- `profile_replay` (bool): If true, enables profiling of frame generation during playback.
- `timeout` (float, seconds): Maximum simulation duration.
- `enforce_speed_limit` (bool): If true, validates strategy timelines to ensure they respect `max_beam_speed`.

### `[3d_viz]` (3D Renderer Defaults)

- `cone_u_steps` (int): Number of longitudinal steps in the beam cone mesh.
- `cone_v_steps` (int): Number of angular steps in the beam cone mesh.
- `spiral_trail_steps` (int): Number of steps in the visual path trail.
- `ribbon_v_steps` (int): Number of steps across the swept ribbon width.
- `profile_frames` (bool): Enables profiling of visualization updates.
- `squish` (float, default `1`): Compression along the inter-satellite axis. Values above `1` shorten apparent link range and scale beam/FOV angles. Simulation physics are unaffected.

### `[eye_viz]` (2D Eye Defaults)

- `axis_limit` (float, milliradians): Limit of the visualizer's 2D angular map plots.
- `slider_debounce_ms` (int): Debounce time in milliseconds for visualizer UI slider inputs.

> [!NOTE]
> Named **Eye** because you are seeing the "eye" of each satellite, and since there are two satellites total, it looks kind of like eyes.

### `[mag_viz]` (2D Magnitude Defaults)

- `visual_limit_deg` (float): Visual angular limit in degrees.
- `fov_cone_length` (float): Default receiver FOV cone length (defaults to `1.0` if omitted).
- `beam_cone_length` (float): Default transmitter beam cone length (defaults to `1.0` if omitted).

> [!NOTE]
> Named **Mag** (short for magnitude) since all you see is in 2D representing the total magnitude offset.
