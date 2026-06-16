# Simulation Configuration Reference

This document provides a comprehensive reference of all configuration parameters for the satellite establishment simulation. 

All angular values, pointing offsets, and angular speeds in the configuration files are defined in **milliradians** or **milliradians per second** (except where explicitly marked as math-only phase values or steering angles).

---

## 1. Scenario Instance Config (`default.toml`)

These parameters define a specific scenario run and are loaded by [load_scenario_config](file:///home/theodore/Documents/satellite/src/satellite/config.py).

### `[scenario]`
* **`name`** (string): Unique identifier for the scenario instance.
* **`chain`** (array of strings, required): Ordered list of search strategies to run (e.g., `["minor_offset", "single_miss"]`).
* **`simulation`** (string, optional): Path to the associated `Simulation.toml` file (resolved relative to the scenario file).
* **Arbitrary Property Overrides**: Any simulation or satellite property can be overridden for the specific scenario using dotted keys or nested sub-tables under `[scenario]`.
  * **Dotted Keys**: `simulation.distance = 500`, `simulation.t_step = 0.001`, or `satellite.max_fsm_radius = 0.5`.
  * **Nested Tables**:
    ```toml
    [scenario.simulation]
    distance = 500
    t_step = 0.001

    [scenario.satellite]
    max_fsm_radius = 0.5
    ```
  * *Note: Hardware and environment values that are defined in milliradians (such as `max_fsm_radius`, `dish_fov`, etc.) will automatically be scaled by $10^{-3}$ to radians when parsed as overrides.*

### `[s1]` and `[s2]` (Spacecraft Instances)
* **`bench_theta_offset`** (float, milliradians): Local theta pointing offset for the spacecraft's optical bench.
* **`bench_phi_offset`** (float, milliradians): Local phi pointing offset for the spacecraft's optical bench.

---

## 2. Hardware and Environment Config (`Simulation.toml`)

These parameters define spacecraft hardware limits, simulation steps, and visualizer properties. Loaded by [load_simulation_config](file:///home/theodore/Documents/satellite/src/satellite/config.py).

### `[satellite]` (Hardware Properties)
* **`body_radius`** (float, meters): Physical radius of the spacecraft body.
* **`dish_fov`** (float, milliradians): Full field-of-view of the receiver dish.
* **`max_beam_speed`** (float, milliradians/second): Maximum bench/beam slew speed.
* **`max_fsm_speed`** (float, milliradians/second): Maximum Fast Steering Mirror deflection speed.
* **`beam_width`** (float, milliradians): Full-width half-max or spread of the transmitter beam.
* **`k`** (float): Density multiplier for spiral strategies.

### `[simulation]` (Environment and Ingestion)
* **`distance`** (float, meters): Nominal distance between satellites (fallback value if scenario distance is not set).
* **`t_step`** (float, seconds): Simulation time step size.
* **`beam_length`** (float or null, meters): Hardcoded length of the transmit beam. If null, computed automatically as distance plus `boresight_extension`.
* **`boresight_extension`** (float, meters): Distance the beam extends past the target plane.
* **`max_search_radius`** (float, milliradians): Maximum search radius for strategies and boundary reflections.
* **`profile_replay`** (bool): If true, enables profiling of frame generation during playback.
* **`timeout`** (float, seconds): Maximum simulation duration.
* **`enforce_speed_limit`** (bool): If true, validates strategy timelines to ensure they respect `max_beam_speed`.

### `[visualization]` (3D Renderer Settings)
* **`enabled`** (bool): Enables the 3D rendering visualizer.
* **`cone_u_steps`** (int): Number of longitudinal steps in the beam cone mesh.
* **`cone_v_steps`** (int): Number of angular steps in the beam cone mesh.
* **`spiral_trail_steps`** (int): Number of steps in the visual path trail.
* **`ribbon_v_steps`** (int): Number of steps across the swept ribbon width.
* **`profile_frames`** (bool): Enables profiling of visualization updates.

### `[map_visualization]` (2D Map Settings)
* **`axis_limit`** (float, milliradians): Limit of the visualizer's 2D angular map plots.
* **`slider_debounce_ms`** (int): Debounce time in milliseconds for visualizer UI slider inputs.

---

## 3. Monte Carlo and Error Config (`MonteCarlo.toml` / `MonteCarlo_all.toml`)

These parameters define jumble distributions, random seeds, and strategy parameter groups. Loaded by [load_monte_carlo_config](file:///home/theodore/Documents/satellite/src/satellite/config.py).

### `[monte_carlo]`
* **`simulation`** (string): Path to the associated `Simulation.toml` file.
* **`seed`** (int): Base random seed for reproducing jumble errors.
* **`runs`** (int): Number of batch runs to execute.
* **`chain`** (array of strings): Ordered list of search strategies to run (e.g., `["minor_offset", "single_miss"]`).

### `[error]` (Launch Jumble Distributions)
* **`distribution`** (string): Type of error distribution (`"uniform"` or `"gaussian"`).
* **`theta_min`** / **`theta_max`** (float, milliradians): Bounds for uniform error on theta offset.
* **`phi_min`** / **`phi_max`** (float, milliradians): Bounds for uniform error on phi offset.
* **`theta_mean`** / **`theta_std`** (float, milliradians): Parameters for gaussian error on theta offset.
* **`phi_mean`** / **`phi_std`** (float, milliradians): Parameters for gaussian error on phi offset.

### `[strategy]` (Base Strategy Settings)
* (This section acts as a container for general settings, such as `chain` list below, but no longer contains parameter variables).

---

## 4. Strategy-Specific Config Parameters

These parameters tune the behavior of individual strategies, parsed via [CONFIG_PARSERS](file:///home/theodore/Documents/satellite/src/satellite/strategy/base.py).

### `[strategy.minor_offset]` (FOV-limited spiral)
* **`max_spiral_radius`** (float or `"fov"`, milliradians): Limit of the spiral search radius. `"fov"` dynamically matches `dish_fov`.
* **`spiral_speed`** (float): Speed multiplier for the spiral track.

### `[strategy.single_miss]` (Two-phase alternating spiral)
* **`a_spiral_radius`** (float or `"fov"`, milliradians): Spiral radius for S1.
* **`b_spiral_radius`** (float or `"fov"`, milliradians): Spiral radius for S2.
* **`spiral_speed`** (float): Speed multiplier for the spirals.

### `[strategy.asymmetric_swap]` (Alternating probe and listen)
* **`spiral_radius`** (float or `"fov"`, milliradians): Probe spiral radius.
* **`lock_duration`** (float, seconds): Hold duration required to declare lock.
* **`spiral_speed`** (float): Speed multiplier for the spirals.

### `[strategy.center_rebias]` (Stochastic center-biased pull)
* **`velocity_a`** (float, milliradians/second): Base search velocity for S1.
* **`velocity_ratio`** (float): Ratio of S2 velocity to S1 velocity (`velocity_b = velocity_a * velocity_ratio`).
* **`drift_sigma`** (float, milliradians/second): Standard deviation of steering wheel drift.
* **`max_turn_radius`** (float, milliradians): Steering angle clamp limit.
* **`bias_strength`** (float): Attraction factor back to boresight center.
* **`seed`** (int): Local RNG seed.

### `[strategy.random_curve]` (Continuous random drift)
* **`velocity_a`** (float, milliradians/second): Base search velocity for S1.
* **`velocity_ratio`** (float): Ratio of S2 velocity to S1 velocity.
* **`drift_sigma`** (float, milliradians/second): Standard deviation of steering drift.
* **`max_turn_radius`** (float, milliradians): Steering angle clamp limit.
* **`seed`** (int): Local RNG seed.

### `[strategy.lissajous_scan]` (Sinusoidal orthogonal scanning)
* **`s1_wx`** / **`s1_wy`** (float, rad/s): Sinusoidal frequencies for S1.
* **`s1_delta`** (float, **radians**): Phase offset for S1 (**kept in radians**).
* **`s2_wx`** / **`s2_wy`** (float, rad/s): Sinusoidal frequencies for S2.
* **`s2_delta`** (float, **radians**): Phase offset for S2 (**kept in radians**).

### `[strategy.rosette_scan]` (Flower-like pattern)
* **`s1_w1`** / **`s1_w2`** (float, rad/s): Frequencies for S1.
* **`s2_w1`** / **`s2_w2`** (float, rad/s): Frequencies for S2.

### `[strategy.dual_spiral]`
* **`speed_a`** (float): Base spiral speed.
* **`speed_ratio`** (float): Ratio of S2 to S1 spiral speed.

### `[strategy.dual_raster]`
* **`steps_a`** (int): Raster grid lines for S1.
* **`steps_b`** (int): Raster grid lines for S2.
* **`speed_a`** (float): Base scan speed.
* **`speed_ratio`** (float): Ratio of S2 to S1 scan speed.

### `[strategy.concentric_shells]`
* **`radii_factors`** (array of floats): Ratios of max search radius (e.g. `[0.2, 0.5, 1.0]`).
* **`spiral_speed_a`** (float): Base spiral speed.
* **`speed_ratio`** (float): Ratio of S2 to S1 spiral speed.
