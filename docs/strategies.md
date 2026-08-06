# Comprehensive Satellite Search Strategies

This document provides a detailed overview of the search strategies available in the `satellite` simulation suite. Each strategy is designed to establish a mutual optical communication link between two satellites (S1 and S2) under varying levels of initial pointing uncertainty.
## Built-in Strategies Overview

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
| `nested_spiral`     | Concentric Archimedean spirals |

Custom strategies are implemented using the Python DSL in `src/strategy/actions.py`; TOML files configure built-in chain parameters.

---

## Global Search Constraints

Several strategies reference a global bounding volume for the search. This is typically configured in `Environment.toml`.


| Parameter             | Default  | Description                                                                                                   |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `max_search_radius`   | 0.07 rad | The hard angular limit for any search pattern. Boresights will not exceed this radius from the initial guess. |
| `scan_envelope_ramp`  | 2.0 s    | Amplitude ramp duration so offset-based scans start at nominal boresight. `0` disables. See [environment.md](environment.md). |
| `scan_envelope_profile` | `smooth` | Ramp shape: `smooth`, `linear`, or `cosine`. See [environment.md](environment.md). |


---

## 1. Minor Offset (`minor_offset`)

### Description

The **Minor Offset** strategy is the simplest search pattern, specifically intended for use when the search radius matches the receiver's **FOV**. 

This strategy assumes that the pointing errors for both satellites are less than the FOV. In this scenario, S1 performs an Archimedean spiral search while S2 remains stationary at its initial "best guess" boresight. Because both satellites share the same FOV, having S1 spiral out to a radius matching that FOV **guarantees** that its beam will cross S2's receiver FOV.

**Dynamic Duration:** The time spent searching is automatically calculated based on the receiver's FOV and spiral speed: $T = FOV / (w \cdot \text{speed})$.

### Configuration Parameters

* `max_spiral_radius` (float or `"fov"`, milliradians): Limit of the spiral search radius. `"fov"` dynamically matches `dish_fov`.
* `spiral_speed` (float): Speed multiplier for the spiral track.


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable()
receiver.enable()
spiral(duration=calc_duration, radius=receiver.fov, speed=spiral_speed)
```

**Satellite S2:**

```python
beam.enable()
receiver.enable()
hold(duration=calc_duration)
```

---

## 2. Single Miss (`single_miss`)

### Description

The **Single Miss** strategy handles larger offsets where a simple FOV search might fail. It uses a two-phase alternating approach. 

- **Phase 1:** S1 performs a wide spiral search while S2 holds position. 
- **Reset:** Both satellites reset their optical benches to the initial boresight.
- **Phase 2:** S2 performs a wide spiral search while S1 holds position.

**Derived Timing:** 

- **Search Durations:** Calculated from the radius and spiral speed ($T = R / (w \cdot \text{speed})$).
- **Reset Durations:** Calculated from the maximum radius and the satellite's `max_beam_speed` hardware parameter ($T_{reset} = R / \text{maxbeamspeed}$). All mechanical movements, including slewing, are limited by this physical speed.

### Configuration Parameters

* `a_spiral_radius` (float or `"fov"`, milliradians): Spiral radius for S1.
* `b_spiral_radius` (float or `"fov"`, milliradians): Spiral radius for S2.
* `spiral_speed` (float): Speed multiplier for the spirals.                                             


### Action Script (Pseudocode)

**Satellite S1:**

```python
# Phase 1
beam.enable(); receiver.enable()
spiral(duration=calc_phase1, radius=spiral_radius)

# Reset & Phase 2
reset(duration=calc_reset)
hold(duration=calc_phase2)
```

**Satellite S2:**

```python
# Phase 1
beam.enable(); receiver.enable()
hold(duration=calc_phase1)

# Reset & Phase 2
hold(duration=calc_reset)
spiral(duration=calc_phase2, radius=spiral_radius)
```

---

## 3. Asymmetric Swap (`asymmetric_swap`)

### Description

The **Asymmetric Swap** strategy is an evolution of the asymmetric probe, designed to mitigate "mutual blindness" and minimize internal interference (glint). It designates initial "Leader" and "Follower" roles, but swaps them midway through the search.

- **Phase 1 (S1 Probes):** S1 performs a spiral search with its beam enabled and receiver disabled. S2 keeps its beam disabled and listens with its receiver.
- **Role Swap:** Both satellites reset their optical benches.
- **Phase 2 (S2 Probes):** S2 performs a spiral search with its beam enabled and receiver disabled. S1 keeps its beam disabled and listens with its receiver.
- **Reciprocal Lock Phase:** Finally, both satellites enable both hardware subsystems and hold their center positions to verify a mutual lock.

**Derived Timing:** 

- **Probe Durations:** Calculated from the radius and spiral speed.
- **Reset Durations:** Calculated from the maximum radius and `max_beam_speed` ($T_{reset} = R / \text{maxbeamspeed}$). All mechanical movements are limited by this physical speed.

### Configuration Parameters

* `spiral_radius` (float or `"fov"`, milliradians): Probe spiral radius.
* `lock_duration` (float, seconds): Hold duration required to declare lock.
* `spiral_speed` (float): Speed multiplier for the spirals.
* `s2_radius_mod` (float): Scale multiplier for S2's spiral search radius (defaults to `1.0` if omitted).


### Action Script (Pseudocode)

**Satellite S1 (Initial Leader):**

```python
# Phase 1: Probing
beam.enable(); receiver.disable()
spiral(duration=calc_phase1, radius=spiral_radius)
reset(duration=calc_reset)

# Phase 2: Listening (Swap)
beam.disable(); receiver.enable()
hold(duration=calc_phase2)
reset(duration=calc_reset)

# Phase 3: Reciprocal Lock
beam.enable(); receiver.enable()
hold(duration=lock_duration)
```

**Satellite S2 (Initial Follower):**

```python
# Phase 1: Listening
beam.disable(); receiver.enable()
hold(duration=calc_phase1)
hold(duration=calc_reset)

# Phase 2: Probing (Swap)
beam.enable(); receiver.disable()
spiral(duration=calc_phase2, radius=spiral_radius)
reset(duration=calc_reset)

# Phase 3: Reciprocal Lock
beam.enable(); receiver.enable()
hold(duration=lock_duration)
```

---

## 4. Dual Spiral (`dual_spiral`)

### Description

In the **Dual Spiral** strategy, both satellites perform an Archimedean spiral search simultaneously. To maximize the probability of beam intersection and avoid "phase locking" where the satellites remain at the same relative phase, their expansion rates (or angular frequencies) are related by an irrational ratio, typically $\sqrt{2}$.

Both satellites spiral from the center $(0,0)$ out to the configured search `radius`, and then immediately spiral back from the maximum radius to the center. This "in-and-out" motion ensures that the search covers the central high-probability region twice per cycle.

### Configuration Parameters

* `radius` (float, milliradians, optional): Spiral envelope radius. Defaults to `simulation.max_search_radius` when omitted.
* `k_ratio` (float): Ratio relating S2 spiral pitch/frequency to S1 (default `1.0`).
* `s2_hold_delay` (float, seconds): Duration S2 holds at boresight before starting its spiral (default `0`).
* `phase_offset` (float, radians): Phase offset applied to S2's spiral (default `0`).


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
spiral(radius=radius, ...)
spiral(radius=0, ...)
```

**Satellite S2:**

```python
beam.enable(); receiver.enable()
hold(duration=s2_hold_delay)  # skipped when delay is 0
spiral(radius=radius, phase_offset=phase_offset, ...)
spiral(radius=0, phase_offset=phase_offset, ...)
```

---

## 5. Dual Orthogonal Raster (`dual_raster`)

### Description

The **Dual Raster** strategy utilizes orthogonal scanning patterns to ensure coverage. Satellite S1 scans the circular search area using horizontal rows (rastering in X, stepping in Y), while S2 scans using vertical columns (rastering in Y, stepping in X). Both satellites cover the circular uncertainty volume defined by the global `max_search_radius`.

This strategy uses a **Boustrophedon (Serpentine)** motion: at the end of each line, the beam steps to the next elevation/offset and immediately scans in the opposite direction. This eliminates "flyback" resets and minimizes wasted movement time.

### Configuration Parameters

* `steps_a` (int): Raster grid lines for S1.
* `steps_b` (int): Raster grid lines for S2.
* `speed_a` (float): Base scan speed.
* `speed_ratio` (float): Ratio of S2 to S1 scan speed (`speed_b = speed_a * speed_ratio`).


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
# serpentine=True ensures no flyback
raster_horizontal(radius=max_search_radius, steps=steps_a, serpentine=True)
```

**Satellite S2:**

```python
beam.enable(); receiver.enable()
raster_vertical(radius=max_search_radius, steps=steps_b, serpentine=True)
```

---

## 6. Lissajous Scan (`lissajous_scan`)

### Description

The **Lissajous Scan** uses sinusoidal motion in orthogonal axes to create a complex, area-filling curve. This is highly efficient for hardware that can drive its axes at specific resonant frequencies.

The boresight position $(x, y)$ is given by:
$x(t) = \text{maxsearchradius} \cdot \sin(\omega_x t + \delta)$
$y(t) = \text{maxsearchradius} \cdot \sin(\omega_y t)$

By selecting a frequency ratio $\omega_x / \omega_y$ that is near-irrational or a large-denominator rational, the curve will fill the rectangular bounding box of the configured search `radius` without repeating itself for a long duration.

### Configuration Parameters

* `radius` (float, milliradians, optional): Pattern envelope radius. Defaults to `simulation.max_search_radius` when omitted.
* `s1_wx` / `s1_wy` (float, rad/s): Sinusoidal frequencies for S1.
* `s1_delta` (float, radians): Phase offset for S1 (kept in radians).
* `s2_wx` / `s2_wy` (float, rad/s): Sinusoidal frequencies for S2.
* `s2_delta` (float, radians): Phase offset for S2 (kept in radians).
* `s2_hold_delay` (float, seconds): Duration S2 holds at boresight before starting its scan (default `0`).


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
lissajous(A=radius, wx=wx, wy=wy, delta=delta, duration=timeout)
```

**Satellite S2:**

```python
beam.enable(); receiver.enable()
hold(duration=s2_hold_delay)  # skipped when delay is 0
lissajous(A=radius, wx=wx, wy=wy, delta=delta, duration=timeout - s2_hold_delay)
```

---

## 7. Rosette Scan (`rosette_scan`)

### Description

The **Rosette Scan** produces a flower-like pattern that provides extremely high density at the center of the search area while still covering the peripheral field of regard. The pattern is defined by the superposition of two counter-rotating circular motions.

The boresight position $(x, y)$ is given by:
$x(t) = \text{radius} \cdot \cos(\omega_1 t)$
$y(t) = \text{radius} \cdot \cos(\omega_2 t)$

If the ratio $\omega_1 / \omega_2$ is irrational, the pattern will eventually fill the entire circular area bounded by `radius`.

### Configuration Parameters

* `radius` (float, milliradians, optional): Pattern envelope radius. Defaults to `simulation.max_search_radius` when omitted.
* `s1_w1` / `s1_w2` (float, rad/s): counter-rotating frequencies for S1.
* `s2_w1` / `s2_w2` (float, rad/s): counter-rotating frequencies for S2.
* `s2_hold_delay` (float, seconds): Duration S2 holds at boresight before starting its scan (default `0`).


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
rosette(A=radius, w1=s1_w1, w2=s1_w2, duration=timeout)
```

**Satellite S2:**

```python
beam.enable(); receiver.enable()
hold(duration=s2_hold_delay)  # skipped when delay is 0
rosette(A=radius, w1=s2_w1, w2=s2_w2, duration=timeout - s2_hold_delay)
```

---

## 8. Stochastic Center Re-bias (`center_rebias`)

### Description

This is a **Center-Heavy** stochastic strategy. It performs a Random Walk, but with a "spring-like" force pulling the boresight back toward the center $(0,0)$. This ensures that the high-probability central region is sampled much more frequently than the edges, which is ideal for Gaussian pointing errors.

### Configuration Parameters

* `velocity_a` (float, milliradians/second): Base search velocity for S1.
* `velocity_ratio` (float): Ratio of S2 velocity to S1 velocity (`velocity_b = velocity_a * velocity_ratio`).
* `drift_sigma` (float, milliradians/second): Standard deviation of steering wheel drift.
* `max_turn_radius` (float, milliradians): Steering angle clamp limit.
* `bias_strength` (float): Attraction factor back to boresight center.
* `seed` (int): Local RNG seed.


---

## 9. Concentric Shell Search (`concentric_shells`)

### Description

A **Center-Heavy** deterministic strategy that performs multiple complete spirals (or rosettes) but resets to the center between each one. 

- **Cycle 1:** Spiral out to $0.2 \times \text{maxsearchradius}$ and back.
- **Cycle 2:** Spiral out to $0.5 \times \text{maxsearchradius}$ and back.
- **Cycle 3:** Spiral out to $1.0 \times \text{maxsearchradius}$ and back.

This ensures the "hot" center is cleared repeatedly while progressively searching deeper into the uncertainty volume.

### Configuration Parameters

* `radii_factors` (array of floats): Ratios of max search radius (e.g. `[0.2, 0.5, 1.0]`).
* `spiral_speed_a` (float): Base spiral speed.
* `speed_ratio` (float): Ratio of S2 to S1 spiral speed (`speed_b = spiral_speed_a * speed_ratio`).

---

## 10. Random Walk (`random_walk`)

### Description

The **Random Walk** strategy is a stochastic search method. At discrete time intervals $T_{step}$, the satellite selects a random direction and moves its boresight by a distance proportional to the beam width, then holds that position. The search is confined within the global `max_search_radius`.

### Configuration Parameters

* `step_duration_a` (float, seconds): Time spent at each stationary point for S1.
* `step_duration_ratio` (float): Ratio of S2 step duration to S1 step duration (`step_duration_b = step_duration_a * step_duration_ratio`).
* `seed` (int): Local RNG seed.


---

## 11. Random Curve (`random_curve`)

### Description

The **Random Curve** strategy is a continuous stochastic search. The boresight moves at a constant angular velocity, but its heading is perturbed by a random "drift". When the boresight reaches the boundary of the global `max_search_radius`, it is reflected back toward the center.

### Configuration Parameters

* `velocity_a` (float, milliradians/second): Base search velocity for S1.
* `velocity_ratio` (float): Ratio of S2 velocity to S1 velocity (`velocity_b = velocity_a * velocity_ratio`).
* `drift_sigma` (float, milliradians/second): Standard deviation of steering drift.
* `max_turn_radius` (float, milliradians): Steering angle clamp limit.
* `seed` (int): Local RNG seed.


---

## 12. Nested Spiral (`nested_spiral`)

### Description

The **Nested Spiral** is a high-confidence, "brute-force" strategy for scenarios with extremely narrow beams. 

- **S1 (Gatekeeper):** Performs an extremely slow spiral. It moves the boresight by one beam width and then stays stationary for a duration $T_{inner}$.
- **S2 (Prober):** Performs a complete, fast spiral search across the entire uncertainty volume during each of S1's "holds".

### Configuration Parameters

* *No configurable TOML parameters.* Both satellites always scan out to `max_search_radius` with S2 probing rapidly while S1 steps by one beam width and holds.

---

## 13. Metadata/Strategy Chain (Runner Logic)

The simulation supports **Strategy Chaining**. If the first strategy fails to achieve a lock, the simulation automatically escalates to the next strategy in the chain.

