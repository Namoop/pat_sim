# Comprehensive Satellite Search Strategies

This document provides a detailed overview of the search strategies available in the `satellite` simulation suite. Each strategy is designed to establish a mutual optical communication link between two satellites (S1 and S2) under varying levels of initial pointing uncertainty.

---

## Global Search Constraints

Several strategies reference a global bounding volume for the search. This is typically configured in `Simulation.toml`.


| Parameter           | Default  | Description                                                                                                   |
| ------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `max_search_radius` | 0.07 rad | The hard angular limit for any search pattern. Boresights will not exceed this radius from the initial guess. |


---

## 1. Minor Offset (`minor_offset`)

### Description

The **Minor Offset** strategy is the simplest search pattern, specifically intended for use when the search radius matches the receiver's **FOV**. 

This strategy assumes that the pointing errors for both satellites are less than the FOV. In this scenario, S1 performs an Archimedean spiral search while S2 remains stationary at its initial "best guess" boresight. Because both satellites share the same FOV, having S1 spiral out to a radius matching that FOV **guarantees** that its beam will cross S2's receiver FOV.

**Dynamic Duration:** The time spent searching is automatically calculated based on the receiver's FOV and spiral speed: $T = FOV / (w \cdot \text{speed})$.

### Configuration Parameters


| Parameter      | Description                                       |
| -------------- | ------------------------------------------------- |
| `spiral_speed` | Rate of angular expansion/rotation multiplier.    |
| `w`            | Archimedean spiral radial coefficient.            |
| `k`            | Archimedean spiral angular frequency coefficient. |


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


| Parameter       | Description                                                                |
| --------------- | -------------------------------------------------------------------------- |
| `spiral_radius` | Maximum radius for the search spiral (radians). Shared by both satellites. |
| `spiral_speed`  | Expansion rate multiplier.                                                 |
| `w`, `k`        | Spiral geometry coefficients.                                              |


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


| Parameter       | Description                                                               |
| --------------- | ------------------------------------------------------------------------- |
| `spiral_radius` | Maximum radius for the probe spiral (radians). Shared by both satellites. |
| `lock_duration` | Time spent attempting a simultaneous reciprocal lock at the end.          |
| `spiral_speed`  | Expansion rate multiplier.                                                |
| `w`, `k`        | Spiral geometry coefficients.                                             |


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

Both satellites spiral from the center $(0,0)$ out to the global `max_search_radius`, and then immediately spiral back from the maximum radius to the center. This "in-and-out" motion ensures that the search covers the central high-probability region twice per cycle.

### Configuration Parameters


| Parameter     | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| `speed_a`     | Expansion speed for S1.                                           |
| `speed_ratio` | Ratio of speeds (e.g., 1.414). `speed_b = speed_a * speed_ratio`. |
| `w`, `k`      | Spiral geometry coefficients.                                     |


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
spiral(radius=max_search_radius, speed=speed_a)
spiral(radius=0, speed=speed_a)
```

**Satellite S2:**

```python
beam.enable(); receiver.enable()
spiral(radius=max_search_radius, speed=speed_a * speed_ratio)
spiral(radius=0, speed=speed_a * speed_ratio)
```

---

## 5. Dual Orthogonal Raster (`dual_raster`)

### Description

The **Dual Raster** strategy utilizes orthogonal scanning patterns to ensure coverage. Satellite S1 scans the circular search area using horizontal rows (rastering in X, stepping in Y), while S2 scans using vertical columns (rastering in Y, stepping in X). Both satellites cover the circular uncertainty volume defined by the global `max_search_radius`.

This strategy uses a **Boustrophedon (Serpentine)** motion: at the end of each line, the beam steps to the next elevation/offset and immediately scans in the opposite direction. This eliminates "flyback" resets and minimizes wasted movement time.

### Configuration Parameters


| Parameter | Description                        |
| --------- | ---------------------------------- |
| `steps_a` | Number of horizontal lines for S1. |
| `steps_b` | Number of vertical lines for S2.   |
| `speed`   | Base scan velocity multiplier.     |


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

By selecting a frequency ratio $\omega_x / \omega_y$ that is near-irrational or a large-denominator rational, the curve will fill the rectangular bounding box of the `max_search_radius` without repeating itself for a long duration.

### Configuration Parameters


| Parameter  | Description                                   |
| ---------- | --------------------------------------------- |
| `wx`, `wy` | Angular frequencies for X and Y axes (rad/s). |
| `delta`    | Phase shift (radians).                        |
| `duration` | Total time to perform the scan.               |


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
lissajous(A=max_search_radius, wx=wx, wy=wy, delta=delta, duration=duration)
```

---

## 7. Rosette Scan (`rosette_scan`)

### Description

The **Rosette Scan** produces a flower-like pattern that provides extremely high density at the center of the search area while still covering the peripheral field of regard. The pattern is defined by the superposition of two counter-rotating circular motions.

The boresight position $(x, y)$ is given by:
$x(t) = \text{maxsearchradius} \cdot \cos(\omega_1 t)$
$y(t) = \text{maxsearchradius} \cdot \cos(\omega_2 t)$

If the ratio $\omega_1 / \omega_2$ is irrational, the pattern will eventually fill the entire circular area bounded by the global `max_search_radius`.

### Configuration Parameters


| Parameter        | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `s1_w1`, `s1_w2` | Angular frequencies for S1 (rad/s).                                |
| `s2_w1`, `s2_w2` | Angular frequencies for S2 (rad/s). Set to 0 to remain stationary. |
| `duration`       | Total time to perform the scan.                                    |


### Action Script (Pseudocode)

**Satellite S1:**

```python
beam.enable(); receiver.enable()
rosette(A=max_search_radius, w1=s1_w1, w2=s1_w2, duration=duration)
```

**Satellite S2:**

```python
beam.enable(); receiver.enable()
rosette(A=max_search_radius, w1=s2_w1, w2=s2_w2, duration=duration)
```

---

## 8. Stochastic Center Re-bias (`center_rebias`)

### Description

This is a **Center-Heavy** stochastic strategy. It performs a Random Walk, but with a "spring-like" force pulling the boresight back toward the center $(0,0)$. This ensures that the high-probability central region is sampled much more frequently than the edges, which is ideal for Gaussian pointing errors.

### Configuration Parameters


| Parameter        | Description                               |
| ---------------- | ----------------------------------------- |
| `step_duration`  | Time spent at each point (seconds).       |
| `bias_strength`  | Pull strength toward center (0.0 to 1.0). |
| `total_duration` | Total simulation time.                    |


---

## 9. Concentric Shell Search (`concentric_shells`)

### Description

A **Center-Heavy** deterministic strategy that performs multiple complete spirals (or rosettes) but resets to the center between each one. 

- **Cycle 1:** Spiral out to $0.2 \times \text{maxsearchradius}$ and back.
- **Cycle 2:** Spiral out to $0.5 \times \text{maxsearchradius}$ and back.
- **Cycle 3:** Spiral out to $1.0 \times \text{maxsearchradius}$ and back.

This ensures the "hot" center is cleared repeatedly while progressively searching deeper into the uncertainty volume.

---

## 10. Random Walk (`random_walk`)

### Description

The **Random Walk** strategy is a stochastic search method. At discrete time intervals $T_{step}$, the satellite selects a random direction and moves its boresight by a distance proportional to the beam width, then holds that position. The search is confined within the global `max_search_radius`.

### Configuration Parameters


| Parameter        | Description                                    |
| ---------------- | ---------------------------------------------- |
| `step_duration`  | Time spent at each stationary point (seconds). |
| `total_duration` | Total simulation time for this strategy.       |


---

## 11. Random Curve (`random_curve`)

### Description

The **Random Curve** strategy is a continuous stochastic search. The boresight moves at a constant angular velocity, but its heading is perturbed by a random "drift". When the boresight reaches the boundary of the global `max_search_radius`, it is reflected back toward the center.

### Configuration Parameters


| Parameter     | Description                                               |
| ------------- | --------------------------------------------------------- |
| `velocity`    | Angular velocity of the boresight (rad/s).                |
| `drift_sigma` | Standard deviation of the random heading change per step. |


---

## 12. Nested Spiral (`nested_spiral`)

### Description

The **Nested Spiral** is a high-confidence, "brute-force" strategy for scenarios with extremely narrow beams. 

- **S1 (Gatekeeper):** Performs an extremely slow spiral. It moves the boresight by one beam width and then stays stationary for a duration $T_{inner}$.
- **S2 (Prober):** Performs a complete, fast spiral search across the entire uncertainty volume during each of S1's "holds".

### Configuration Parameters


| Parameter      | Description                  |
| -------------- | ---------------------------- |
| `outer_radius` | Radius for S1's slow search. |
| `inner_radius` | Radius for S2's fast search. |


---

## 13. Golden Angle Spiral (`golden_angle_spiral`)

### Description

The **Golden Angle Spiral** (Vogel's Spiral) is a discrete search pattern that provides a nearly optimal uniform distribution of points within a circle. 

The point positions are defined by:
$r_n = c \sqrt{n}$
$\theta_n = n \cdot 137.508^\circ$

---

## 14. Metadata/Strategy Chain (Runner Logic)

The simulation supports **Strategy Chaining**. If the first strategy fails to achieve a lock, the simulation automatically escalates to the next strategy in the chain.

---

## TODOs

- Update `src/satellite/config.py`:
  - Remove `duration` and `reset_duration` fields from strategy TOML loading.
  - Add `max_beam_speed` and `max_fsm_speed` to `SharedSatelliteConfig` ([satellite] section).
  - Add `max_search_radius` to `SimulationConfig`.
- Implement `StrategyConfig.spiral_duration()` helper for $T = R / (w \cdot \text{speed})$.
- Implement `StrategyConfig.reset_duration()` helper for $T_{reset} = R / \text{maxbeamspeed}$.
- Refactor `MinorOffsetStrategy`, `SingleMissStrategy`, and `AsymmetricProbeStrategy` to use strictly calculated timings.
- Implement the actual `asymmetric_swap` logic in a new strategy class.
- Implement new strategies:
  - `dual_spiral`
  - `dual_raster` (serpentine)
  - `lissajous_scan`
  - `rosette_scan`
  - `center_rebias`
  - `concentric_shells`
  - `random_walk`
  - `random_curve`
  - `nested_spiral`
  - `golden_angle_spiral`

