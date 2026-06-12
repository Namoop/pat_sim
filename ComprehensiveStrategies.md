# Comprehensive Satellite Search Strategies

This document provides a detailed overview of the search strategies available in the `satellite` simulation suite. Each strategy is designed to establish a mutual optical communication link between two satellites (S1 and S2) under varying levels of initial pointing uncertainty.

---

## 1. Minor Offset (`minor_offset`)

### Description
The **Minor Offset** strategy is the simplest search pattern, specifically intended for use when `max_spiral_radius` is set to match the receiver's **FOV**. 

This strategy assumes that the pointing errors for both satellites are less than the FOV. In this scenario, S1 performs an Archimedean spiral search while S2 remains stationary at its initial "best guess" boresight. Because both satellites share the same FOV, having S1 spiral out to a radius matching that FOV **guarantees** that its beam will cross S2's receiver FOV, as S1's search area covers the entire potential misalignment volume of the stationary S2.

### Configuration Parameters
| Parameter | Description |
| :--- | :--- |
| `duration` | Total time allowed for the spiral search (seconds). |
| `max_spiral_radius` | Maximum angular radius of the spiral (radians). Should be set to `"fov"` to ensure the mathematical detection guarantee. |
| `spiral_speed` | Rate of angular expansion/rotation multiplier. |
| `w` | Archimedean spiral radial coefficient. |
| `k` | Archimedean spiral angular frequency coefficient. |

### Action Script (Pseudocode)

**Satellite S1:**
```python
beam.enable()
receiver.enable()
spiral(duration=duration, radius=max_spiral_radius, speed=spiral_speed)
```

**Satellite S2:**
```python
beam.enable()
receiver.enable()
hold(duration=duration)
```

---

## 2. Single Miss (`single_miss`)

### Description
The **Single Miss** strategy handles larger offsets where a simple FOV search might fail. It uses a two-phase alternating approach. 

- **Phase 1:** S1 performs a wide spiral search while S2 holds position. 
- **Reset:** Both satellites reset their optical benches to the initial boresight.
- **Phase 2:** S2 performs a wide spiral search while S1 holds position.

This ensures that even if the "blind" satellite was pointing in the wrong direction during Phase 1, the roles reverse to cover the uncertainty volume from the other perspective.

### Configuration Parameters
| Parameter | Description |
| :--- | :--- |
| `phase1_duration` | Duration of the first spiral phase (seconds). |
| `a_spiral_radius` | Maximum radius for S1's spiral in Phase 1 (radians). |
| `reset_duration` | Time allocated for the bench to slew back to center between phases. |
| `phase2_duration` | Duration of the second spiral phase (seconds). |
| `b_spiral_radius` | Maximum radius for S2's spiral in Phase 2 (radians). |
| `spiral_speed` | Expansion rate multiplier. |
| `w`, `k` | Spiral geometry coefficients. |

### Action Script (Pseudocode)

**Satellite S1:**
```python
# Phase 1
beam.enable(); receiver.enable()
spiral(duration=phase1_duration, radius=a_spiral_radius)

# Reset & Phase 2
reset(duration=reset_duration)
hold(duration=phase2_duration)
```

**Satellite S2:**
```python
# Phase 1
beam.enable(); receiver.enable()
hold(duration=phase1_duration)

# Reset & Phase 2
hold(duration=reset_duration)
spiral(duration=phase2_duration, radius=b_spiral_radius)
```

---

## 3. Asymmetric Swap (`asymmetric_swap`)

### Description
The **Asymmetric Swap** strategy is an evolution of the asymmetric probe, designed to mitigate "mutual blindness" and minimize internal interference (glint). It designates initial "Leader" and "Follower" roles, but swaps them midway through the search.

- **Phase 1 (S1 Probes):** S1 performs a spiral search with its beam enabled and receiver disabled. S2 keeps its beam disabled and listens with its receiver.
- **Role Swap:** Both satellites reset their optical benches.
- **Phase 2 (S2 Probes):** S2 performs a spiral search with its beam enabled and receiver disabled. S1 keeps its beam disabled and listens with its receiver.
- **Reciprocal Lock Phase:** Finally, both satellites enable both hardware subsystems and hold their center positions to verify a mutual lock.

This strategy is highly effective for terminals where the local transmitter might overpower the local receiver, and it ensures that detection is tested in both directions ($S1 \rightarrow S2$ and $S2 \rightarrow S1$) under ideal conditions before requiring a simultaneous link.

### Configuration Parameters
| Parameter | Description |
| :--- | :--- |
| `phase1_duration` | Duration S1 spends probing (seconds). |
| `a_spiral_radius` | Maximum radius for S1's probe spiral (radians). |
| `reset_duration` | Time allocated for bench resets between phases. |
| `phase2_duration` | Duration S2 spends probing (seconds). |
| `b_spiral_radius` | Maximum radius for S2's probe spiral (radians). |
| `lock_duration` | Time spent attempting a simultaneous reciprocal lock at the end. |
| `spiral_speed` | Expansion rate multiplier. |
| `w`, `k` | Spiral geometry coefficients. |

### Action Script (Pseudocode)

**Satellite S1 (Initial Leader):**
```python
# Phase 1: Probing
beam.enable(); receiver.disable()
spiral(duration=phase1_duration, radius=a_spiral_radius)
reset(duration=reset_duration)

# Phase 2: Listening (Swap)
beam.disable(); receiver.enable()
hold(duration=phase2_duration)
hold(duration=reset_duration)

# Phase 3: Reciprocal Lock
beam.enable(); receiver.enable()
hold(duration=lock_duration)
```

**Satellite S2 (Initial Follower):**
```python
# Phase 1: Listening
beam.disable(); receiver.enable()
hold(duration=phase1_duration)
hold(duration=reset_duration)

# Phase 2: Probing (Swap)
beam.enable(); receiver.disable()
spiral(duration=phase2_duration, radius=b_spiral_radius)
reset(duration=reset_duration)

# Phase 3: Reciprocal Lock
beam.enable(); receiver.enable()
hold(duration=lock_duration)
```

---

## 4. Bi-Directional Raster (`raster_scan`)

### Description
The **Bi-Directional Raster** strategy replaces the spiral search with a traditional rectangular grid scan. This is used in high-uncertainty scenarios where systematic, linear coverage of a square angular volume is preferred over the radially-weighted coverage of a spiral.

In this pattern, S1 performs a serpentine scan (left-to-right, then right-to-left at a different elevation) while S2 remains stationary. This is often the precursor to the "Comprehensive" search meta-strategy.

### Configuration Parameters
| Parameter | Description |
| :--- | :--- |
| `duration` | Total time for the full raster scan. |
| `fov_width` | Horizontal width of the scan area (radians). |
| `fov_height` | Vertical height of the scan area (radians). |
| `steps` | Number of scan lines in the vertical direction. |
| `spiral_speed` | Velocity multiplier for the scan head. |

### Action Script (Pseudocode)

**Satellite S1:**
```python
beam.enable()
receiver.enable()
raster(duration=duration, width=fov_width, height=fov_height, steps=steps)
```

**Satellite S2:**
```python
beam.enable()
receiver.enable()
hold(duration=duration)
```

---

## 5. Comprehensive Search (`comprehensive`)

### Description
The **Comprehensive Search** is a meta-strategy or placeholder for high-uncertainty scenarios. It is currently implemented as a stub that performs a minimal `hold` and fails, intended to be expanded into a full raster scan or interleaved multi-spiral pattern that covers the entire 3-sigma uncertainty sphere of both satellites simultaneously.

### Configuration Parameters
*(Currently utilizes standard simulation/t_step defaults)*

### Action Script (Pseudocode)

**Satellite S1 & S2:**
```python
hold(duration=t_step)
```

---

## 5. Metadata/Strategy Chain (Runner Logic)

While not a "search pattern" in itself, the simulation supports **Strategy Chaining**. This allows the user to define a sequence (e.g., `["minor_offset", "single_miss"]`). If the first strategy fails to achieve a lock within its allocated duration, the simulation automatically "escalates" to the next strategy in the chain, resetting the satellites' internal search states while preserving any successful acquisitions.

### Configuration (`MonteCarlo.toml`)
```toml
[strategy]
k = 10.0
chain = ["minor_offset", "single_miss"]
```
