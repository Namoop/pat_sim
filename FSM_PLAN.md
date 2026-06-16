# FSM Implementation & Visualization Plan

This plan describes the implementation of Fast Steering Mirror (FSM) physical constraints and coupling to both transmit (TX) and receive (RX) systems, along with the visualization of coarse gimbal pointing (the "beam director") on the map view.

---

## 1. Physical Model & Design

In the physical system:
1. **Coarse Gimbal (Beam Director):** The optical bench itself, which has slow movement speed (`max_beam_speed` $\approx$ 87 mrad/s).
2. **Fine Steering (FSM):** Fast actuators on the optical bench that steer the optical path with high speed (`max_fsm_speed` $\approx$ 1000 mrad/s) but limited range (`max_fsm_radius` $\approx$ 1.0 mrad).

### Acquisition Dynamics
* **Initial Detection:** When Satellite A sees the beam from Satellite B, it activates its FSM. The FSM instantly (within a single simulation step, since $t_{\text{step}} = 10\text{ ms}$) shifts up to `max_fsm_radius` in the direction of Satellite B.
* **Beam/FOV Coupling:** Both the receiver FOV (camera) and the transmit beam share the FSM path, meaning **both** are deflected by the FSM offset. This allows Satellite A's beam to instantly point back at Satellite B, prompting Satellite B to acquire as well.
* **Coarse Slew & Offset Bleeding:** The slow beam director (optical bench) slews towards the target. As the bench boresight moves closer to the target, the required FSM offset decreases (bleeds off) until the bench boresight is aligned with the target, the FSM returns to center (offset = 0), and the FSM is centered again.

### Visualization Behavior
* When the FSM is centered (offset $\approx 0$), only the beam/FOV circles and the partner dot are displayed.
* When the FSM uncenters (offset $> 0$), a **gray dot** appears on the map representing the coarse beam director's boresight.
* The beam and camera FOV circles follow the FSM, so they lock onto the partner target almost instantly.
* The gray dot (coarse pointing) slews toward the center of the circles (the partner target) until it arrives and disappears (as the FSM offset bleeds back to zero).

---

## 2. Configuration Changes

We need to add a `max_fsm_radius` configuration parameter.

### Files to Modify:
* **`Simulation.toml`**: Add `max_fsm_radius` under `[satellite]`:
  ```toml
  [satellite]
  # ... existing config ...
  max_fsm_radius = 1.0 # milliradians
  ```
* **`src/satellite/config.py`**:
  * Add `max_fsm_radius` to the `SharedSatelliteConfig` dataclass:
    ```python
    max_fsm_radius: float = 1.0e-3  # Default to 1.0 milliradian (in radians)
    ```
  * In `_load_shared_satellite(data: dict)`, parse the parameter and convert it from milliradians to radians:
    ```python
    max_fsm_radius=float(data.get("max_fsm_radius", 1.0)) * 1e-3,
    ```

---

## 3. Core Physics & Logic Changes (CPU)

We need to enforce `max_fsm_radius` limits during the FSM update and ensure both the receiver and transmitter use the deflected effective boresight.

### Files to Modify:

#### A. `src/satellite/sda/fsm.py`
* **Update `FastSteeringMirror.update`** to accept `max_fsm_radius` and clamp the target offsets:
  ```python
  def update(self, bench_boresight: Vec3, dq: float, max_fsm_speed: float, max_fsm_radius: float) -> None:
      """Slew mirror toward target at max_fsm_speed, clamped to max_fsm_radius."""
      if self.track_target is None:
          return

      # Always recompute target offsets as bench moves
      self._target_theta, self._target_phi = _offsets_to_target(
          bench_boresight,
          self.track_target,
      )

      # Clamp target to the physical range of the FSM
      target_dist = np.hypot(self._target_theta, self._target_phi)
      clamped_target_theta = self._target_theta
      clamped_target_phi = self._target_phi
      if target_dist > max_fsm_radius:
          clamped_target_theta = (self._target_theta / target_dist) * max_fsm_radius
          clamped_target_phi = (self._target_phi / target_dist) * max_fsm_radius

      if max_fsm_speed <= 0:
          # Infinite speed fallback
          self.theta_offset = clamped_target_theta
          self.phi_offset = clamped_target_phi
          self.locked = (target_dist <= max_fsm_radius + 1e-12)
          return

      # Slew toward the clamped target
      du = clamped_target_theta - self.theta_offset
      dv = clamped_target_phi - self.phi_offset
      dist = np.hypot(du, dv)
      max_step = max_fsm_speed * dq

      if dist <= max_step + 1e-12:
          self.theta_offset = clamped_target_theta
          self.phi_offset = clamped_target_phi
          # Locked means we are pointing exactly at the actual target
          self.locked = (target_dist <= max_fsm_radius + 1e-12)
      else:
          self.theta_offset += (du / dist) * max_step
          self.phi_offset += (dv / dist) * max_step
          self.locked = False
  ```
* **Update `effective_receive_boresight`** to apply the offset at all times, not just when `self.locked` is True:
  ```python
  def effective_receive_boresight(self, bench_boresight: Vec3) -> Vec3:
      """Receive/transmit aim with FSM steering applied on top of bench boresight."""
      u_x, u_y, _ = transmitter_basis(
          *spherical_angles_from_direction(bench_boresight)
      )
      return direction_with_tangent_offset(
          bench_boresight,
          u_x,
          u_y,
          self.theta_offset,
          self.phi_offset,
      )
  ```

#### B. `src/satellite/sda/bench.py`
* **Pass the FSM radius limit** inside `observe_beam`:
  ```python
  # Update FSM slew every step
  fsm.update(self.bench_boresight, dq, self.max_fsm_speed, self.max_fsm_radius)
  ```
  *(Note: We will need to store `max_fsm_radius` as an instance field in `OpticalBench` upon build.)*

#### C. `src/satellite/strategy/runner.py`
* **Update `_hold_or_track`** to return the **effective transmit boresight** instead of the coarse bench boresight when tracking:
  ```python
  def _hold_or_track(self, sat, partner_position, t_step: float, events: list[str]):
      if sat.receiver.has_seen_beam:
          _, acq_events = sat.receiver.observe_beam(
              False,
              partner_position,
              t_step,
          )
          for event in acq_events:
              if event == "Slew complete":
                  events.append(f"{sat.name} slew complete")
          # Return FSM deflected beam direction
          return sat.receiver.fsm.effective_receive_boresight(sat.bench.bench_boresight)
      return sat.bench.bench_boresight
  ```
* **Update final lock evaluation checks** in `_evaluate_lock_fast` to also pass the FSM-deflected beam axis when evaluating `link_established`:
  ```python
  # Lines ~248 & ~262: Instead of direct bench_boresight, apply FSM offsets:
  effective_aim1 = self.ctx.s1.receiver.fsm.effective_receive_boresight(
      self.ctx.s1.bench.bench_boresight
  )
  # ... same for s2
  ```

#### D. `src/satellite/strategy/base.py`
* **Update `link_established`** to use the receiver's effective boresight (including FSM offset) for the FOV check:
  ```python
  def link_established(
      tx_sat: Satellite,
      rx_sat: Satellite,
      beam_axis: Vec3,
      config: ScenarioConfig,
  ) -> bool:
      coarse_boresight = rx_sat.bench.dish_boresight_inertial()
      mount = rx_sat.bench.dish_mount_for_boresight(
          coarse_boresight, rx_sat.receiver.body_radius
      )
      # Apply receive FSM offset to the receiver FOV boresight
      effective_rx_boresight = rx_sat.receiver.fsm.effective_receive_boresight(
          coarse_boresight
      )

      return beam_hits_dish_fast(
          tx_sat.position,
          mount,
          effective_rx_boresight,
          rx_sat.receiver.cos_dish_fov,
          beam_axis,
          tx_sat.cos_alpha,
          tx_sat.beam_length,
      )
  ```

---

## 4. CUDA / GPU Simulation Changes

To ensure the GPU-based Monte Carlo matches the CPU results exactly, we must adapt the CUDA kernel.

### Files to Modify:

#### `src/satellite/cuda_monte_carlo.py`
* **Extend parameter unpacking:**
  * Add `max_fsm_radius` as the 10th parameter in the `sim_params` array.
* **Update FSM updates in CUDA:**
  * Apply `max_fsm_radius` clamping inside `simulate_batch_kernel` whenever FSM angles are updated:
    * During initial detection (`s1_just_detected`).
    * During the finite FSM slew update step.
    * During the coarse bench slew update step (where FSM targets are recomputed).
* **Apply FSM offset to both TX and RX:**
  * Modify `s1_tx_aim` and `s2_tx_aim` to include the FSM offset `(fsm_theta, fsm_phi)` as long as acquisition has started (`s1_has_seen`/`s2_has_seen` is True).
  * Do the same for the receiver's boresight `rx_aim_temp`.

---

## 5. Map Visualization Changes

Now we render the FSM-deflected circles and coarse pointing gray dot in the PyQt map.

### Files to Modify:

#### A. `src/satellite/mapviz/scene.py`
* **Extend `MapPanel` dataclass** to store the beam director (coarse) coordinates if the FSM is uncentered:
  ```python
  @dataclass(frozen=True)
  class MapPanel:
      # ... existing fields ...
      beam_director: tuple[float, float] | None = None  # (theta, phi) of bench boresight
  ```
* **Update `build_panel`**:
  * Point the beam and camera circles to the **FSM effective receive boresight** instead of `bench_boresight`:
    ```python
    fsm = sat.receiver.fsm
    # Center circles around the FSM deflected direction
    effective_aim = fsm.effective_receive_boresight(sat.bench.bench_boresight)
    center_theta, center_phi = direction_to_tangent_angles(origin, effective_aim)
    ```
  * Populate `beam_director` with coarse coordinates only if the FSM is off-center:
    ```python
    if not fsm.is_neutral():
        bd_theta, bd_phi = direction_to_tangent_angles(origin, sat.bench.bench_boresight)
        beam_director = (bd_theta, bd_phi)
    else:
        beam_director = None
    ```

#### B. `src/satellite/mapviz/panel_widget.py`
* **Update `AngularMapPanel.paintEvent`** to paint the gray dot representing the coarse pointing:
  ```python
  if panel.beam_director is not None:
      bd_pt = self._to_pixel(plot, panel.beam_director[0], panel.beam_director[1])
      painter.setPen(Qt.PenStyle.NoPen)
      painter.setBrush(QBrush(QColor(128, 128, 128, 200)))  # Semi-transparent gray
      painter.drawEllipse(bd_pt, 4.0, 4.0)
  ```

> [!NOTE]
> **3D Visualization (pyvista):** Similar updates will eventually be needed for the 3D visualization (e.g. updating beam cones/FOV meshes to align with the FSM effective boresights, and visualizing the coarse beam director). However, this work is **deferred** until the core physics, logic, and map visualizations are fully working.

---

## 6. Validation & Verification Plan

1. **Visual Walkthrough Verification:**
   * Run the visualizer (`python -m satellite` or corresponding command).
   * Play the timeline for a scenario where one satellite acquires the other.
   * Observe the map view:
     * When acquisition starts, the beam/FOV circles should immediately snap close to the partner satellite.
     * A gray dot should appear at the old center and slew smoothly towards the center of the circles (representing the coarse gimbal catching up).
     * Once it reaches the center, the gray dot should disappear.
2. **Mathematical Correctness Check:**
   * Ensure that the Monte Carlo CPU and GPU runs match.
   * Verify that clamping works correctly when initial misalignment exceeds `max_fsm_radius` (the FSM should saturate at the edge of the circle and wait for the gimbal to bring the target closer).
