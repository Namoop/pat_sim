# Satellite Communication Simulation Model

This document outlines the underlying physical and logical models used in the simulation, as well as the repository layout.

## Model Summary

* **Detection:** A transmission cone hits the target satellite's dish mount; the incoming direction from the transmitter body must fall within the receiver satellite's `dish_fov`.
* **Search:** Independent per-satellite timelines (hold, spiral, reset, etc.) with optional beam/receiver enable states; the FSM snaps and tracking starts upon acquisition.
* **Lock:** Mutual optical lock is achieved when both satellites are transmitting and receiving simultaneously, both bench slews are complete, and there is simultaneous bidirectional visibility (`visible_12 ∧ visible_21`).
* **Partial Acquisition:** If one satellite acquires the other before a strategy times out, the acquiring satellite keeps tracking and ignores later scripted search; the non-acquired satellite continues its strategy chain normally.
* **Replay:** The headless runner and visualizer share the same coupled replay timeline, which is capped at the lock time on successful runs.

---

## Project Layout

The repository is structured as a Python package containing the following modules:

```
src/satellite/
  config.py       — Simulation / scenario / MC loaders
  monte_carlo.py  — error sampling and batch runner
  strategy/
    actions.py    — timeline DSL and StrategyScript
    runner.py     — frame runner
    schedule.py   — compiled timeline for replay
    meta.py       — strategy chain orchestrator
    strategies/   — built-in strategy implementations
  sda/            — bench, transmitter, receiver
  scenario.py     — orchestration and replay
  visualize/      — unified 3D + map visualizer
  mapviz/         — QPainter angular map panels
```
