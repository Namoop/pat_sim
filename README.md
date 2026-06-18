# Satellite Communication Search

Monte Carlo satellite link-establishment simulation. Two satellites search using a **meta-strategy chain** (`minor_offset`, `asymmetric_swap`, `single_miss`, …). Success requires **mutual lock**: both beams enabled, both receivers enabled, both bench slews complete, and simultaneous bidirectional visibility.

The spacecraft body is assumed correctly pointed. Launch mispoint is modeled as **optical-bench rotation**; dish and TX beam share the bench boresight.

---

## System Architecture & Model

### 1. Physical Setup

The simulation models two satellites: **S1** (placed at the origin) and **S2** (placed on the positive x-axis at `distance`).

- **Optical Bench (Beam Director):** Each satellite has a coarse-pointing optical bench. It has a wide range of motion but slews slowly, limited by `max_beam_speed`. Initial pointing errors (launch mispoints) are modeled as angular theta/phi offsets of this bench.
- **Fast Steering Mirror (FSM):** Mounted on the optical bench is a fine-pointing FSM. The FSM is extremely fast (`max_fsm_speed`) and precise, but has a narrow angular range of motion (`max_fsm_radius`).
- **Transmitter & Receiver:** The transmitter narrow beam (`beam_width`) and the receiver dish (`dish_fov`) share the bench boresight and are steered dynamically by the FSM.

### 2. General Strategy Concept

Because of the initial pointing offsets, the satellites' default boresights do not align. To establish a link, they execute coordinated search strategies (e.g., spirals, rasters, Lissajous scans) to sweep their beams and receiver fields of view across the uncertainty region.
A strategy chain orchestrates this search, escalating to wider or more robust strategies if a lock is not established within a given strategy's timeout.

### 3. FSM Snapping and Offset Bleeding

When a satellite (e.g., S2) detects an incoming signal from S1 (meaning S1's beam hits S2 and falls within S2's receiver FOV):

1. **FSM Snap (Acquisition):** S2's FSM deflects ("snaps") immediately to point S2's receiver and transmitter beam directly at S1, establishing an immediate, temporary lock.
2. **Offset Bleeding:** While the FSM holds the connection, the slow optical bench begins slewing toward S1. As the bench rotates closer to S1's true position, the required FSM deflection angle decreases. This offset is "bled" off the FSM and back to the beam director, keeping the FSM centered so it has its full range of motion available to track high-frequency movement.

### 4. Lock Condition

The ultimate goal is a stable **mutual lock**. A mutual lock is declared when:

- Both satellites are simultaneously transmitting to and receiving from one another.
- Both satellites have completed their bench slews (the pointing offset is fully bled, and the optical benches are aligned).
- There is bidirectional line-of-sight visibility.

---

## Project Layout

The repository is structured as a Python package containing the following modules:

```
src/
  config.py         — TOML helpers, path resolution, override merging
  satellite/
    config.py       — Environment.toml loader
    detection.py    — beam–dish hit tests
    physics/        — bench, transmitter, receiver, FSM
    math/           — vector math and geometric constructions
  strategy/
    config.py       — [strategy] section parser
    actions.py      — timeline DSL and StrategyScript
    runner.py       — frame runner
    schedule.py     — compiled timeline for replay
    meta.py         — strategy chain orchestrator
    strategies/     — built-in strategy implementations
  visualize/        — unified 3D + eye + mag visualizer
    frames.py       — tangent-plane projection (shared by eye and mag)
    scene.py        — eye scene builder (shared by eye and mag)
    panels/
      panel_3d.py   — PyVista 3D view
      eye_panel.py
      mag_panel.py
  scenario/         — single-scenario CLI and runner (config, types, run, replay, diagnostics)
  montecarlo/       — Monte Carlo batch CLI and runner (config, types, run, cuda)
  optimize/
    config.py       — [optimize] section parser
    __main__.py     — parameter optimizer entrypoint
    run_all.sh      — batch script to run optimization
```

---

## Installation

```bash
pip install -e .
pip install -e ".[gpu]"      # optional GPU-acceleration with Numba and CUDA
pip install -e ".[viz]"      # interactive 3D + eye map (PyVista, PyQt6)
pip install -e ".[opt]"      # optional Optuna strategy parameter optimizer
pip install -e ".[all]"      # install everything (viz, gpu, opt, dev)
```

> Installing the package in editable mode (`pip install -e ...`) is highly recommended, as it automatically registers the packages in your environment's Python path. If you run the scripts without an editable installation, you will need to prefix commands with `PYTHONPATH=src`.

---

## Quickstart

```bash
# Run the default scenario
python -m scenario

# Visualize with specific starting time and eye tab initially active
python -m scenario --visualize eye --t 2.5

# Run Monte Carlo simulation batch (defaults to config/MonteCarlo.toml)
python -m montecarlo
```

---

## Documentation Directory

For details on configuration, strategy details, CLI reference, and architecture/modelling:

- **[docs/environment.md](docs/environment.md)**: Configuration reference for physical constraints and simulator parameters.
- **[docs/scenario.md](docs/scenario.md)**: Configuration reference for Scenario run offsets, overrides, and strategy stubs.
- **[docs/monte_carlo.md](docs/monte_carlo.md)**: Configuration reference for Monte Carlo batch runs, random seeds, and error distributions.
- **[docs/satellite.md](docs/satellite.md)**: Project model summary, package structure, CLI options reference, troubleshooting, and testing details.
- **[docs/strategies.md](docs/strategies.md)**: Descriptions and mathematical behaviors of built-in pointing search strategies.
- **[docs/optimize.md](docs/optimize.md)**: Strategy parameter optimizer — methodology, CLI options, custom configuration TOML, and physical speed validation.

---

## Agentic Testing (For AI Assistants / Agents)

If you are an agentic coding assistant working on this codebase, follow these instructions to run tests and scripts successfully in sandbox or headless environments:

1. **Bypass the Bootstrap Wrapper:**
  The project uses an auto-bootstrap mechanism in `src/satellite/__init__.py` to link CUDA dependencies. In sandbox, containerized, or headless environments lacking a GPU, this wrapper triggers process re-execution and causes test commands to fail or hang.
   **Always set `SATELLITE_NO_GPU=1`** when running tests.
2. **Set the correct PYTHONPATH:**
  To locate source packages and test configuration packages (`tests.conftest`) simultaneously, the python search path must include both the `src/` directory and the root directory `.`.
   **Always set `PYTHONPATH=src:.`**.
3. **Running the Test Suite:**
  Execute tests within the virtual environment using:
  ```bash
  PYTHONPATH=src:. SATELLITE_NO_GPU=1 .venv/bin/pytest
  ```
