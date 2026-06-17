# Satellite Communication — SDA Meta-Strategy Search

Monte Carlo satellite link-establishment simulation. Two satellites search using a **meta-strategy chain** (`minor_offset`, `asymmetric_probe`, `single_miss`, …). Success requires **mutual lock**: both beams enabled, both receivers enabled, both bench slews complete, and simultaneous bidirectional visibility.

The spacecraft body is assumed correctly pointed. Launch mispoint is modeled as **optical-bench rotation**; dish and TX beam share the bench boresight.

---

## Installation

```bash
pip install -e .
pip install -e ".[gpu]"      # optional GPU-acceleration with Numba and CUDA
pip install -e ".[viz]"      # interactive 3D + angular map (PyVista, PyQt6)
pip install -e ".[opt]"      # optional Optuna strategy parameter optimizer
pip install -e ".[all]"      # install everything (viz, gpu, opt, dev)
```

> [!NOTE]
> Installing the package in editable mode (`pip install -e ...`) is highly recommended, as it automatically registers the packages in your environment's Python path. If you run the scripts without an editable installation, you will need to prefix commands with `PYTHONPATH=src`.

---

## Quickstart

```bash
# Run the default scenario
python -m satellite

# Run Monte Carlo simulation batch
python -m satellite --monte-carlo config/MonteCarlo.toml

# Visualize with specific starting time and map tab initially active
python -m satellite --visualize map --t 2.5
```

---

## Documentation Directory

For details on configuration, strategy details, CLI reference, and architecture/modelling:

- **[docs/config.md](docs/config.md)**: Configuration reference for Simulation, Scenarios, and Monte Carlo.
- **[docs/strategies.md](docs/strategies.md)**: Descriptions of built-in pointing search strategies.
- **[docs/cli.md](docs/cli.md)**: Detailed command-line reference, benchmarking, and troubleshooting.
- **[docs/model.md](docs/model.md)**: Physics model description, FSM acquisition logic, and package layout.
- **[docs/optimize.md](docs/optimize.md)**: Strategy parameter optimizer — methodology, CLI options, and physical speed validation.

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

