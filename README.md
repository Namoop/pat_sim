# Satellite Communication — SDA Meta-Strategy Search

Monte Carlo satellite link-establishment simulation. Two satellites search using a **meta-strategy chain** (`minor_offset`, `asymmetric_probe`, `single_miss`, …). Success requires **mutual lock**: both beams enabled, both receivers enabled, both bench slews complete, and simultaneous bidirectional visibility.

The spacecraft body is assumed correctly pointed. Launch mispoint is modeled as **optical-bench rotation**; dish and TX beam share the bench boresight.

---

## Installation

```bash
pip install -e .
pip install -e ".[gpu]"      # optional GPU-accelerated Numba detection
pip install -e ".[viz]"      # interactive 3D + angular map (PyVista, PyQt6)
pip install -e ".[opt]"      # optional Optuna strategy parameter optimizer
pip install -e ".[all]"      # install everything (viz, gpu, opt, dev)
```

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
