# Monte Carlo and Error Configuration (`MonteCarlo.toml`)

This document describes the parameters defined in `MonteCarlo.toml` (e.g., `config/MonteCarlo.toml`), which configures batch trials over randomized pointing error distributions.

The configuration file is loaded by [load_monte_carlo_config](file:///home/theodore/Documents/satellite/src/satellite/config.py). All angular values are defined in **milliradians**.

---

## Configuration Blocks

### `[monte_carlo]`

- `environment` (string): Path to the associated `Environment.toml` base config file.
- `seed` (int): Base random seed for reproducing pointing error draws.
- `runs` (int): Number of batch runs to execute.
- `chain` (array of strings): Ordered list of search strategies to run in sequence (e.g., `["lissajous_scan"]`).
- **Arbitrary Property Overrides**: Any simulation or satellite property can be overridden for the entire batch run using dotted keys or nested sub-tables under `[monte_carlo]` (exactly matching the overrides behavior in [docs/scenario.md](scenario.md)).

### `[monte_carlo.error]`

- `distribution` (string): Type of error distribution (`"uniform"` or `"gaussian"`).
- `uniform.min` (float, milliradians, default: `0.0`): The minimum absolute value of the uniform error.
- `uniform.max` (float, milliradians): The maximum absolute value of the uniform error. The magnitude is drawn from `[min, max]` and assigned a random sign ($+$ or $-$).
- `gaussian.mean` (float, milliradians, default: `0.0`): The mean of the Gaussian distribution.
- `gaussian.std` (float, milliradians): The standard deviation of the Gaussian distribution. Both $\theta$ and $\phi$ offsets are generated independently.

### `[strategy.<strategy_name>]` (Strategy Parameters)

Monte Carlo configuration files contain parameters to tune individual strategies in the chain. For a complete list of parameters available for each strategy, see [docs/strategies.md](strategies.md).

*Example:*

```toml
[strategy.lissajous_scan]
s1_wx = 15.526
s1_wy = 14.249
s1_delta = 2.957
s2_wx = 16.992
s2_wy = 3.821
s2_delta = 5.530
```

