# Pointing Acquisition Strategy Parameter Optimization

The `src/optimize` module provides a utility to find optimal parameters for built-in acquisition strategies in the satellite simulator.

## Optimization Methodology

Finding the optimal search parameters (like movement speed, frequency ratios, drift deviations, etc.) is challenging because:

1. **Discontinuous Outcomes**: A simulation run either achieves a lock (success) or times out (failure).
2. **Highly Non-Linear**: Small changes in frequencies (e.g., in Lissajous or Rosette scans) can lead to vastly different coverage trajectories.
3. **Stochastic Nature**: Pointing offset distributions (and stochastic strategies like `random_curve`) introduce randomness.

To solve this, the optimizer incorporates several key techniques:

### 1. Common Random Numbers (CRN)

To compare parameter set $A$ and parameter set $B$ fairly, we must evaluate them against the **exact same set of initial pointing offsets**. 

- The script pre-samples $N$ initial offsets (default: 30) from the Gaussian or Uniform distribution defined in the Monte Carlo configuration.
- Every parameter trial is executed against this exact set of offsets, eliminating variance from "lucky" or "unlucky" pointing errors.

### 2. Objective Function Design

We seek to maximize the **success rate** and minimize the **time-to-lock** ($T_{\text{lock}}$). We define a composite Cost Score to minimize:

$$\text{Cost} = (1.0 - \text{Success Rate}) \cdot (2 \times \text{timeout}) + \bar{T}_{\text{lock}}$$

- **Failure Penalty**: A strategy that fails to lock is heavily penalized.
- **Speed Incentive**: Among strategies that achieve a 100% success rate, the optimizer naturally favors parameters that achieve lock faster.

### 3. Optimization Algorithms

The script supports three search methods:

- **Grid Search (`grid`)**: Systematically evaluates combinations on an evenly spaced grid. Recommended only for 1 or 2 parameters to avoid exponential execution time.
- **Random Search (`random`)**: Randomly samples parameters from predefined uniform bounds. Surprisingly robust for higher-dimensional spaces.
- **Bayesian Optimization (`optuna`)**: Uses **Optuna**'s Tree-structured Parzen Estimator (TPE) to build a surrogate model of the objective function, predicting which parameters will perform best. Highly recommended for multi-parameter strategies.

### 4. Physical Speed Validation

Every candidate parameter set is checked analytically against the hardware `max_beam_speed` limit **before** any simulation is run.

If the estimated peak beam angular speed would exceed the limit, the candidate is **resampled immediately** — a replacement candidate is drawn and the trial counter does not advance.  The optimizer always delivers exactly the number of valid evaluated trials you requested.

For **grid search** there is no sampling budget, so out-of-range grid points are simply skipped and noted in the summary.

For **Optuna**, pruned (unphysical) trials are still reported to the study so the surrogate model learns to avoid that region of parameter space, but they do not count toward the trial budget.

This prevents the optimiser from wasting CPU time on parameter combinations that the hardware cannot physically execute, and avoids `ValueError` exceptions propagating up from the strategies.

**Speed estimators by strategy:**


| Strategy                           | Peak speed formula                                       |
| ---------------------------------- | -------------------------------------------------------- |
| `dual_spiral`, `concentric_shells` | `speed × √(w² + (k·sin(R))²)` at `R = max_search_radius` |
| `dual_raster`                      | `2·R·steps·speed / 10` (widest raster chord)             |
| `lissajous_scan`                   | `A·√(wx² + wy²)` where `A = R/√2`                        |
| `rosette_scan`                     | `A·(w1 + w2)`                                            |
| `random_curve`, `center_rebias`    | `velocity_a × velocity_ratio` (faster satellite)         |


A preflight check also runs at startup and will abort with a clear error message if even the **minimum-bound** parameters across the entire search space exceed the limit — saving you from a run where every single trial is rejected.

---

## Pre-requisites & Setup

Both **Grid** and **Random** search require only standard libraries + `numpy` (already installed in the environment).

To use the **Optuna** Bayesian search:

```bash
pip install -e ".[opt]"
```

---

## CLI Usage

Run the optimization script from the root repository directory:

```bash
# Basic Random Search optimization for random_curve (20 trials)
python -m optimize --strategy random_curve --method random --trials 20

# Explicit config file
python -m optimize config/Optimize.toml --strategy lissajous_scan --method optuna --trials 50

# Grid Search optimization for random_walk (10 grid points)
python -m optimize --strategy random_walk --method grid --grid-points 10
```

### Command-Line Options


| Argument / flag | Type  | Default                | Description                                                                                                                                                                   |
| --------------- | ----- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`        | `path` | `config/Optimize.toml` | Optional positional path to the optimize TOML.                                                                                                                                |
| `--strategy`    | `str` | Loaded from config     | Name of the strategy to optimize (e.g., `random_curve`, `lissajous_scan`, `rosette_scan`, `center_rebias`, `random_walk`, `dual_spiral`, `dual_raster`, `concentric_shells`). |
| `--method`      | `str` | Loaded from config     | Optimization algorithm: `random`, `grid`, or `optuna`.                                                                                                                        |
| `--trials`      | `int` | Loaded from config     | Number of iterations for `random` or `optuna` search.                                                                                                                         |
| `--grid-points` | `int` | `5`                    | Points per parameter axis for `grid` search.                                                                                                                                  |
| `--eval-runs`   | `int` | Loaded from config     | Number of pre-sampled scenarios used to evaluate each candidate configuration. Higher is more accurate but slower.                                                            |
| `--env-config`  | `str` | Loaded from config     | Path to the environment configuration file.                                                                                                                                   |
| `--workers`     | `int` | `CPU-1`                | Maximum parallel worker processes to use.                                                                                                                                     |
| `--seed`        | `int` | Loaded from config     | Random seed used to pre-sample pointing offsets.                                                                                                                              |
| `--trial-seed`  | `int` | Loaded from config     | Random seed for random/optuna search.                                                                                                                                         |


---

## Optimization Configuration File (`Optimize.toml`)

Instead of passing all parameters via the command line, the optimizer can be configured using a dedicated TOML file (e.g., `config/Optimize.toml`).

The `[optimize]` block is parsed by [`optimize/config.py`](../src/optimize/config.py). Monte Carlo evaluation settings use the same parsers as batch runs (`montecarlo/config.py`, assembled by `load_monte_carlo_config` in `montecarlo/run.py`).

The configuration file contains two primary blocks:

### `[optimize]` (Search Control)

- `strategy` (string, required): The name of the strategy to optimize (e.g., `"lissajous_scan"`).
- `method` (string): The search algorithm to employ (`"grid"`, `"random"`, or `"optuna"`).
- `trials` (int): Number of search iterations (for `"random"` and `"optuna"`).
- `trial_seed` (int, optional): Seed for the search algorithm's random sampler.

### `[monte_carlo]` (Evaluation Environment)

Defines the parameters used to evaluate each candidate set of parameters.

- `environment` (string): Path to the associated `Environment.toml` base config.
- `seed` (int): Seed for sampling fixed pointing offsets to evaluate candidates.
- `runs` (int): Number of evaluation offsets tested per trial.

### `[monte_carlo.error]`

Defines the error distribution used to sample the pointing offsets (exactly matches the format of `[monte_carlo.error]` described in [docs/monte_carlo.md](monte_carlo.md)).

*Example `Optimize.toml`:*

```toml
[optimize]
strategy = "lissajous_scan"
method = "optuna"
trials = 50
trial_seed = 101

[monte_carlo]
environment = "Environment.toml"
seed = 42
runs = 30
error.distribution = "gaussian"
error.gaussian.std = 2.0
```

