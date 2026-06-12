# Pointing Acquisition Strategy Parameter Optimization

This folder contains a utility script to find optimal parameters for built-in acquisition strategies in the satellite simulator.

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

---

## Pre-requisites & Setup

Both **Grid** and **Random** search require only standard libraries + `numpy` (already installed in the environment).

To use the **Optuna** Bayesian search:
```bash
.venv/bin/pip install optuna
```

---

## CLI Usage

Run the optimization script from the root repository directory:

```bash
# Basic Random Search optimization for random_curve (20 trials)
python optimization/optimize.py --strategy random_curve --method random --trials 20

# Optuna Bayesian optimization for center_rebias (50 trials)
python optimization/optimize.py --strategy center_rebias --method optuna --trials 50

# Grid Search optimization for random_walk (10 grid points)
python optimization/optimize.py --strategy random_walk --method grid --grid-points 10
```

### Command-Line Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--strategy` | `str` | *Required* | Name of the strategy to optimize (e.g., `random_curve`, `lissajous_scan`, `rosette_scan`, `center_rebias`, `random_walk`, `dual_spiral`, `dual_raster`). |
| `--method` | `str` | `random` | Optimization algorithm: `random`, `grid`, or `optuna`. |
| `--trials` | `int` | `20` | Number of iterations for `random` or `optuna` search. |
| `--grid-points` | `int` | `5` | Points per parameter axis for `grid` search. |
| `--eval-runs` | `int` | `30` | Number of pre-sampled scenarios used to evaluate each candidate configuration. Higher is more accurate but slower. |
| `--sim-config` | `str` | `Simulation.toml` | Path to the simulation configuration file. |
| `--mc-config` | `str` | `MonteCarlo_all.toml` | Path to the Monte Carlo configuration file. |
| `--workers` | `int` | `CPU-1` | Maximum parallel worker processes to use. |
| `--seed` | `int` | `42` | Random seed used to pre-sample pointing offsets. |
