# GPU Acceleration Plan (Keeping Optuna)

> **Goal:** Maximize GPU throughput for the satellite link-acquisition Monte Carlo simulation  
> while keeping Optuna as the hyperparameter optimizer.  
> **Target hardware:** Any NVIDIA GPU with Compute Capability ≥ 7.0 (Turing+)  
> **Estimated total speedup:** 10–30× over current implementation for 200-trial optimizations

---

## Current Architecture Summary

```
Optuna (CMA-ES Sampler)
  └─ evaluate_candidate()                    # per trial
       ├─ is_strategy_chain_supported_on_gpu()  # builds dummy scenario every call
       ├─ run_monte_carlo_cuda(mc)
       │    ├─ Build strategy scripts (CPU)      # extract legs/hardware arrays
       │    ├─ Generate offsets (CPU, NumPy)      # sample_offsets() per chain
       │    ├─ Transfer to GPU                    # cuda.to_device()
       │    ├─ Launch simulate_batch_kernel       # 256 threads/block, 1 kernel
       │    ├─ cuda.synchronize()
       │    ├─ Copy results to host               # (N, 2) array
       │    └─ Post-process results (CPU)         # Python for-loop, N iterations
       └─ Compute cost from MonteCarloSummary
```

### Identified Bottlenecks (ranked by impact)

| # | Bottleneck | Severity | Location |
|---|---|---|---|
| 1 | **float64 on consumer GPU** — RTX 3050/4000 have 1/32 FP64 throughput | 🔴 Critical | Kernel-wide |
| 2 | **Python result post-processing loop** — rebuilds configs per run | 🔴 High | `cuda_monte_carlo.py` L1022-1097 |
| 3 | **Redundant GPU-compat check** — builds dummy scenario every trial | 🟡 Medium | `cuda_monte_carlo.py` L822-852 |
| 4 | **High register/local-mem pressure** — ~40 `cuda.local.array` calls | 🟡 Medium | Kernel L401-466+ |
| 5 | **No multi-trial batching** — one kernel launch per Optuna trial | 🟡 Medium | `optimize.py` L248-268 |
| 6 | **No CUDA streams** — synchronous launch, no overlap | 🟢 Low | `cuda_monte_carlo.py` L1010-1017 |

---

## Phase 1: Float32 Kernel (Biggest Single Win)

### Rationale
The entire kernel uses `np.float64`. On the Quadro RTX 4000 (Turing, CC 7.5), FP64 throughput
is **1/32 of FP32**. Switching to float32 can yield a **10–20× kernel speedup** with negligible
accuracy loss — the simulation's angular precision requirements (~5 μrad) are well within float32's
~7 decimal digits of precision.

### Changes

**File: `cuda_monte_carlo.py`**

1. **Add a `FLOAT_DTYPE` constant** at module level:
   ```python
   FLOAT_DTYPE = np.float32
   ```

2. **Update all device helper functions** (L45-260) — change every `np.float64` to `FLOAT_DTYPE`:
   - `norm_device`, `normalize_device`, `dot_device`, `cross_device`
   - `angle_between_device`, `slerp_device`, `rotate_vector_device`
   - `rotate_toward_device`, `spherical_angles_from_direction_device`
   - `spherical_to_cartesian_device`, `transmitter_basis_device`
   - `direction_with_tangent_offset_device`, `direction_with_local_offset_device`
   - `offsets_to_target_device`, `beam_hits_dish_device`

3. **Update `get_aim_device()`** (L265-353) — all local arrays → `FLOAT_DTYPE`

4. **Update `simulate_batch_kernel`** (L359-816) — all `cuda.local.array(..., dtype=np.float64)` → `FLOAT_DTYPE`

5. **Update `run_monte_carlo_cuda()` orchestration** (L858-1099):
   - Cast all host arrays to `np.float32` before `cuda.to_device()`:
     ```python
     offsets_arr = offsets_arr.astype(np.float32)
     legs_s1_arr = legs_s1_arr.astype(np.float32)
     # ... etc for all arrays
     sim_params = sim_params.astype(np.float32)
     positions = positions.astype(np.float32)
     d_results = cuda.device_array((total_runs, 2), dtype=np.float32)
     ```

6. **Keep positions as float64 on host** for config building (only cast at the GPU boundary)

### Validation

- [ ] Run full Monte Carlo (`python -m satellite monte-carlo`) with float32 kernel
- [ ] Compare success_rate and mean_t against float64 baseline on same seed
  - Acceptance: success_rate matches exactly, mean_t within ±0.5%
- [ ] Run optimizer with 20 trials, verify same best parameters (±tolerance)
- [ ] Check for any NaN/Inf in results with pathological offsets (large θ/φ)

### 🔖 Commit: `perf: convert CUDA kernel to float32 for 10-20x throughput gain`

---

## Phase 2: Eliminate Python Post-Processing Bottleneck

### Rationale
After the kernel completes, `run_monte_carlo_cuda()` runs a Python for-loop (L1022-1097) that:
- Defines `@dataclass` classes *inside the loop*
- Imports modules inside the loop (`from dataclasses import dataclass`)
- Calls `build_scenario_config()` per run (rebuilds full config objects)
- Creates `MonteCarloRunResult` per run with heavyweight `MockScenarioResult`

For 1000 runs, this adds **hundreds of milliseconds** of pure Python overhead *after* a
kernel that may only take 50ms.

### Changes

**File: `cuda_monte_carlo.py`**

1. **Move dataclass definitions outside the function** (above `run_monte_carlo_cuda`):
   ```python
   @dataclass
   class _MockSchedule:
       total_duration: float

   @dataclass
   class _MockScenarioResult:
       success: bool
       hit_at_t: float | None
       strategy_name: str | None
       schedule: _MockSchedule
       config: object
   ```

2. **Vectorize result extraction** — replace the Python loop with NumPy operations:
   ```python
   results_host = d_results.copy_to_host()
   locked_mask = results_host[:, 0] > 0.5
   hit_times = results_host[:, 1]
   
   success_count = int(locked_mask.sum())
   success_rate = success_count / total_runs
   mean_t = float(hit_times[locked_mask].mean()) if success_count > 0 else float('inf')
   ```

3. **Build MonteCarloRunResult list with pre-computed values** — avoid per-run
   `build_scenario_config()`:
   ```python
   # Pre-build shared config ONCE
   shared_config = build_scenario_config(sim, ScenarioInstance(...), strategy=mc.strategy)
   
   # Build results in batch
   run_results = []
   for idx in range(total_runs):
       run_results.append(MonteCarloRunResult(
           run_index=idx,
           s1_theta=offsets_arr[idx, 0],
           s1_phi=offsets_arr[idx, 1],
           s2_theta=offsets_arr[idx, 2],
           s2_phi=offsets_arr[idx, 3],
           result=_MockScenarioResult(
               success=locked_mask[idx],
               hit_at_t=float(hit_times[idx]) if locked_mask[idx] else None,
               strategy_name=winning_names[idx],  # pre-computed
               schedule=_MockSchedule(total_duration=0.0),
               config=shared_config,
           ),
           computation_time_ms=gpu_time_ms / total_runs,
       ))
   ```

4. **Pre-compute winning strategy names** from kernel output:
   - The kernel currently doesn't output *which* strategy succeeded for multi-chain runs
   - Add a third output column to `results` → `(N, 3)`: `[locked, hit_time, strategy_idx]`
   - In the kernel's lock evaluation block (L777-811), write the current chain index
   - On host, map `strategy_idx → strategy_name` with a simple list lookup

### Validation

- [ ] Run Monte Carlo, verify `MonteCarloSummary` fields match pre-optimization baseline
- [ ] Time the post-processing: should drop from ~200ms to <20ms for 1000 runs
- [ ] Verify `--output` JSON file structure is unchanged

### 🔖 Commit: `perf: vectorize GPU result post-processing, eliminate per-run config rebuilds`

---

## Phase 3: Cache GPU Compatibility Check

### Rationale
`is_strategy_chain_supported_on_gpu(mc)` is called every Optuna trial (via `evaluate_candidate()`
L261). It builds a dummy satellite pair, creates a full MetaStrategy, generates strategy scripts,
and checks all movement types — **even though the strategy chain doesn't change between trials**
(only `params` change, not `chain`).

### Changes

**File: `cuda_monte_carlo.py`**

1. **Add a module-level cache**:
   ```python
   _gpu_compat_cache: dict[tuple[str, ...], bool] = {}
   ```

2. **Check cache before doing work** in `is_strategy_chain_supported_on_gpu()`:
   ```python
   def is_strategy_chain_supported_on_gpu(mc: MonteCarloConfig) -> bool:
       if os.environ.get("SATELLITE_NO_GPU"):
           return False
       if not CUDA_AVAILABLE:
           return False
       
       # Cache key: tuple of all chain names across all chain configs
       cache_key = tuple(name for chain_cfg in mc.chains for name in chain_cfg.chain)
       if cache_key in _gpu_compat_cache:
           return _gpu_compat_cache[cache_key]
       
       # ... existing check logic ...
       
       _gpu_compat_cache[cache_key] = result
       return result
   ```

**File: `optimization/optimize.py`**

3. **Move the check outside the trial loop** — call once before optimization starts:
   ```python
   # In run_optuna_search(), before the ThreadPoolExecutor:
   test_mc = MonteCarloConfig(...)  # with the target chain
   use_gpu = is_strategy_chain_supported_on_gpu(test_mc)
   # Pass use_gpu flag into evaluate_candidate
   ```

### Validation

- [ ] Run optimizer with 50 trials, verify GPU is used on every trial (not just first)
- [ ] Add a log message: `"GPU compatibility: cached (chain: {cache_key})"`
- [ ] Verify CPU fallback still works when chain contains unsupported patterns

### 🔖 Commit: `perf: cache GPU compatibility check across Optuna trials`

---

## Phase 4: Reduce Kernel Register Pressure

### Rationale
The kernel allocates ~40+ `cuda.local.array(3, dtype=...)` calls (L401-466, L536, L565,
L576-583, L620-628, etc.). Each allocation consumes registers or spills to local memory.
High register usage limits GPU occupancy (fewer warps can run concurrently).

### Changes

**File: `cuda_monte_carlo.py`**

1. **Reuse temporary arrays** — many arrays are used only in small scopes:
   ```python
   # Instead of allocating a new array for every intermediate result,
   # declare a small pool of "scratch" arrays at kernel top:
   scratch_a = cuda.local.array(3, dtype=FLOAT_DTYPE)
   scratch_b = cuda.local.array(3, dtype=FLOAT_DTYPE)
   scratch_c = cuda.local.array(3, dtype=FLOAT_DTYPE)
   
   # Then pass these as output parameters to device functions
   # that currently allocate their own arrays
   ```

2. **Audit each device function** for internal allocations:
   - `beam_hits_dish_device` (L233-259): allocates several temp arrays internally
   - `direction_with_local_offset_device` (L187): allocates rotation intermediates
   - Convert these to accept pre-allocated scratch arrays as parameters

3. **Combine related scalar state into arrays** to reduce variable count:
   ```python
   # Instead of: s1_fsm_theta, s1_fsm_phi, s2_fsm_theta, s2_fsm_phi (4 vars)
   # Use: fsm_state = cuda.local.array(4, dtype=FLOAT_DTYPE)  (1 array)
   ```

4. **Use `cuda.jit(max_registers=64)` experimentally** to test occupancy tradeoffs

### Validation

- [ ] Compare kernel execution time before/after on same workload
- [ ] Use `numba.cuda.compile_ptx()` or Nsight Compute to check register count
- [ ] Run full Monte Carlo to verify numerical results unchanged
- [ ] Test with varying `max_registers` values: 32, 48, 64, 80 — report occupancy vs time

### 🔖 Commit: `perf: reduce kernel register pressure via array reuse`

---

## Phase 5: Multi-Trial Kernel Batching

### Rationale
Currently, each Optuna trial launches its own kernel with its own `cuda.synchronize()`.
For a typical trial with N=30-100 Monte Carlo runs at 256 threads/block, that's only 1 block —
the GPU is severely underutilized. By batching multiple trials into a single kernel launch,
we can fill the GPU and amortize launch overhead.

### Design

```
Optuna Thread Pool (ask N trials)
        │
        ▼
   Batch Collector (accumulate trials)
        │
        ▼
   Single Kernel Launch (all trials' offsets concatenated)
        │
        ▼
   Split Results Back to Individual Trials
        │
        ▼
   Optuna Thread Pool (tell N trials)
```

### Changes

**File: `cuda_monte_carlo.py`**

1. **Create `run_monte_carlo_cuda_batch()`** — accepts a list of `MonteCarloConfig`:
   ```python
   def run_monte_carlo_cuda_batch(
       configs: list[MonteCarloConfig]
   ) -> list[MonteCarloSummary]:
       """Launch multiple Monte Carlo configs in a single GPU kernel."""
       # All configs must share same simulation, strategy chain structure
       # (only params/offsets differ)
       
       # Concatenate all offset arrays
       all_offsets = np.vstack([generate_offsets(mc) for mc in configs])
       # Track boundaries: trial_boundaries = [0, N0, N0+N1, ...]
       
       # Build legs arrays (may differ per config due to different params)
       # Strategy A: if all configs have same chain, share legs
       # Strategy B: encode trial_idx in offsets, use per-trial leg arrays
       
       # Launch single kernel over ALL runs
       # Split results by trial_boundaries
       # Return list of MonteCarloSummary
   ```

2. **Handle per-trial parameter differences**:
   - Strategy params (spiral_w, duration, etc.) change between trials
   - **Option A (simpler):** Each trial still gets its own leg arrays, kernel indexes by trial_id
   - **Option B (more efficient):** Encode params directly in an extra column of offsets
   - Recommend **Option A** for correctness, optimize later if needed

**File: `optimization/optimize.py`**

3. **Implement batch ask/tell loop**:
   ```python
   BATCH_SIZE = 8  # Number of trials to batch together
   
   while completed < n_trials:
       # Ask batch of trials from Optuna
       batch_trials = [study.ask() for _ in range(BATCH_SIZE)]
       batch_configs = [build_mc_config(trial) for trial in batch_trials]
       
       # Filter physically valid candidates
       valid = [(t, c) for t, c in zip(batch_trials, batch_configs)
                if is_physically_valid(c)]
       
       # Launch single batched kernel
       summaries = run_monte_carlo_cuda_batch([c for _, c in valid])
       
       # Tell Optuna results
       for (trial, _), summary in zip(valid, summaries):
           cost = compute_cost(summary)
           study.tell(trial, cost)
   ```

> [!WARNING]
> This changes the Optuna interaction pattern from sequential ask/tell to batched.
> CMA-ES may behave slightly differently with batched updates — test convergence quality.

### Validation

- [ ] Run optimizer with batch sizes 1, 4, 8, 16 — measure wall-clock time and GPU utilization
- [ ] Compare optimization quality (best cost) against sequential baseline at 50 trials
- [ ] Verify that batch size 1 produces identical results to current implementation
- [ ] Profile GPU occupancy with Nsight Systems or `nvprof`

### 🔖 Commit: `feat: batch multiple Optuna trials into single GPU kernel launch`

---

## Phase 6: CUDA Streams for Async Overlap

### Rationale
Even with batching, there's still sequential overhead: host→device transfer, kernel execution,
device→host transfer, and Python post-processing all happen serially. CUDA streams allow
overlapping these stages in a pipeline.

### Changes

**File: `cuda_monte_carlo.py`**

1. **Create a stream pool**:
   ```python
   N_STREAMS = 2
   streams = [cuda.stream() for _ in range(N_STREAMS)]
   ```

2. **Pipeline trial batches across streams**:
   ```python
   # Stream 0: transfer batch 0 → launch kernel 0
   # Stream 1: transfer batch 1 → launch kernel 1
   # Stream 0: copy results 0 → process batch 0 (while stream 1 kernel runs)
   ```

3. **Use pinned memory** for host arrays to enable async transfers:
   ```python
   offsets_pinned = cuda.pinned_array((N, 4), dtype=np.float32)
   ```

### Validation

- [ ] Measure kernel-to-kernel latency with vs without streams
- [ ] Verify correctness with 2 and 4 streams
- [ ] Profile overlap with `nsys profile` (or Nsight Systems on Windows)

### 🔖 Commit: `perf: add CUDA streams for async transfer/compute overlap`

---

## Implementation Priority & Expected Speedup

| Phase | Effort | Expected Speedup | Cumulative |
|---|---|---|---|
| **Phase 1: Float32** | 2-3 hours | 10-20× kernel time | 10-20× |
| **Phase 2: Post-processing** | 1-2 hours | 2-5× end-to-end | 15-30× |
| **Phase 3: Cache compat check** | 30 min | 1.1× per trial | 15-30× |
| **Phase 4: Register pressure** | 2-3 hours | 1.3-2× kernel time | 20-40× |
| **Phase 5: Multi-trial batch** | 4-6 hours | 2-4× for optimization | 40-80× |
| **Phase 6: CUDA streams** | 2-3 hours | 1.2-1.5× pipeline | 50-100× |

> [!IMPORTANT]
> **Phases 1-3 are high-impact, low-risk changes** that should be done first.
> Phases 4-6 are more invasive and should be validated carefully.
> The cumulative speedup estimates assume a 200-trial optimization with 100 MC runs per trial.

---

## Testing Strategy

### Baseline Capture (Do Before Any Changes)

```bash
# Record baseline timing and results with a fixed seed
python -m satellite monte-carlo --seed 42 --output baseline_results.json
# Save: success_rate, mean_t, median_t, per-run hit_times

# Record baseline optimization
cd optimization && python optimize.py --trials 20 --seed 42 --output baseline_optim.json
# Save: best_cost, best_params, wall_time
```

### Per-Phase Regression Tests

After each phase, run:
1. **Correctness check**: Same seed → same success_rate (within tolerance)
2. **Performance check**: Time the kernel and end-to-end, log in a results table
3. **Stress test**: 1000+ MC runs to catch rare numerical issues
4. **Cross-GPU test**: Verify on both RTX 3050 (Ampere) and Quadro RTX 4000 (Turing)

### Acceptance Criteria

| Metric | Tolerance |
|---|---|
| `success_rate` (same seed) | Exact match (Phase 1: ±1 run in 1000) |
| `mean_t` (same seed) | ±1% (float32 rounding) |
| Best optimization cost (20 trials) | ±5% |
| No NaN/Inf in results | Zero tolerance |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Float32 precision loss in edge cases | Test with extreme offsets (near π). Keep float64 fallback flag |
| Multi-trial batching breaks CMA-ES convergence | Compare convergence curves at 50/100/200 trials |
| Register reduction changes numerical results | Verify with bit-exact comparison tool before/after |
| CUDA stream complexity increases debugging difficulty | Phase 6 is optional; only pursue if Phases 1-5 aren't sufficient |
| Different results on different GPUs | Use deterministic seeds, document expected per-GPU variance |

---

## Non-Goals (Explicitly Out of Scope)

- ❌ Replacing Optuna with a GPU-native optimizer
- ❌ Porting offset sampling to GPU (NumPy RNG is fast enough)
- ❌ Supporting unsupported movement patterns on GPU (Circle, Line, DiscretePattern)
- ❌ Multi-GPU support
- ❌ Mixed-precision (FP16/TF32) — not enough precision for angular calculations
