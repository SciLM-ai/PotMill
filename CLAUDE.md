# PotMill

Automated active-design pipeline for machine-learned interatomic potentials.

## Overview

PotMill is an HPC pipeline that iteratively generates training data for machine learning interatomic potentials (MLIPs) by maximizing information entropy in the descriptor space. It orchestrates:

1. **Structure generation** (entropy maximization) - generates atomic configurations that maximally explore the bispectrum descriptor space
2. **Labeling** - computes DFT-quality energies and forces using VASP or universal ML potentials (fairchem/UMA)
3. **Featurization** - computes ACE or SNAP descriptors via FitSNAP
4. **Fitting** - least-squares fitting of MLIP coefficients across hyperparameter grid
5. **Pareto front** - identifies optimal hyperparameters balancing accuracy vs computational cost
6. **Uncertainty quantification** - POPSRegression for prediction intervals

## Architecture

The pipeline runs on HPC clusters using [Flux](https://flux-framework.org/) as the job scheduler and [executorlib](https://github.com/pyiron/executorlib) `FluxJobExecutor` for distributed task execution. Three nested executors manage resources:

- `entropy_exe`: block-allocated workers with persistent state for entropy maximization
- `labeling_exe`: block-allocated GPU workers for energy/force labeling
- `exe`: dynamic executor for featurization, fitting, pareto, pops, and batch coordination

### CRITICAL: Nested executor design with futures-based dynamic load balancing

**DO NOT convert the pipeline to sequential phases.** The nested executor structure in `__main__.py` is the core architectural design and must be preserved.

The pipeline works by submitting ALL tasks upfront into a single nested executor context. Tasks declare dependencies via futures (e.g., labeling futures depend on entropy futures, featurization depends on batched labeling, fitting depends on featurization + b_futures, etc.). executorlib and Flux handle scheduling: as soon as a task's dependencies resolve and resources are available, it runs immediately. This gives **dynamic load balancing** — stages overlap naturally:

- Labeling starts as soon as the first entropy config is ready (not after all entropy is done)
- Featurization starts as soon as the first labeling batch completes
- Fitting starts as soon as featurization + b.csv are ready
- GPUs are released early via `labeling_exe.shutdown()` once all labeling futures resolve

This pipelining is essential for GPU utilization: without it, GPUs sit idle waiting for all entropy to finish, then all labeling to finish, etc. The polling loop with `check_and_print_status` monitors progress and triggers early executor shutdown to free resources.

**Never replace this with sequential phases** (e.g., "Phase 1: entropy, Phase 2: labeling, ...") — that destroys the overlap and wastes resources.

## Package structure

```
potmill/
  __main__.py          # Main orchestrator, executor setup, pipeline coordination
  entropy.py           # Bridge to structuregen module
  structuregen/        # Unified structure generation (entropy maximization)
    renorm.py          # Phase 1: random configs for normalization matrices
    optimizer.py       # Phase 2: Monte Carlo entropy maximization
    model.py           # CNModel (MLIAP-compatible JAX model) and CNManager
    calculator.py      # EntropyCalculator (LAMMPS wrapper), generate_random_cell
    lammps_utils.py    # SNAP descriptor file generation, LAMMPS script generation
    samplers.py        # Radius sampling strategies (binary NN-based, Mendeleev-based)
  binary_entropy/      # [Legacy] Original binary-element implementation
  multi_element_entropy/ # [Legacy] Original multi-element implementation
  tools.py             # Config parsing, hyperparameter utilities
  featurize.py         # FitSNAP ACE/SNAP featurization
  fit.py               # Least-squares fitting
  pareto.py            # Pareto front identification
  pops.py              # POPSRegression uncertainty quantification
  vasp.py              # VASP labeling interface
  uma.py               # Universal ML potential (fairchem) labeling
  lammps.py            # LAMMPS-based labeling
  fake_vasp.py         # Mock labeling for testing
  unary.py             # Alternative entry point using pre-existing data
```

## Entropy-based structure generation

The structure generation uses SNAP bispectrum descriptors as the feature space. The goal is to generate atomic configurations that maximize the information entropy (minimize the negative log-determinant of the normalized covariance matrix).

Two methods are supported, controlled by `[STRUCTUREGEN] method` in the inputfile:

- **binary**: Fixed element pair (e.g., W-Re). Uses nearest-neighbor distances for radii, chemically-aware SNAP descriptors (`chemflag=1`). Each element pair has distinct descriptor components.
- **multi_element**: Arbitrary elements sampled from the periodic table using Mendeleev-based radius distributions. Uses pseudo-species mapping where each atom is a unique LAMMPS type with its own cutoff radius. Standard SNAP descriptors without `chemflag`.

Both methods follow the same two-phase approach:
1. **Normalization phase** (`RandomEntropyInitializer`): Generate random configurations to build normalization (renormalization) matrices for the descriptor covariance
2. **Optimization phase** (`EntropyMaximizer`): Monte Carlo search accepting configurations that decrease the log-determinant of the normalized information matrix

## Running

```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m potmill
```

## Run directory placement (ALWAYS use $SCRATCH)

**Always run pipelines on `$SCRATCH` (Lustre), NOT `$WORK` (CFS).** A controlled A/B test
(2026-06-02) showed `$SCRATCH` is ~1.7× faster than CFS for entropy generation due to much
lower metadata-server latency on the many small per-config writes (descriptors, labeling
trajs, features). CPU util on CFS was ~5–20% (workers I/O-blocked) vs ~40% on SCRATCH
(workers actually computing).

Workflow pattern (implemented in `launch_scratch.sh`):
1. `$WORK/PotMill_runs/` keeps persistent inputs (`inputfile_*`, `FitSNAP.in`, `sbatch_*.sh`,
   `launch_scratch.sh`) and small post-run results in `<name>_results/`
   (`pipeline_monitor.csv`, `pareto-front/`, log).
2. `$SCRATCH/PotMill_experiments/<run_name>/` is the working directory during execution —
   all heavy intermediate files (`entropy/`, `labeling/`, `features/`, `fits/`) live here.
3. After the job, `launch_scratch.sh` copies the small artifacts back to
   `$WORK/PotMill_runs/<name>_results/`. The heavy scratch dir is left in place for
   analysis (or eventual scratch purge).

Do **not** put run output dirs under `$WORK/PotMill_runs/<name>` directly anymore — use
`launch_scratch.sh`.

## Configuration

The pipeline is configured via an `inputfile` in the working directory. Key sections:
- `[MAIN]`: Pipeline stage toggles, resource allocation
- `[FitSNAP]`: MLIP type (ACE/SNAP), element specification
- `[STRUCTUREGEN]`: Structure generation method and parameters
- `[RCUT]`, `[NMAX]`, `[LMAX]`, `[TWOJMAX]`, `[EWEIGHT]`: Hyperparameter grids

## Configuration constraints

When `[RCUT] max_rcut` in the inputfile is increased, the `pair_style` cutoff in `FitSNAP.in`
(`[REFERENCE]` section) MUST also be `>= max_rcut`. Otherwise LAMMPS aborts every featurize task
with `rcut > pair_style cutoff` with:

```
ERROR: Compute pace cutoff is longer than pairwise cutoff (src/ML-PACE/compute_pace.cpp:129)
```

The pipeline prints a `WARNING:` line at startup if it detects this mismatch (logic lives in
`potmill/__main__.py` right after the hyperparameter setup). It does NOT auto-override the
user's `pair_style` — users may have custom pair_style setups (more complex than `zero <X>`),
so the right action is to update `FitSNAP.in`:

```
pair_style = zero <X>     # with X >= max_rcut + 0.1
```

(With `restart_limit=3` on the block executors, executorlib fails the offending tasks cleanly
rather than deadlocking the whole pipeline. But the affected tasks' results are still lost, so
this is not a substitute for fixing FitSNAP.in.)

**Agents: if you see the `WARNING: FitSNAP.in [REFERENCE] pair_style cutoff ...` line at startup,
surface it to the user immediately and propose the one-line fix to FitSNAP.in.**

## Dependencies

- executorlib, flux-core, flux-sched (HPC scheduling)
- LAMMPS with MLIAP/SNAP support
- FitSNAP (featurization)
- JAX (entropy model gradients)
- ASE, ase-ga (atomic simulation)
- fairchem-core (UMA labeling) or VASP
- spglib, mendeleev (crystal/element utilities)
- POPSRegression (uncertainty quantification)
- numpy, scipy, pandas, scikit-learn

---

## Completed: Unifying binary_entropy and multi_element_entropy into structuregen

### Status: COMPLETE - Both binary and multi_element paths are fully functional

### What was done

1. **Created `potmill/structuregen/` module** with 7 files combining `binary_entropy/` and `multi_element_entropy/` into one unified directory. Both methods dispatched via `config['method']` ('binary' or 'multi_element').

2. **Updated `entropy.py`** to accept `structuregen_config` parameter and import from `potmill.structuregen` instead of `potmill.binary_entropy`.

3. **Updated `__main__.py`** with:
   - Closure pattern for executorlib's `init_function` (which accepts NO arguments):
     `make_init_atoms_from_entropy(structuregen_config)` returns a closure that captures config.
     Works because executorlib uses cloudpickle (handles closures, unlike stdlib pickle).
   - Parses `[STRUCTUREGEN]` config section with fallback to FitSNAP chem_elem.

4. **Updated `examples/WRe/ACE/inputfile`** with `[STRUCTUREGEN]` section.

### Key design: binary vs multi_element LAMMPS parameterization

The two methods use fundamentally different LAMMPS soft potential parameterizations. Each has its own code path:

**Binary path** (`generate_binary_lammps_scripts` in `lammps_utils.py`):
- Three independent core radii: `core_radius_0` (sampled), `core_radius_1` (fixed at NN dist), `core_radius_cross` (independently sampled)
- Zero script: `pair_style soft 5.0`, A values `10/8/5` for `1-1/1-2/2-2`, cutoffs = `core_radius * 0.9`
- Min script: `pair_style hybrid/overlay soft 5`, A=10 for all, cutoffs = full core radii
- Distance check: per-pair thresholds with single-atom edge cases (cell length proxy)

**Multi_element path** (`generate_lammps_scripts` in `lammps_utils.py`):
- Each atom is a pseudo-species with its own radius; pair cutoff = sum of two radii
- A=10 for all pairs in both scripts
- Global soft cutoff = max core radius
- Distance check: `r_min_i + r_min_j` (natural for per-atom radii)

### Binary `BinaryRadiusSampler` API

`sample_radii()` and `sample_radii_fixed()` return:
```python
(core_radius_0, core_radius_1, core_radius_cross, atom_types, symbols)
```
- `core_radius_0`: element 0, sampled from `NN_dist[elem0] * arange(0.7, 1.8, 0.15)`
- `core_radius_1`: element 1, FIXED at `NN_dist[elem1]`
- `core_radius_cross`: independently sampled from `(NN_dist[elem0]/2 + NN_dist[elem1]/2) * arange(0.7, 1.8, 0.15)`

### Binary `_check_distances_binary` signature

```python
_check_distances_binary(atoms, elements, atom_types, min_dist_0, min_dist_1, min_dist_cross)
```
Each pair type has its own single-radius threshold (NOT sum-of-radii). Handles edge cases: 2-atom cells, single atom of a type (uses min cell length as proxy distance).

### executorlib init_function: closure pattern with worker ID injection

executorlib supports automatic parameter injection via signature introspection. Each worker maintains a `memory` dict containing `{"executorlib_worker_id": <int>}`. When calling any function (including `init_function`), executorlib uses `inspect.getfullargspec()` to check if the function declares an `executorlib_worker_id` parameter — if so, it injects the value from memory as a keyword argument. If the function declares no such parameter, it is called with no arguments (the `args=()` and `kwargs={}` are empty by default).

To pass config AND receive the worker ID:
```python
def make_init_atoms_from_entropy(structuregen_config):
    def init_atoms_from_entropy(executorlib_worker_id):
        # executorlib_worker_id is auto-injected by executorlib
        worker_config = structuregen_config.copy()
        worker_config['_worker_id'] = executorlib_worker_id
        from potmill.entropy import max_entropy_atoms_iterator
        return {"entropy_iterator": max_entropy_atoms_iterator(worker_config)}
    return init_atoms_from_entropy
```
The closure works because executorlib uses `cloudpickle` (not stdlib pickle) for serialization, which handles closures with captured variables. The worker ID injection works for any submitted function, not just `init_function`.

### Files created/modified in this refactoring

**Created (new, untracked):**
- `potmill/structuregen/__init__.py` (empty)
- `potmill/structuregen/model.py` - CNModel with mask slicing, CNManager with jaxnp.linalg.slogdet
- `potmill/structuregen/calculator.py` - EntropyCalculator, generate_random_cell (both variants)
- `potmill/structuregen/lammps_utils.py` - SNAP descriptor files, LAMMPS script generation (both `generate_lammps_scripts` for multi_element and `generate_binary_lammps_scripts` for binary)
- `potmill/structuregen/samplers.py` - BinaryRadiusSampler (returns 3 independent radii), MendeleevUniformRadiusSampler
- `potmill/structuregen/renorm.py` - RandomEntropyInitializer (binary + multi_element), `_check_distances_binary`, `_check_distances_multi`
- `potmill/structuregen/optimizer.py` - EntropyMaximizer (binary + multi_element)

**Modified:**
- `potmill/entropy.py` - now takes structuregen_config param, imports from structuregen
- `potmill/__main__.py` - closure pattern, STRUCTUREGEN config parsing
- `examples/WRe/ACE/inputfile` - added [STRUCTUREGEN] section

**Legacy (kept for reference, not modified):**
- `potmill/binary_entropy/` - original binary implementation
- `potmill/multi_element_entropy/` - original multi-element implementation

---

## Completed: Performance optimizations for entropy maximization

### Status: COMPLETE

### 1. OMP_NUM_THREADS for LAMMPS SNAP parallelization

- `__main__.py`: Passes `n_threads=32` into `structuregen_config`
- `entropy.py`: Sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` **before any LAMMPS/JAX/numpy imports** — critical because OpenMP thread pool size is locked at library load time
- `entropy.py`: Configures JAX for CPU with 64-bit precision early

### 2. Reuse LAMMPS calculators in binary optimizer

- Binary path in `optimizer.py` creates `calculator_relax` and `calculator_min` once in `_init_binary()` with `keep_alive=True`
- LAMMPS scripts are fixed for binary (same radii every iteration), so the same LAMMPS instances are reused across all ~5000-10000 iterations — avoids expensive LAMMPS process creation/teardown each time

### 3. Model reuse via `update_state()`

- `model.py`: Added `CNModel.update_state()` that updates `cross`, `count`, `active`, `K` in-place and clears the JIT cache
- Both `_create_binary_config` and `_create_multi_element_config` in `optimizer.py` call `self.model.update_state()` instead of creating a new CNModel each iteration
- Model is initialized once with `count_=1` (to avoid the early-return branch) and a dummy zero cross matrix
- JIT cache stays at size 1-2 since it's always the same object being traced

### 4. Simplified JAX cache clearing

- `model.py`: Removed the expensive iteration over all `sys.modules` starting with "jax" and calling `cache_clear()` on every object
- With model reuse, the JIT cache naturally stays small (size 1-2), so the threshold of 30 is rarely hit
- When it does trigger, just `self.cn._clear_cache()` + `gc.collect()` is sufficient

---

## Completed: Multi-element entropy speedup via pure Python soft potential + early rejection

### Status: COMPLETE - Tested on Perlmutter 2-node H-Be-W, full pipeline successful

### Problem

The multi_element entropy method was the pipeline bottleneck: 8 GPUs sat idle waiting for structures. Each of ~10,000 MC iterations created 2 LAMMPS processes and ran 130 relaxation steps, but ~99.6% of configs were rejected due to distance violations. Total: ~20,000 LAMMPS process creations and ~1.3M relaxation steps for just 40 accepted configs.

### Solution: Three optimizations (math preserved 100%)

#### 1. `SoftRepulsionCalculator` — pure Python soft potential (`calculator.py`)

New ASE `Calculator` subclass implementing `V(r) = A[1 + cos(pi*r/r_c)]` (identical to LAMMPS soft pair_style) in pure Python using `ase.geometry.get_distances()` for MIC-correct periodic distances. For 12 atoms (66 pairs), this computes in microseconds vs ~100-500ms for LAMMPS process creation. Pair cutoff = `core_radii[i] + core_radii[j]`, matching the multi_element LAMMPS parameterization exactly.

#### 2. Early distance check + deferred LAMMPS creation (`optimizer.py`, `renorm.py`)

After pure-Python soft relaxation (30 steps), check distances BEFORE creating the LAMMPS `EntropyCalculator`. If distances fail, reject immediately with no LAMMPS overhead. Only create LAMMPS + write descriptor file + run entropy relaxation for configs that pass the distance check. This skips ~99.6% of the expensive LAMMPS work.

#### 3. Skip entropy relaxation when model inactive (`optimizer.py`)

For first 10 accepted configs (`active=False`), the entropy model returns zero energy/forces. The 100-step entropy relaxation was redundant (just repeating soft relaxation through hybrid/overlay). Now skipped — only `compute_descriptors()` is called for the single LAMMPS evaluation needed to get SNAP bispectrum.

### Results (H-Be-W, 2 Perlmutter nodes, 128 cores, 8 GPUs)

| Metric | Before | After |
|---|---|---|
| MC iterations for 40 configs | ~10,000 | **41** |
| Acceptance rate | ~0.4% | **~95%** |
| Distance rejections | ~9,960 | **1** |
| LAMMPS processes created | ~20,000 | **~40** |
| Entropy monotonically decreasing | Yes | **Yes (743 → 238)** |

Entropy generation now outpaces GPU labeling — GPUs are fully utilized.

### Why it works

For multi_element with `strict_entropy_decrease=0` (default), acceptance is purely distance-based. The previous code spent ~99.6% of time creating LAMMPS processes and running relaxation for configs that would be rejected for distance violations. The `SoftRepulsionCalculator` produces identical physics to LAMMPS soft potential, so the early distance check after pure-Python soft relax is conservative and correct.

### Files modified

- `potmill/structuregen/calculator.py` — Added `SoftRepulsionCalculator` class, added `from ase.calculators.calculator import Calculator, all_changes`
- `potmill/structuregen/optimizer.py` — Import `SoftRepulsionCalculator`, rewrote `_create_multi_element_config()` with pure Python soft relax, early distance check, deferred LAMMPS creation, conditional entropy relaxation
- `potmill/structuregen/renorm.py` — Import `SoftRepulsionCalculator`, rewrote `_create_multi_element_config()` with same optimizations

### Detailed change log

See `CHANGES_multi_element_speedup.txt` in the repository root for line-by-line diff descriptions, full code snippets, and mathematical verification.
