# autopiad

Automated Pipeline for Interatomic Potential Active Design.

## Overview

autopiad is an HPC pipeline that iteratively generates training data for machine learning interatomic potentials (MLIPs) by maximizing information entropy in the descriptor space. It orchestrates:

1. **Structure generation** (entropy maximization) - generates atomic configurations that maximally explore the bispectrum descriptor space
2. **Labeling** - computes DFT-quality energies and forces using VASP or universal ML potentials (fairchem/UMA)
3. **Featurization** - computes ACE or SNAP descriptors via FitSNAP
4. **Fitting** - least-squares fitting of MLIP coefficients across hyperparameter grid
5. **Pareto front** - identifies optimal hyperparameters balancing accuracy vs computational cost
6. **Uncertainty quantification** - POPSRegression for prediction intervals

## Architecture

The pipeline runs on HPC clusters using [Flux](https://flux-framework.org/) as the job scheduler and [executorlib](https://github.com/pyiron/executorlib) `FluxJobExecutor` for distributed task execution. Three nested executors manage resources:

- `labeling_exe`: block-allocated GPU workers for energy/force labeling
- `exe`: dynamic executor for featurization, fitting, pareto, pops, and batch coordination
- `entropy_exe`: block-allocated single worker with persistent state for entropy maximization

## Package structure

```
autopiad/
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
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m autopiad
```

## Configuration

The pipeline is configured via an `inputfile` in the working directory. Key sections:
- `[MAIN]`: Pipeline stage toggles, resource allocation
- `[FitSNAP]`: MLIP type (ACE/SNAP), element specification
- `[STRUCTUREGEN]`: Structure generation method and parameters
- `[RCUT]`, `[NMAX]`, `[LMAX]`, `[TWOJMAX]`, `[EWEIGHT]`: Hyperparameter grids

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

1. **Created `autopiad/structuregen/` module** with 7 files combining `binary_entropy/` and `multi_element_entropy/` into one unified directory. Both methods dispatched via `config['method']` ('binary' or 'multi_element').

2. **Updated `entropy.py`** to accept `structuregen_config` parameter and import from `autopiad.structuregen` instead of `autopiad.binary_entropy`.

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

### executorlib init_function: closure pattern

executorlib's `init_function` is called with NO arguments (hardcoded `args=(), kwargs={}`). To pass config:
```python
def make_init_atoms_from_entropy(structuregen_config):
    def init_atoms_from_entropy():
        from autopiad.entropy import max_entropy_atoms_iterator
        return {"entropy_iterator": max_entropy_atoms_iterator(structuregen_config)}
    return init_atoms_from_entropy
```
This works because executorlib uses `cloudpickle` (not stdlib pickle) for serialization, which handles closures with captured variables.

### Files created/modified in this refactoring

**Created (new, untracked):**
- `autopiad/structuregen/__init__.py` (empty)
- `autopiad/structuregen/model.py` - CNModel with mask slicing, CNManager with jaxnp.linalg.slogdet
- `autopiad/structuregen/calculator.py` - EntropyCalculator, generate_random_cell (both variants)
- `autopiad/structuregen/lammps_utils.py` - SNAP descriptor files, LAMMPS script generation (both `generate_lammps_scripts` for multi_element and `generate_binary_lammps_scripts` for binary)
- `autopiad/structuregen/samplers.py` - BinaryRadiusSampler (returns 3 independent radii), MendeleevUniformRadiusSampler
- `autopiad/structuregen/renorm.py` - RandomEntropyInitializer (binary + multi_element), `_check_distances_binary`, `_check_distances_multi`
- `autopiad/structuregen/optimizer.py` - EntropyMaximizer (binary + multi_element)

**Modified:**
- `autopiad/entropy.py` - now takes structuregen_config param, imports from structuregen
- `autopiad/__main__.py` - closure pattern, STRUCTUREGEN config parsing
- `examples/WRe/ACE/inputfile` - added [STRUCTUREGEN] section

**Legacy (kept for reference, not modified):**
- `autopiad/binary_entropy/` - original binary implementation
- `autopiad/multi_element_entropy/` - original multi-element implementation
