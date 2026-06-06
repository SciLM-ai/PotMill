# H-Be-W ACE Multi-Element Example

See the main [CLAUDE.md](../../../CLAUDE.md) for full package documentation.

## Overview

This example runs the full PotMill pipeline for a ternary hydrogen-beryllium-tungsten (H-Be-W) system using ACE descriptors. It uses the **multi_element** entropy method for structure generation, which supports arbitrary numbers of elements via Mendeleev-based pseudo-species radius sampling. UMA (fairchem universal ML potential) is used for labeling, and the pipeline sweeps over ACE hyperparameters (rcut, nmax, lmax, eweight).

## Key difference from binary examples (e.g., WRe)

The **binary** method (used for 2-element systems like W-Re) assigns one LAMMPS type per real element and uses chemically-aware SNAP descriptors (`chemflag=1`). The **multi_element** method treats each atom as a unique pseudo-species with its own LAMMPS type and cutoff radius. This means:

- Each of the 12 atoms gets its own LAMMPS type (pseudo-species), with radii sampled from a beta distribution centered on the Pyykko covalent radius of its assigned real element.
- SNAP descriptors use `chemflag=0` and `bzeroflag=1` (standard mode, not chemically-aware).
- `twojmax=8` gives 55 descriptors per atom (vs 112 for binary with `chemflag=1` and `twojmax=4`).
- Distance checking uses sum-of-radii thresholds (natural for per-atom pseudo-species).
- Soft potential uses `A=10` for all pairs and `pair_style soft {r_core_max}` (variable global cutoff).

## Files (self-contained; copy to a $SCRATCH run dir and `sbatch run_perlmutter.sh`)

- `inputfile` - Pipeline configuration matching the proven 100k 4rcut run (1h 42m wall on 4 GPU nodes, 0 errors):
  - `nconfigurations = 100000`, `batch_size = 1000` (configs per combine_b)
  - `label_batch_size = 20` (configs per UMA forward, amortizes the ~160 ms fixed forward overhead)
  - `fit_gpus_per_node = 3` (1 UMA labeling GPU + 3 GPU fit workers per node)
  - `fit_engine = incremental` (R-collecting QR/SVD, O(N) memory in number of batches)
  - Hyperparameter grid: 4 rcuts (5.0-6.5), nmax 5-9 / 2-4, lmax 0 / 1-4, 5 eweights
- `FitSNAP.in` - FitSNAP ACE configuration for 3 elements (9 bond types). `pair_style = zero 6.6` >= `max_rcut`.
- `run_perlmutter.sh` - 4-node premium 4h `sbatch` script. Edit the `USER-SPECIFIC PATHS` block, then `sbatch run_perlmutter.sh` from a fresh `$SCRATCH` working dir.

## STRUCTUREGEN parameters (from original multi_element_entropy)

All values in the `[STRUCTUREGEN]` section match the original `multi_element_entropy/d-opti-chem.py`:

| Parameter | Value | Meaning |
|---|---|---|
| `method` | `multi_element` | Pseudo-species radius sampling with Mendeleev covalent radii |
| `elements` | `H Be W` | Real element species to sample from |
| `twojmax` | `8` | SNAP angular momentum cutoff (55 bispectrum components) |
| `n_atoms` | `12` | Atoms per generated cell (each becomes a pseudo-species) |
| `n_renorm_configs` | `100` | Random configs for Phase 1 normalization |
| `n_optimizer_iterations` | `10000` | Monte Carlo trials in Phase 2 optimization |
| `energy_mode` | `1` | Use per-config mean descriptor (True) vs per-atom (False) |
| `epsilon` | `1e-6` | Regularization for information matrix (multi_element default) |
| `radius_width` | `0.3` | Relative width of beta distribution for radius sampling |
| `radius_beta_a` | `1.25` | Alpha parameter of beta distribution |
| `radius_beta_b` | `1.25` | Beta parameter of beta distribution |
| `volume_scaling_min` | `1.0` | Min volume multiplier (1.0 = sum of exclusion volumes) |
| `volume_scaling_max` | `3.5` | Max volume multiplier |

## FitSNAP.in: ACE bond-type parameters for 3 elements

For N elements, FitSNAP ACE requires N^2 values for per-bond-type parameters (`rcutfac`, `lambda`, `rcinner`, `drcinner`). Bond types are ordered as `itertools.product(elements, elements)`:

For `type = H Be W` (3 elements, 9 bond types):
```
(H,H) (H,Be) (H,W) (Be,H) (Be,Be) (Be,W) (W,H) (W,Be) (W,W)
```

Parameters that are per-rank (`ranks`, `lmax`, `nmax`, `lmin`, `nmaxbase`) do NOT change with element count.

## Adapting to other multi-element systems

To create a new N-element example:

1. **`inputfile`**: Set `elements = A B C ...` in both `[STRUCTUREGEN]` and `chem_elem = A B C ...` in `[FitSNAP]`
2. **`FitSNAP.in`**:
   - `numTypes = N`
   - `mumax = N`
   - `type = A B C ...`
   - `rcutfac`, `lambda`, `rcinner`, `drcinner`: provide N^2 values each
   - `[ESHIFT]`: one entry per element (`A = 0.0`, `B = 0.0`, ...)
3. **`inputfile` `[STRUCTUREGEN]`**: No changes needed to other parameters; the defaults from the original multi_element code work for any element count

## Running

Per the main CLAUDE.md, always run from `$SCRATCH` (Lustre, ~1.7x faster than CFS for the many
small per-config writes). The repo `run_perlmutter.sh` is a 4-node premium 4h `sbatch` script
that handles everything once you edit its `USER-SPECIFIC PATHS` block:

```bash
cd $SCRATCH/potmill_experiments
mkdir my_HBeW_run && cd my_HBeW_run
cp <repo>/examples/HBeW/ACE/{inputfile,FitSNAP.in,run_perlmutter.sh} .
# edit run_perlmutter.sh CONDA_ENV / AUTOPIAD / EXECUTORLIB / SUBDATAPY paths
sbatch run_perlmutter.sh
```

## Output directories

When the pipeline runs, it creates:
- `entropy/` - Entropy-generated atomic configurations (POSCAR files, renormalization data)
- `labeling/` - UMA/VASP energy and force labels for each configuration
- `features/` - FitSNAP ACE descriptor matrices per rcut
- `fits/` - Fitted MLIP coefficients per hyperparameter set per batch
- `costs/` - Computational cost measurements for Pareto analysis
- `pareto-front/` - Pareto front results (accuracy vs cost)
- `pops/` - POPSRegression uncertainty quantification results
