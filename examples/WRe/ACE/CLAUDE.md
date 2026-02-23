# W-Re ACE Example

See the main [CLAUDE.md](../../../CLAUDE.md) for full package documentation.

## Overview

This example runs the full autopiad pipeline for a tungsten-rhenium (W-Re) binary system using ACE (Atomic Cluster Expansion) descriptors. It uses the **binary** entropy method for structure generation, UMA (fairchem universal ML potential) for labeling, and sweeps over ACE hyperparameters (rcut, nmax, lmax, eweight).

## Files

- `inputfile` - Pipeline configuration. Key settings:
  - `[STRUCTUREGEN] method = binary` with `elements = W Re`
  - `[FitSNAP] mlip = ACE` with `chem_elem = W Re`
  - 40 configurations, batched into groups of 20 for incremental fitting
  - Hyperparameter grid: rcut 5-6 (3 values), nmax 8-10/3-4, lmax 0/1-2, eweight centered at 10 (3 values)
- `FitSNAP.in` - FitSNAP configuration for ACE descriptor calculation
- `run_perlmutter.sh` - SLURM submission script for NERSC Perlmutter

## Running

From this directory on a Perlmutter interactive session or via batch:
```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m autopiad
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
