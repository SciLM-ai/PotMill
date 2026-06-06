#!/bin/bash
# Reproduces the 100k 4rcut HBeW run (~2h wall, zero errors on Perlmutter).
# Submit with:  sbatch run_perlmutter.sh
# Edit the paths in the "USER-SPECIFIC PATHS" block below to match your install.
#
# Per the main CLAUDE.md "Run directory placement", submit from $SCRATCH (Lustre
# is ~1.7x faster than CFS for the many small per-config writes). Easiest pattern:
#   cd $SCRATCH/PotMill_experiments
#   mkdir my_run && cd my_run
#   cp <repo>/examples/HBeW/ACE/{config.ini,FitSNAP.in,run_perlmutter.sh} .
#   sbatch run_perlmutter.sh

#SBATCH -J potmill_HBeW
#SBATCH -A m1883_g
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -o run.%j.log

set -uo pipefail

# ---------- USER-SPECIFIC PATHS (edit me) -----------------------------------
CONDA_ENV=/global/cfs/cdirs/m1883/ilgar/conda_envs/potmill   # conda env with jax, ase, lammps, fitsnap3lib, fairchem, torch, executorlib
POTMILL=$HOME/codes/PotMill                                   # this repo's clone
EXECUTORLIB=$HOME/codes/executorlib/src                        # executorlib clone with the PR #589 dynamic max_workers + id()-dedup fix
SUBDATAPY=/global/cfs/cdirs/m1883/ilgar/codes/SubDataPy        # SubDataPy for GPU lstsq (optional -- fit.py falls back if missing)
# ----------------------------------------------------------------------------

export PATH="$CONDA_ENV/bin:$PATH"
# Prepend so the local executorlib + potmill + SubDataPy win over any conda copies.
export PYTHONPATH="$EXECUTORLIB:$SUBDATAPY:$POTMILL:${PYTHONPATH:-}"

# fairchem-core pulls in wandb at import time, and wandb does many filesystem
# stat()s during init -- on a contended CFS this can take 10+ minutes per
# labeling worker (8x parallel). We don't log to wandb, so disable it entirely.
export WANDB_MODE=disabled

pwd; hostname -f; date
echo "PYTHONPATH=$PYTHONPATH"
echo "NODES=$SLURM_NNODES"

# Flux drives executorlib's nested entropy / labeling / featurize / fitting / pareto
# executors. Run python -u so all worker prints flush in real time.
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" flux start python -u -m potmill

date
