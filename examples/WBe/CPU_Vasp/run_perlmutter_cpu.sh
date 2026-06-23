#!/bin/bash
# WBe CPU + VASP full-pipeline run on Perlmutter (allocation m4884, CPU nodes).
# Submit with:  sbatch run_perlmutter_cpu.sh   (from a fresh $SCRATCH working dir holding
# config.ini + FitSNAP.in, per the main CLAUDE.md "Run directory placement"):
#   cd $SCRATCH/PotMill_experiments && mkdir my_WBe_cpu && cd my_WBe_cpu
#   cp <repo>/examples/WBe/CPU_Vasp/{config.ini,FitSNAP.in,run_perlmutter_cpu.sh} .
#   sbatch run_perlmutter_cpu.sh
#
# To switch to GPU VASP: change "-C cpu"->"-C gpu", "-A m4884"->"-A m4884_g", add
# "--gpus-per-node=4", set [Main] device = cuda and a GPU VASP command in config.ini -- no code edits.

#SBATCH -J potmill_WBe_cpu
#SBATCH -A m4884
#SBATCH -C cpu
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -o run.%j.log

set -uo pipefail

# ---------- USER-SPECIFIC PATHS (edit me) -----------------------------------
CONDA_ENV=/global/cfs/cdirs/m1883/ilgar/conda_envs/potmill   # jax, ase, lammps, fitsnap3lib, torch, executorlib
POTMILL=/global/cfs/cdirs/m1883/ilgar/codes/PotMill          # this repo
SUBDATAPY=/global/cfs/cdirs/m1883/ilgar/codes/SubDataPy      # optional (fit falls back to numpy/CPU)
# ----------------------------------------------------------------------------

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$SUBDATAPY:$POTMILL:${PYTHONPATH:-}"
export WANDB_MODE=disabled

# Intel MKL + classic Fortran runtime for the Cray-MPICH VASP binary (vasp_std_pm_cpu_01); this
# propagates srun -> flux start -> workers -> `flux run` -> VASP ranks. Cray MPICH is already on
# the default /opt/cray/pe path. (intel-oneapi-mixed/2023.2.0 provides these same dirs.)
export LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH:-}"

pwd; hostname -f; date
echo "PYTHONPATH=$PYTHONPATH  NODES=$SLURM_NNODES"

srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" flux start python -u -m potmill

date
