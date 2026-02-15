#!/bin/bash
#SBATCH -J test
#SBATCH -A w25_foundation_g
#SBATCH -t 01:30:00
#SBATCH -N 2
#SBATCH -C gpu40
#SBATCH -p gpu_debug
#SBATCH --reservation=gpu_debug
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --exclude=nid001237

export MPICH_GPU_SUPPORT_ENABLED=0

# rm -rf /lustre/scratch5/baghishov/tmp/*

srun flux start python -u -m autopiad
#srun flux start -o,-Sstatedir=/lustre/scratch5/baghishov/tmp/ python -u -m autopiad
#srun bash -c 'mkdir -p /tmp/flux-job-${SLURM_JOB_ID} && flux start -o,-Sstatedir=/tmp/flux-job-${SLURM_JOB_ID} python -u -m autopiad'
# python -u featurize.py /lustre/scratch5/baghishov/auto_multi_test/
