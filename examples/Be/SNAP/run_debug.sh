#!/bin/bash
#SBATCH -J test
#SBATCH -A w25_foundation_g
#SBATCH -t 01:30:00
#SBATCH -N 4
#SBATCH -C gpu40
#SBATCH -p gpu_debug
#SBATCH --reservation=gpu_debug
#SBATCH -o test.out
#SBATCH -e test.err

export MPICH_GPU_SUPPORT_ENABLED=0

rm -rf /lustre/scratch5/baghishov/auto_multi_test/tmp/*

srun flux start -o,-Sstatedir=/lustre/scratch5/baghishov/auto_multi_test/tmp python -u -m potmill
# python -u featurize.py /lustre/scratch5/baghishov/auto_multi_test/
