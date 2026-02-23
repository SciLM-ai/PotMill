#!/bin/bash
#SBATCH -J WRe
#SBATCH --account=m1883_g
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm_minimal_%j.log
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00

pwd; hostname -f; date
#export MPICH_GPU_SUPPORT_ENABLED=1  # Turn on GTL; crucial if transferring data between GPUs on different nodes

# --- Library paths for pip-installed CUDA libs ---
# export PY_SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
# export NVIDIA_DIR="${PY_SITE_PKGS}/nvidia"
# export LD_LIBRARY_PATH="${NVIDIA_DIR}/cuda_runtime/lib:${NVIDIA_DIR}/nvjitlink/lib:${NVIDIA_DIR}/cusparse/lib:${NVIDIA_DIR}/cublas/lib:${NVIDIA_DIR}/cufft/lib:${NVIDIA_DIR}/cudnn/lib:${NVIDIA_DIR}/curand/lib:${NVIDIA_DIR}/cusolver/lib:${NVIDIA_DIR}/nccl/lib:${LD_LIBRARY_PATH}"

srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m autopiad

date
