#!/bin/bash
source /users/baghishov/.bashrc

############### This is important for other programs ##########################
echo $FLUX_PMI_LIBRARY_PATH
# The FLUX_PMI_LIBRARY_PATH variable is always created under a flux instance (flux start).
PMIPATH=$(dirname $FLUX_PMI_LIBRARY_PATH)
# This is to stack LD_LIBRARY_PATH exports to look at the conda environment and flux pmi paths
# Suggested by Danny /https://flux-framework.readthedocs.io/en/latest/tutorials/lab/coral2.html 
# BOTH exports are needed for pretty much any MPI process under flux
CONDA_LD="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/lib64
export TMP_LD_LIBRARY_PATH=$CONDA_LD:$PMIPATH:$LD_LIBRARY_PATH
###############################################################################

export VASP_ROOT=/usr/projects/icapt/applications/vasp/vasp-6.4.2-nvidia-gpu
source ${VASP_ROOT}/setenv_chicoma.sh
export MPICH_GPU_SUPPORT_ENABLED=0

# pwd
flux resource list

#use as many gpu as we can
echo "EXECUTING  GPU RUN"
export MPICH_GPU_SUPPORT_ENABLED=1
flux run -n 1 -c 1 -g 1 --env=LD_LIBRARY_PATH=${TMP_LD_LIBRARY_PATH} ${VASP_ROOT}/bin/vasp_std &> vasp.out
