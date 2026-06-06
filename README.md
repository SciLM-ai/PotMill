# PotMill

(Formerly `autopiad`. The importable package and module entry point are now `potmill`,
e.g. `python -u -m potmill`. The conda env, repo directory, and `$WORK` launchers below
intentionally keep their existing `autopiad` names.)

Installation that worked on Chicoma (it might be outdated now):

module purge

module load cudatoolkit/24.7_12.5

module load libfabric

```CONDA_OVERRIDE_CUDA="12.5" conda create -n executorlib -c conda-forge python=3.11 numpy flux-core flux-sched openmpi=4.1.6 executorlib cxx-compiler mpi4py libhwloc=*=cuda* jpeg libpng ase h5py numpy scipy scikit-learn virtualenv psutil pandas tabulate Cython setuptools sympy pyyaml```

Install LAMMPS and FitSNAP like it is explained in FitSNAP installation guide

conda install spglib jax

pip install POPSRegression




Installation for Perlmutter:

CONDA_OVERRIDE_CUDA="12.9" mamba create -p /global/cfs/cdirs/m1883/ilgar/conda_envs/potmill -c conda-forge python=3.12 flux-core flux-sched executorlib cxx-compiler mpi4py "libhwloc=*=cuda*" ase numpy scipy pandas spglib jax scikit-learn Cython

mamba activate potmill

Check if flux installed properly:
srun -N 2 -n 2 flux start flux resource list

pip install fairchem-core ase-ga POPSRegression mendeleev

Install LAMMPS and FitSNAP like it is explained in FitSNAP installation guide