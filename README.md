# PotMill

Automated active-design pipeline for machine-learned interatomic potentials (MLIPs).

PotMill iteratively generates training data for MLIPs by maximizing information entropy in the
descriptor space, then labels, featurizes, fits, and Pareto-ranks candidate potentials — all
orchestrated on HPC clusters with [Flux](https://flux-framework.org/) and
[executorlib](https://github.com/pyiron/executorlib). The stages overlap via a futures-based
dynamic load balancer (see `CLAUDE.md` for the architecture).

## Pipeline stages

1. **Structure generation** — entropy maximization over SNAP bispectrum descriptors (`structuregen/`)
2. **Labeling** — energies/forces from a configurable backend (`labeling/`): UMA (fairchem), VASP, or LAMMPS
3. **Featurization** — ACE/SNAP descriptors via FitSNAP (`featurization/`)
4. **Fitting** — least-squares MLIP coefficients across a hyperparameter grid (`fitting/`)
5. **Pareto front & uncertainty** — accuracy-vs-cost ranking and POPSRegression intervals (`analysis/`, `fitting/`)

## Installation

PotMill needs the Flux scheduler, a Python-enabled LAMMPS (MLIAP/SNAP/ML-PACE), and FitSNAP —
all conda-built or from source — so it is **not** `pip install potmill`. The recipe below is for
**NERSC Perlmutter** (4×A100 GPU nodes, Cray + Slurm); a LANL-Chicoma variant follows. Replace
the `m1883`/`m1883_g` accounts with your own CPU/GPU allocation.

```bash
# 1. Modules (loaded by default on Perlmutter; make them explicit for reproducibility)
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python

# 2. Conda base: Flux + executorlib + GPU hwloc + compiler + MPI (package cache on scratch).
#    CONDA_OVERRIDE_CUDA must match the cudatoolkit module (12.9 here); cuda/ucx/nccl warnings are OK.
mamba config --set pkgs_dirs $SCRATCH/.cache/conda
CONDA_OVERRIDE_CUDA="12.9" mamba create -p $WORK/conda_envs/potmill -c conda-forge \
    python=3.12 flux-core flux-sched executorlib "libhwloc=*=cuda*" cxx-compiler mpi4py
conda activate $WORK/conda_envs/potmill

# 3. GPU Python stack + LAMMPS/FitSNAP build prerequisites (pip).
#    Install these BEFORE building LAMMPS -- its Python lib links the active numpy.
pip install "jax[cuda12]" torch
pip install numpy scipy scikit-learn pandas Cython setuptools psutil tabulate virtualenv sympy pyyaml

# 4. Build LAMMPS as a Python library (MLIAP/SNAP/ML-PACE). The conda `lammps` package omits
#    ACE (`compute pace`) and MLIAP-Python, which PotMill's featurization and entropy steps
#    need -- so build from source (see FitSNAP/docs/source/Installation.rst for detail).
git clone https://github.com/lammps/lammps ~/codes/lammps
cd ~/codes/lammps && mkdir build && cd build
cmake ../cmake -DBUILD_SHARED_LIBS=yes -DMLIAP_ENABLE_PYTHON=yes -DPKG_PYTHON=yes \
      -DPKG_ML-SNAP=yes -DPKG_ML-IAP=yes -DPKG_ML-PACE=yes -DPKG_SPIN=yes \
      -DPYTHON_EXECUTABLE:FILEPATH=$(which python)
make -j 16 && make install-python
export LD_LIBRARY_PATH=$HOME/codes/lammps/build:$LD_LIBRARY_PATH

# 5. FitSNAP (clone + PYTHONPATH; no build).
git clone https://github.com/FitSNAP/FitSNAP ~/codes/FitSNAP
export PYTHONPATH=$HOME/codes/FitSNAP:$PYTHONPATH

# 6. PotMill + UMA + uncertainty extras (pulls ase, ase-ga, spglib, mendeleev, SubDataPy,
#    fairchem-core, POPSRegression; fairchem-core pins torch to a compatible build). Add the
#    [dev] extra for ruff/black/mypy/pre-commit.
git clone https://github.com/IlgarBaghishov/PotMill ~/codes/PotMill
cd ~/codes/PotMill && pip install -e ".[all]"
```

> **Tested on Perlmutter (2026-06)** with python 3.12, jax 0.10.1, jaxlib 0.10.1, torch 2.8.0+cu128,
> fairchem-core 2.20.0, numpy 2.4.6, scipy 1.17.1, ase 3.28.0, executorlib 1.9.3, mendeleev 1.1.0,
> POPSRegression 0.4.0, SubDataPy 0.1.0, LAMMPS 11 Feb 2026 (`b75dfcc930`), FitSNAP `master`. Entropy
> generation is CPU/contention-bound, so wall-clock entropy throughput varies ~±15% with cluster load
> and node draw — don't read small run-to-run timing differences as regressions.

### Verify the install

```bash
# In a GPU allocation, e.g.:
#   salloc -N 2 -A m1883_g -C gpu --gpus-per-node=4 -q interactive -t 04:00:00
srun -n $SLURM_NNODES flux start flux resource list        # Flux sees every node's cores + 4 GPUs

# LAMMPS has the needed packages and the Python stack imports (login node is fine):
python -c "from lammps import lammps; l=lammps(); print([p for p in ('ML-PACE','ML-SNAP','ML-IAP','PYTHON') if p in l.installed_packages])"
python -c "import potmill, fitsnap3lib, fairchem, POPSRegression, subdatapy, mendeleev; print('imports ok')"
python -m unittest discover -s tests                       # the stdlib test suite

# GPU stack (on a GPU node): jax + torch see CUDA, and a real UMA force call runs
# (catches a torch/CUDA misconfig that a bare `import torch` would miss):
python -c "import jax, torch; print('jax', jax.devices()); print('torch cuda', torch.cuda.is_available(), torch.cuda.device_count())"
python -c "
from ase.build import bulk
from fairchem.core import FAIRChemCalculator
a = bulk('Cu', 'fcc', a=3.6, cubic=True); a.pbc = True
a.calc = FAIRChemCalculator.from_model_checkpoint('uma-m-1p1', task_name='omat', device='cuda')
print('UMA forces', a.get_forces().shape)"
```

### `~/.bashrc` (reproducible logins/jobs)

```bash
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate $WORK/conda_envs/potmill
export LD_LIBRARY_PATH=$HOME/codes/lammps/build:$LD_LIBRARY_PATH   # LAMMPS shared lib
export PYTHONPATH=$HOME/codes/FitSNAP:$PYTHONPATH                  # FitSNAP (not pip-installable)
export FAIRCHEM_CACHE_DIR=$SCRATCH/.cache/fairchem                 # UMA weights on scratch
export HF_TOKEN="hf_xxx"                                           # huggingface.co/settings/tokens (UMA download)
export WANDB_MODE=disabled                                         # fairchem imports wandb; skip its slow CFS stats
# Usually unnecessary (torch/jax ship their own CUDA libs); only add if they can't find CUDA at runtime:
# PY_SITE=$(python -c "import site;print(site.getsitepackages()[0])")
# for l in cuda_runtime nvjitlink cusparse cublas cufft cudnn curand cusolver nccl; do \
#   export LD_LIBRARY_PATH=$PY_SITE/nvidia/$l/lib:$LD_LIBRARY_PATH; done
```

### Chicoma (LANL) variant

Steps 3–6 and the verification are identical; only the modules (1) and conda env (2) differ:

```bash
module purge && module load cudatoolkit/24.7_12.5 libfabric
CONDA_OVERRIDE_CUDA="12.5" conda create -n potmill -c conda-forge python=3.11 \
    flux-core flux-sched executorlib openmpi=4.1.6 cxx-compiler mpi4py "libhwloc=*=cuda*" \
    jpeg libpng h5py
```

Chicoma uses `openmpi=4.1.6` (not Cray-MPICH) and `python=3.11`; in the `sbatch` script add
`export MPICH_GPU_SUPPORT_ENABLED=0`, and if Flux state races on the shared filesystem, give it a
per-job statedir: `flux start -o,-Sstatedir=$SCRATCH/flux-$SLURM_JOB_ID python -u -m potmill`.

## Running

From a working directory containing a `config.ini` and a `FitSNAP.in`:

```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m potmill
```

**Always run on `$SCRATCH` (Lustre), not `$WORK` (CFS)** — see `CLAUDE.md` "Run directory placement".
After a run, plot the resource/stage monitor with:

```bash
python -m potmill.analysis.plot_monitor pipeline_monitor.csv
```

## Configuration

The pipeline reads `config.ini` (parsed by `potmill.config.ConfigManager`). Sections are of two kinds:

- **"our" sections** — PotMill's own parameters with documented defaults in `ConfigManager.DEFAULTS`:
  `[Main]` (stage toggles + global counts), `[FitSNAP]` (MLIP + elements), and the per-stage
  `[ourStructureGen]`, `[ourLabeling]`, `[ourFeaturization]`, `[ourFit]`, plus `[ourHyperparameters]`
  (the swept rcut/nmax/lmax/twojmax/eweight grid). Unknown keys are warned about.
- **passthrough sections** — keyword arguments forwarded verbatim to external calculator classes
  (`[FAIRChemCalculator]`, `[Vasp]`, `[LAMMPS]`); omitted keys fall back to that library's defaults.

The labeling backend is selected by `[ourLabeling] calculator` (`FAIRChemCalculator`, `Vasp`, or
`LAMMPS`). Both labeling and fitting devices are configurable (`device` / `fit_device` = `cpu` or `cuda`).

See `examples/` for complete, runnable configs (`HBeW/ACE` is the multi-element UMA reference).

## Examples

| Example | Method | Labeling | Notes |
|---|---|---|---|
| `examples/HBeW/ACE` | multi_element | UMA | Ternary H-Be-W, the proven 100k reference run |
| `examples/WRe/ACE`, `WRe/SNAP` | binary | VASP | W-Re |
| `examples/Be/ACE`, `Be/SNAP` | binary | VASP | Single-element |

## License

BSD-3-Clause (see `LICENSE`).
