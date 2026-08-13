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
6. **LAMMPS potentials** — the selected fits written as ready-to-run `.yace` + `.mod` files (`potential/`)
7. **MD screening** — a short trajectory per exported potential, to catch fits that are accurate but unstable (`md/`)

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
> fairchem-core 2.20.0, numpy 2.4.6, scipy 1.17.1, ase 3.28.0, executorlib 1.9.4, mendeleev 1.1.0,
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

## Running MD with a fitted potential

With `[Main] potential = 1` (the default), the run ends by writing the selected fits as LAMMPS
potentials — by default every point on the Pareto front:

```
potentials/
  index.csv                                          # hyperparameters + CV RMSEs + cost per potential
  rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0/
      rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0.yace   # the potential
      rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0.mod    # pair_style + pair_coeff + element mapping
```

Use one from any LAMMPS input script — `include` the `.mod` after `read_data`:

```lammps
units metal
atom_style atomic
read_data my_structure.data
include rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0.mod
fix 1 all nve
run 10000
```

The same files run on **CPU** (`lmp -in in.lammps`) and on **GPU** via KOKKOS — there is no separate
GPU potential file; `-sf kk` selects `pace/kk` automatically:

```bash
srun -n 4 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half -in in.lammps
```

**Why the `.mod` contains an `if` line.** LAMMPS has two interchangeable algorithms for evaluating
an ACE potential, `product` and `recursive`. They give identical energies and forces; `recursive` is
~18% faster on CPU, but LAMMPS's KOKKOS build *rejects* it (`pair_pace_kokkos.cpp:570` — so any
`-sf kk` run would abort). Rather than make you pick — and discover the problem on your first GPU
job — the `.mod` asks LAMMPS at runtime which one is valid. You never have to think about it.

### What's in `index.csv`

One row per exported potential: its hyperparameters, its k-fold cross-validated RMSEs (energy and
force, train and test), its featurization `cost`, whether it sits on the `pareto_front`, whether it
is the `knee` (the accuracy-favouring pick on the front), and two configuration counts:

| column | meaning |
|---|---|
| `n_configs` | how many configurations **this potential's coefficients** were fitted on |
| `n_configs_errors` | how many configurations the **RMSE and cost columns** describe |

They are equal for a run that finished. They can differ **only if you stopped a run mid-way**: the
fit for each hyperparameter point accumulates batch by batch and keeps going, while errors and the
Pareto ranking are computed at synchronised checkpoints, so an interrupted run leaves some points
having consumed a batch or two beyond the last checkpoint. That is harmless — the extra data only
improves those potentials, and the ranking is still a fair comparison because every point was ranked
on the same checkpoint — but the export prints a `NOTE:` and records both numbers rather than let
you assume they match.

**Stopping a run early is fully supported.** The export always uses the latest *completed*
checkpoint, so if you kill a job once the errors have converged you still get potentials:

```bash
python -m potmill.potential <run_dir> --which pareto
```

The coefficients written are fitted on **all** labeled configurations available to that point, not
on one cross-validation fold, while the reported errors remain the honest k-fold CV values.

### Is the potential stable in MD?

A potential can have an excellent RMSE and still be unusable for dynamics. `[Main] md = 1` runs a
short MD trajectory with every exported potential — in parallel, one task each — and records the
outcome in `potentials/md.csv` (also joined into `index.csv`):

| column | meaning |
|---|---|
| `md_ok` | survived: no crash, no lost atoms, nothing non-finite, no collapsed pairs |
| `md_drift_per_atom_per_ps` | NVE total-energy drift — small means forces really are the gradient of the energy |
| `md_T_final` | final temperature (under `nvt`, a fit that can't hold the thermostat is a red flag) |
| `md_min_dist` | closest approach in Å; below 0.5 Å the structure collapsed |
| `md_note` | why it failed, when it did |

By default the test structure comes from the run itself: the **least compressed** of the 20 lowest
**formation energy per atom** configurations (composition removed by a per-element reference fit),
replicated until it is at least twice the potential cutoff in every direction and holds `min_atoms`
atoms, then relaxed with the potential under test.

That two-step choice is not fussiness — every configuration PotMill generates is entropy-*maximized*,
i.e. deliberately strange. Starting MD from a raw one heats it to thousands of kelvin no matter how
good the potential is; the lowest-energy one often still has a contact inside ACE's inner cutoff, so
MD cannot even start and every potential looks broken; and picking purely on energy made the same
four potentials of one run come back 4/4 stable from one candidate and 0/4 from another. Which
configuration was used, its energy rank, how compressed it is and how many were skipped are all
written to `md/structure.txt`. Point `structure` at your own file to test what you actually care
about — a supplied structure is used exactly as given.

```ini
[Main]
md = 1

[ourMD]
structure = auto        # 'auto', or a path to any ASE-readable structure (used as-is)
min_atoms = 200         # replicate the auto-picked cell up to at least this many atoms
minimize = 1            # relax with the potential before starting MD
ensemble = nvt          # nvt | nve
temperature = 300       # K
timestep = 0.001        # ps
steps = 10000
max_potentials = 32     # how many potentials to test (bounds the number of MD tasks)
```

The same screening runs standalone on any run that already has potentials:

```bash
python -m potmill.md <run_dir> --steps 20000 --temperature 600
```

### Re-checking a potential

The exported files are verified against the fitted model in the test suite, so the pipeline does not
re-check them on every run. Worth running yourself **after upgrading FitSNAP or LAMMPS**, since that
is what could change the answer:

```bash
python -m potmill.potential <run_dir> --which knee --verify 3 --md-steps 2000
```

`--verify N` runs LAMMPS on N labeled structures and compares its energies and forces against the
fitted model (agreement is ~1e-13 eV/atom); `--md-steps` additionally runs a short NVE trajectory
and reports the energy drift.

## Configuration

The pipeline reads `config.ini` (parsed by `potmill.config.ConfigManager`). Sections are of two kinds:

- **"our" sections** — PotMill's own parameters with documented defaults in `ConfigManager.DEFAULTS`:
  `[Main]` (stage toggles + global counts), `[FitSNAP]` (MLIP + elements), and the per-stage
  `[ourStructureGen]`, `[ourLabeling]`, `[ourFeaturization]`, `[ourFit]`, `[ourPotential]`,
  `[ourMD]`, plus `[ourHyperparameters]` (the swept rcut/nmax/lmax/twojmax/eweight grid). Unknown
  keys are warned about.
- **passthrough sections** — keyword arguments forwarded verbatim to external calculator classes
  (`[FAIRChemCalculator]`, `[Vasp]`, `[LAMMPS]`); omitted keys fall back to that library's defaults.

The labeling backend is selected by `[ourLabeling] calculator` (`FAIRChemCalculator`, `Vasp`, or
`LAMMPS`). A single `[Main] device` = `cpu` or `cuda` drives both labeling and fitting placement.
Which fits become LAMMPS potentials is set by `[ourPotential] which` = `none` | `knee` | `pareto`
(default) | `all` — the only key that stage needs.

See `examples/` for complete, runnable configs (`HBeW/ACE` is the multi-element UMA reference).

## Examples

| Example | Method | Labeling | Notes |
|---|---|---|---|
| `examples/HBeW/ACE` | multi_element | UMA | Ternary H-Be-W, the proven 100k reference run |
| `examples/WRe/ACE`, `WRe/SNAP` | binary | VASP | W-Re |
| `examples/Be/ACE`, `Be/SNAP` | binary | VASP | Single-element |

## License

BSD-3-Clause (see `LICENSE`).
