# W-Be CPU + VASP Example

See the main [CLAUDE.md](../../../CLAUDE.md) for full package documentation.

## Overview

Runs the **entire PotMill pipeline on CPU nodes** (NERSC Perlmutter, allocation `m4884`) with
**VASP DFT labeling** instead of UMA — the CPU/VASP counterpart to the GPU/UMA `HBeW` and `WRe`
examples. Structure generation uses the **binary** method for the W-Be pair; labeling runs the
Cray-MPICH VASP binary `vasp_std_pm_cpu_01`; featurization (FitSNAP/LAMMPS) and the incremental
R-collecting fit both run on CPU (`device = cpu`).

## CPU vs GPU: a single switch

`[Main] device = cpu | cuda` drives where labeling + fitting run. Each stage uses the same two
knobs: `<stage>_jobs_per_node` (concurrent jobs) and `<stage>_cores_per_job` (cores each). To move
this example to GPU VASP later: set `device = cuda`, give a GPU VASP `command`, and in the sbatch
script use `-C gpu`, `-A m4884_g`, `--gpus-per-node=4` — no code changes.

## Per-node CPU core budget (128-core node)

The overlapping pipeline keeps every stage progressing at once, so the per-node cores are budgeted
(checked in `resources.worker_layout`, which leaves >=2 cores free for the dynamic executor that
runs combine_b / cost / pareto):

| stage | jobs/node x cores/job | cores |
|---|---|---|
| entropy (strict -> 1 serial worker) | 1 x 8 | 8 |
| **labeling (VASP)** | **4 x (1 Python + 24 VASP)** | **100** |
| featurize | 2 x 4 | 8 |
| fit | 2 x 4 | 8 |
| free for dynamic exe | | 4 (12 once entropy finishes) |

Across 4 nodes this is **16 concurrent VASP jobs**, each pinned to its own 24 cores.

## VASP launching (determined by testing)

`vasp_std_pm_cpu_01` is a Cray-MPICH build. Under `flux start` + executorlib the working launcher
is **flux-native**:

```
[Vasp] command = flux run -n 24 -o cpu-affinity=per-task /global/cfs/cdirs/m1883/vasp_bin/vasp_std_pm_cpu_01
```

The 1-core Python worker calls `flux run`, which grabs 24 cores from the flux instance and pins
each rank to its own core (`-o cpu-affinity=per-task` was ~40% faster than without). Flux schedules
the 4 concurrent VASP jobs on disjoint cores — no oversubscription. The `-n` here MUST match
`[ourLabeling] labeling_cores_per_job`. (Hydra `mpiexec` and nested `srun` do NOT work under flux.)

The binary needs Intel MKL + the classic Intel Fortran runtime on `LD_LIBRARY_PATH`; the sbatch
script exports it (from `intel-oneapi-mixed/2023.2.0`) and it propagates to the `flux run` ranks.

## DFT settings (from `vasp-ase-sp.py`, overridable, any element)

The backend (`potmill/labeling/vasp.py`) applies the exact single-point settings from the reference
script as defaults: `xc=pbe, encut=500, ismear=0, sigma=0.1, ediff=1e-6, kspacing=0.125,
prec=Accurate, nelm=200, lorbit=11, lwave=lcharg=False`. Per-atom initial MAGMOMs (tabulated values;
unknown elements -> 1.0, so it works for any element) are set unless `[Vasp] ispin = 1`. Any of
these is overridable from `[Vasp]`. `setups` is a string: tokens without `:` set the base PAW set,
`El:label` sets a per-element potential (e.g. `recommended W:_sv`). This example also sets the
SaddleMill-derived `ncore = 4` and `isym = 0` as VASP overrides.

## Files

- `config.ini` — 500 configs, `batch_size = 100`, `device = cpu`, binary W-Be, strict entropy.
- `FitSNAP.in` — 2-element ACE (`type = W Be`), `pair_style = zero 6.6` >= `max_rcut`.
- `run_perlmutter_cpu.sh` — 4-node CPU sbatch (`-A m4884 -C cpu`); sets the Intel runtime
  `LD_LIBRARY_PATH` and runs `srun ... flux start python -m potmill` from a `$SCRATCH` run dir.

## Running

```bash
cd $SCRATCH/PotMill_experiments && mkdir my_WBe_cpu && cd my_WBe_cpu
cp <repo>/examples/WBe/CPU_Vasp/{config.ini,FitSNAP.in,run_perlmutter_cpu.sh} .
sbatch run_perlmutter_cpu.sh
```

## Adapting to other elements

The DFT settings and MAGMOMs already work for any element. To change the system: set `elements`
in `[ourStructureGen]` and `chem_elem` in `[FitSNAP]`, update `[Vasp] setups` for any element that
needs a non-default PAW potential, and rebuild `FitSNAP.in` (`type`, N^2 per-bond params, `[ESHIFT]`)
as described in the `HBeW` example. Keep `pair_style` cutoff >= `max_rcut`.
