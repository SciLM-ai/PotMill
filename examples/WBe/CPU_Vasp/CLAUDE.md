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
(checked in `resources.worker_layout`, which leaves `DYNAMIC_RESERVE_CORES` (=2) cores free for the
dynamic executor that runs combine_b / cost / pareto):

| stage | jobs/node x cores/job | cores |
|---|---|---|
| entropy (non-strict, ~1 s/config) | 1 x 1 | 1 |
| **labeling (VASP, nested)** | **14 x 8** | **112** |
| featurize | 2 x 2 | 4 |
| fit | 2 x 2 | 4 |
| free for dynamic exe (reserve >= 2) | | 7 |

Across 16 nodes this is **224 concurrent 8-rank VASP jobs** (14/node), each pinned to its own 8
cores. Benchmarked (nested, packed, nelm=10, 20-atom cell, 112 cores): **8-rank BEATS 4-rank** --
KPAR1/NCORE8 ~109/h and KPAR2/NCORE4 111.6/h vs 4r/KPAR1/NCORE4 98.9/h -- because VASP scales ~2x per
step from 4->8 ranks (near-linear, 81.7 -> 42 s/step) while halving the job count cuts per-job
overhead and the tail. **KPAR1/NCORE8 is chosen**: tied-fastest, lowest memory (~49 GB vs KPAR2's
66 GB, no k-group replication), most robust (1 k-group, no k-point-balance dependency). >=4 ranks/
k-group is the SEGV floor, so KPAR=1 (8/group) is safe. featurize/fit are trimmed to 2 cores (both
over-provisioned -- featurize peaked at ~3 concurrent, fit's R-factor SVD is tiny) so labeling gets
the cores.

## VASP launching (nested flux, no orchestrator core)

`vasp_std_pm_cpu_01` is a Cray-MPICH build that only accepts **flux's PMI** — bare launches, Hydra
`mpiexec -fork`, and forked `srun` all fail (PMI mismatch / `PMI2_Initialized` abort). To run it
*without* wasting a core, each labeling worker is a **nested flux instance** (`flux_executor_nesting`,
enabled automatically in cpu mode) that owns `labeling_cores_per_job` cores, and the `[Vasp]`
`command` runs `flux run` **inside** it:

```
[Vasp] command = flux run -N 1 -n 8 -o cpu-affinity=per-task /global/cfs/cdirs/m1883/vasp_bin/vasp_std_pm_cpu_01
```

The nested `flux run` places VASP's 8 ranks on the worker's **own** cores (the broker overlaps), so
there is **no separate orchestrator core** (the GPU/flat-`flux run` layout wasted one Python core per
job). `-n` MUST match `labeling_cores_per_job`; `-N 1` keeps each job's ranks on one node (else the
Slingshot CXI endpoints exhaust: "OFI EP enable failed"); `-o cpu-affinity=per-task` pins one rank
per core (~40% faster).

When labeling finishes, its cores are reclaimed for the fit tail **by resource** (freed cores ÷
`fit_cores_per_job`, floored per node), never by worker count — so fitting can't oversubscribe and
the reserved cores stay free for combine_b/cost/pareto. (Counting workers, as the GPU path does
where label and fit each take 1 GPU, oversubscribes on CPU when `fit_cores > labeling_cores`.)

The binary needs Intel MKL + the classic Intel Fortran runtime on `LD_LIBRARY_PATH`; the sbatch
script exports it (from `intel-oneapi-mixed/2023.2.0`) and it propagates to the nested ranks.

## DFT settings (from `vasp-ase-sp.py`, overridable, any element)

The backend (`potmill/labeling/vasp.py`) applies the exact single-point settings from the reference
script as defaults: `xc=pbe, encut=500, ismear=0, sigma=0.1, ediff=1e-6, kspacing=0.125,
prec=Accurate, nelm=200, lorbit=11, lwave=lcharg=False`. Per-atom initial MAGMOMs (tabulated values;
unknown elements -> 1.0, so it works for any element) are set unless `[Vasp] ispin = 1`. Any of
these is overridable from `[Vasp]`. `setups` is a string: tokens without `:` set the base PAW set,
`El:label` sets a per-element potential (e.g. `recommended W:_sv`). This example also sets the
parallelization-only overrides `kpar = 1` and `ncore = 4` (these never change the converged result).

## Files

- `config.ini` — 4096 configs, `batch_size = 128`, `device = cpu`, binary W-Be, non-strict entropy.
- `FitSNAP.in` — 2-element ACE (`type = W Be`), `pair_style = zero 6.6` >= `max_rcut`.
- `run_perlmutter_cpu.sh` — 16-node CPU sbatch (`-A m4884 -C cpu`, 12 h); sets the Intel runtime
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
