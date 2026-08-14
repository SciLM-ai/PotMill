# PotMill

Automated active-design pipeline for machine-learned interatomic potentials.

## Working principle (agents: non-negotiable)

**Never silently implement something inconsistent because you couldn't find what you needed.**
If a required input is missing or ambiguous (a file, structure source, reference value, config),
STOP and raise it to the user — never assume, and never apply a different/inconsistent method to
one case than another and bury it in a caveat. Validate every reconstruction/derivation against
ground truth (e.g. a reconstructed RMSE must match the stored RMSE), and never paper over a
discrepancy — surface and fix it, or escalate.

## Overview

PotMill is an HPC pipeline that iteratively generates training data for machine learning interatomic potentials (MLIPs) by maximizing information entropy in the descriptor space. It orchestrates:

1. **Structure generation** (entropy maximization) - generates atomic configurations that maximally explore the bispectrum descriptor space
2. **Labeling** - computes DFT-quality energies and forces using VASP or universal ML potentials (fairchem/UMA)
3. **Featurization** - computes ACE or SNAP descriptors via FitSNAP
4. **Fitting** - least-squares fitting of MLIP coefficients across hyperparameter grid
5. **Pareto front** - identifies optimal hyperparameters balancing accuracy vs computational cost
6. **LAMMPS potential export** - writes the selected fits as ready-to-run LAMMPS potentials
7. **MD screening** - a short trajectory per exported potential
8. **Uncertainty quantification** - a streaming POPS error bar shipped with each potential, usable from ASE

## Architecture

The pipeline runs on HPC clusters using [Flux](https://flux-framework.org/) as the job scheduler and [executorlib](https://github.com/pyiron/executorlib) `FluxJobExecutor` for distributed task execution. Nested executors manage resources:

- `entropy_exe`: block-allocated workers with persistent state for entropy maximization
- `labeling_exe`: block-allocated GPU workers for energy/force labeling
- `featurize_exe`: block-allocated workers for FitSNAP featurization
- `fitting_exe`: block-allocated GPU workers for least-squares fitting
- `exe`: dynamic executor for batch coordination (combine_b), pareto, pops, and cost

### CRITICAL: Nested executor design with futures-based dynamic load balancing

**DO NOT convert the pipeline to sequential phases.** The nested executor structure in `__main__.py` is the core architectural design and must be preserved.

The pipeline works by submitting ALL tasks upfront into a single nested executor context. Tasks declare dependencies via futures (e.g., labeling futures depend on entropy futures, featurization depends on batched labeling, fitting depends on featurization + b_futures, etc.). executorlib and Flux handle scheduling: as soon as a task's dependencies resolve and resources are available, it runs immediately. This gives **dynamic load balancing** — stages overlap naturally:

- Labeling starts as soon as the first entropy config is ready (not after all entropy is done)
- Featurization starts as soon as the first labeling batch completes
- Fitting starts as soon as featurization + b.csv are ready
- GPUs are released early via `labeling_exe.shutdown()` once all labeling futures resolve

This pipelining is essential for GPU utilization: without it, GPUs sit idle waiting for all entropy to finish, then all labeling to finish, etc. The polling loop with `check_and_print_status` monitors progress and triggers early executor shutdown to free resources.

**Never replace this with sequential phases** (e.g., "Phase 1: entropy, Phase 2: labeling, ...") — that destroys the overlap and wastes resources.

### CRITICAL: never make futures setup data-dependent (keep it dynamic)

All futures are submitted UPFRONT (before the polling `while` loop) so tasks start the instant their
dependencies resolve. **Do NOT compute anything at setup time that requires a future's RESULT** —
e.g., do not size a labeling task's resources from the structure's atom count, because that forces
the setup code to block on the entropy future before it can submit the labeling future, serializing
the whole submission and gutting the dynamic overlap (the package's main selling point). Per-task
resources must be UNIFORM and config-driven (static), never data-dependent. The same rule killed an
earlier "allocate VASP cores per structure size" idea — it's not allowed.

### Scale/throughput regime (optimize for this, not per-structure latency)

PotMill runs at SCALE: ~100k+ structures across as many nodes as available (100+), and the goal is to
collect AS MUCH labeled DATA as possible within a wall-clock budget (e.g., 24 h) at high CPU/GPU
utilization. Optimize for **total throughput (structures/node-hour) and utilization, NOT per-structure
latency.** Labeling is embarrassingly parallel, so prefer MANY SMALL jobs over FEW LARGE parallel jobs:
parallel speedup is ≤ linear, so the most throughput-efficient layout is the FEWEST cores per job that
still fits (down to 1-core serial VASP — zero MPI/KPAR overhead), running as many concurrent jobs as
**memory (capacity + bandwidth)** allows. Per-job MPI/KPAR parallelism only helps latency, which we
don't care about here. Tune the uniform `<stage>_cores_per_job` to the memory-bound throughput peak.

## Package structure

```
potmill/
  __main__.py          # Orchestrator skeleton: executor setup, task submission, polling loop
  config.py            # ConfigManager (config.ini defaults/coercion/validation) + load_fitsnap_config
  resources.py         # Flux allocation query + per-stage worker layout
  pipeline.py          # Orchestration helpers: entropy init, combine_b, run-dir setup, progress
  bfile.py             # The labeling->fitting b-file format (write_b / read_b)
  tools.py             # Config value coercion, hyperparameter grid/string utilities
  monitor.py           # ResourceMonitor (background GPU/CPU/task-progress logger)
  entropy.py           # Bridge to structuregen module
  structuregen/        # Unified structure generation (entropy maximization)
    renorm.py          # Phase 1: random configs for normalization matrices
    optimizer.py       # Phase 2: Monte Carlo entropy maximization
    model.py           # CNModel (MLIAP-compatible JAX model) and CNManager
    calculator.py      # EntropyCalculator (LAMMPS wrapper), SoftRepulsionCalculator, random cells
    lammps_utils.py    # SNAP descriptor file generation, LAMMPS script generation
    samplers.py        # Radius sampling strategies (binary NN-based, Mendeleev-based)
  labeling/            # Energy/force labeling backends, selected by [ourLabeling] calculator
    __init__.py        # make_labeling(config) backend selector
    uma.py             # UMA (fairchem) backend, configured via [FAIRChemCalculator]
    vasp.py            # VASP backend, configured via [Vasp]
    lammps.py          # LAMMPS backend, configured via [LAMMPS]
  featurization/       # FitSNAP ACE/SNAP featurization
  fitting/             # Least-squares fitting (fit.py, foldfit) + POPSRegression UQ (pops.py)
  analysis/            # Pareto front (pareto.py) + monitor plotting (plot_monitor.py)
  potential/           # LAMMPS potential export ([Main] potential)
    export.py          # point selection (none|knee|pareto|all), index.csv, per-point error isolation
    betas.py           # ALL-DATA coefficients merged from the per-fold R-factors (see below)
    labels.py          # ACE descriptor-label reconstruction (blist + symbolic nu), asserted vs featurize
    ace.py             # beta -> AcePot -> .yace (minimal basis; coefficients attached BY LABEL)
    mod.py             # .mod include file; [REFERENCE] zero / hybrid handling
    verify.py          # LAMMPS vs natoms*(a_E@beta) / a_F@beta cross-check + NVE MD probe
  md/                  # MD stability screening of the exported potentials ([Main] md)
    structure.py       # test structure: lowest formation energy per atom, replicated (or user file)
    runner.py          # minimize + MD via LAMMPS; drift / temperature / collapse metrics
    stage.py           # prepare -> N parallel md tasks -> merge into potentials/md.csv + index.csv
  uq/                  # POPS uncertainty shipped with each exported potential ([Main] uq)
    pops.py            # streaming POPS estimator (O(p^2) memory) + evidence ridge from a Gram
    artifact.py        # the <name>.uq.npz written beside the .yace + split-conformal calibration
    stage.py           # N parallel uq tasks -> potentials/uq.csv + index.csv
    calculator.py      # PotMillCalculator: ASE calculator with get_uncertainty()/get_bounds()
```

Tests live in `tests/` (stdlib `unittest`); run them with `python -m unittest discover -s tests`.

## Entropy-based structure generation

The structure generation uses SNAP bispectrum descriptors as the feature space. The goal is to generate atomic configurations that maximize the information entropy (minimize the negative log-determinant of the normalized covariance matrix).

Two methods are supported, controlled by `[ourStructureGen] method` in `config.ini`:

- **binary**: Fixed element pair (e.g., W-Re). Uses nearest-neighbor distances for radii, chemically-aware SNAP descriptors (`chemflag=1`). Each element pair has distinct descriptor components.
- **multi_element**: Arbitrary elements sampled from the periodic table using Mendeleev-based radius distributions. Uses pseudo-species mapping where each atom is a unique LAMMPS type with its own cutoff radius. Standard SNAP descriptors without `chemflag`.

Both methods follow the same two-phase approach:
1. **Normalization phase** (`RandomEntropyInitializer`): Generate random configurations to build normalization (renormalization) matrices for the descriptor covariance
2. **Optimization phase** (`EntropyMaximizer`): Monte Carlo search accepting configurations that decrease the log-determinant of the normalized information matrix

## Running

```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m potmill
```

## Run directory placement (ALWAYS use $SCRATCH)

**Always run pipelines on `$SCRATCH` (Lustre), NOT `$WORK` (CFS).** A controlled A/B test
(2026-06-02) showed `$SCRATCH` is ~1.7× faster than CFS for entropy generation due to much
lower metadata-server latency on the many small per-config writes (descriptors, labeling
trajs, features). CPU util on CFS was ~5–20% (workers I/O-blocked) vs ~40% on SCRATCH
(workers actually computing).

Workflow pattern (implemented in `launch_scratch.sh`):
1. `$WORK/PotMill_runs/` keeps persistent inputs (`config.ini`, `FitSNAP.in`, `sbatch_*.sh`,
   `launch_scratch.sh`) and small post-run results in `<name>_results/`
   (`pipeline_monitor.csv`, `pareto-front/`, log).
2. `$SCRATCH/PotMill_experiments/<run_name>/` is the working directory during execution —
   all heavy intermediate files (`entropy/`, `labeling/`, `features/`, `fits/`) live here.
3. After the job, `launch_scratch.sh` copies the small artifacts back to
   `$WORK/PotMill_runs/<name>_results/`. The heavy scratch dir is left in place for
   analysis (or eventual scratch purge).

Do **not** put run output dirs under `$WORK/PotMill_runs/<name>` directly anymore — use
`launch_scratch.sh`.

## Configuration

The pipeline is configured via a `config.ini` in the working directory, parsed by
`potmill.config.ConfigManager`. "Our" sections have documented defaults in
`ConfigManager.DEFAULTS` and warn on unknown keys; passthrough sections forward kwargs verbatim
to external calculators. Key sections:
- `[Main]`: pipeline stage toggles (`entropy`/`labeling`/`featurize`/`fit`/`pareto`/`potential`/`md`/`uq`, plus the legacy `pops`), `nconfigurations`, `batch_size`, and `device` = `cuda` | `cpu` (drives labeling + fitting placement)
- `[FitSNAP]`: MLIP type (ACE/SNAP), element specification, FitSNAP.in filename
- `[ourStructureGen]`: structure generation method and parameters (defaults resolved in `structuregen`), plus `entropy_jobs_per_node`, `entropy_cores_per_job`
- `[ourLabeling]`: `calculator` = `FAIRChemCalculator` | `Vasp` | `LAMMPS`, plus `label_batch_size`, `labeling_jobs_per_node`, `labeling_cores_per_job`
- `[ourFeaturization]`: `featurize_jobs_per_node`, `featurize_cores_per_job`
- `[ourFit]`: `fit_jobs_per_node`, `fit_cores_per_job`, `fit_method`, `n_fold`, `fit_engine`
- Per-stage layout is uniform: each stage has `<stage>_jobs_per_node` concurrent jobs of `<stage>_cores_per_job` cores. In `cuda` mode each labeling/fit job takes one GPU; in `cpu` mode each takes its cores and `worker_layout` checks the per-node sum leaves cores free for the dynamic executor.
- `[ourPotential]`: `which` = `none` | `knee` | `pareto` (default) | `all` (its only key -- the written `.mod` chooses its pace evaluator at runtime, see below)
- `[ourMD]`: MD screening of the written potentials -- `structure` (`auto` | path), `min_atoms`, `minimize`, `ensemble`, `temperature`, `timestep`, `steps`, `max_potentials`, `md_cores_per_job`
- `[ourUQ]`: POPS uncertainty per written potential -- `posterior` (`hypercube` | `ensemble`), `minimum_relative_error`, `percentile_clipping`, `max_potentials`, `uq_cores_per_job`
- `[ourHyperparameters]`: the swept grid (`min/max_rcut`, `num_rcut`, `min/max_nmax`, `min/max_lmax`, `min/max_twojmax`, `middle_eweight`, `num_eweights`)
- `[FAIRChemCalculator]`, `[Vasp]`, `[LAMMPS]`: passthrough kwargs for the chosen labeling backend

## Configuration constraints

When `[RCUT] max_rcut` in `config.ini` is increased, the `pair_style` cutoff in `FitSNAP.in`
(`[REFERENCE]` section) MUST also be `>= max_rcut`. Otherwise LAMMPS aborts every featurize task
with `rcut > pair_style cutoff` with:

```
ERROR: Compute pace cutoff is longer than pairwise cutoff (src/ML-PACE/compute_pace.cpp:129)
```

The pipeline prints a `WARNING:` line at startup if it detects this mismatch (logic lives in
`ConfigManager.validate()`, called from `__main__`). It does NOT auto-override the
user's `pair_style` — users may have custom pair_style setups (more complex than `zero <X>`),
so the right action is to update `FitSNAP.in`:

```
pair_style = zero <X>     # with X >= max_rcut + 0.1
```

(With `restart_limit=3` on the block executors, executorlib fails the offending tasks cleanly
rather than deadlocking the whole pipeline. But the affected tasks' results are still lost, so
this is not a substitute for fixing FitSNAP.in.)

**Agents: if you see the `WARNING: FitSNAP.in [REFERENCE] pair_style cutoff ...` line at startup,
surface it to the user immediately and propose the one-line fix to FitSNAP.in.**

## LAMMPS potential export (`potmill/potential/`)

`[Main] potential = 1` ends a run by writing the selected fits (default: the whole Pareto front) as
`potentials/<point>/<point>.yace` + `.mod`, plus one `potentials/index.csv` carrying each point's
hyperparameters, k-fold CV RMSEs and cost. The same code is a CLI for finished runs:
`python -m potmill.potential <run_dir> [--which ...] [--verify N] [--md-steps N]`.
The stage is submitted UPFRONT into the dynamic `exe` with its dependencies as futures (final
featurization for descriptor labels, final pareto for the selection, final fit chain for the
accumulated state), so it does not serialize anything.

Facts that are easy to get wrong, all established by measurement:

- **The evaluator must be chosen at RUNTIME, not written into the file.** `pair_style pace` has two
  algorithms, `product` and `recursive`, which produce identical energies and forces (<1e-15 eV/atom
  across ranks 1..4, lmax<=4, bases to 4308 columns). `recursive` is ~18% faster on CPU at a
  production basis (1254 columns: 86.7 vs 106.0 us/atom/step, same at 250 and 1024 atoms) -- but
  `src/KOKKOS/pair_pace_kokkos.cpp:570` hard-errors on it (`"Must use 'product' algorithm with pair
  pace/kk on the GPU"`), inside `compute()` with no execution-space branch, and `pace/kk/host` is a
  registered style too. So the rule is KOKKOS => product, not merely GPU => product; a `.mod` that
  hardcodes `recursive` aborts every `-sf kk` run. The writer emits
  `if "$(is_active(package,kokkos))" then "pair_style pace product" else "pair_style pace recursive"`
  so the user never chooses (and FitSNAP's hardcoded `product` is explained: it is the safe half).

- **ALL-DATA coefficients, for free.** A shipped potential must not be one CV fold's fit (that
  throws away 1/k of the data), and re-fitting from the cumulative design matrix is the O(N^2)
  reload the incremental engine exists to avoid. Because the k test sets PARTITION the data, every
  row is in exactly `k-1` training sets, so QR-merging the k saved augmented solve R-factors gives
  the R of `sqrt(k-1) x` the full weighted design matrix — and an LS solution from an augmented R is
  scale-invariant, so that factor cancels. Only the global normalizations are rescaled
  (`Sw_full = sum_f Sw_f / (k-1)`). `tests/test_potential.py` pins this against a one-shot `lstsq`.
- **A swept (nmax, lmax) point is a genuine ACE basis.** The column-filtered full label list is set-
  AND order-identical to the labels generated directly at that point's nmax/lmax (verified for the
  HBeW grid), so each potential carries only its own functions — minimal basis, fastest MD — instead
  of the full basis padded with zeros. Coefficients are still attached BY SYMBOLIC LABEL (`nu`), and
  the label sets are asserted equal per element before anything is written.
- **`AcePot.write_pot` truncates E0 to six decimals** (`'%f'`, while every other number goes through
  `json.dumps`). That alone shifted energies by ~2e-7 eV/atom with forces still exact (E0 is a
  constant). `ace._rewrite_e0_full_precision` rewrites the line; agreement then goes to ~1e-13.
- **Both pace evaluators are exact**; `recursive` (LAMMPS's default, ~10% faster than FitSNAP's
  hardcoded `product` on a 250-atom NVE benchmark) is the default. Measured `<1e-15` eV/atom vs the
  fitted model for ranks 1..4, lmax up to 4, bases up to 4308 columns.
- **`bzeroflag = 1` is refused**: `featurize` always prepends a `[[0]]` constant label per element,
  which bzeroflag=1 removes from the design matrix, so such a run's fit is already inconsistent.
- **Interrupted runs are supported, and that is why `index.csv` has two configuration counts.**
  `foldfit` rewrites each subset's `state.pt` IN PLACE every batch, so a past batch's fit state does
  not exist anywhere; errors and the Pareto ranking, by contrast, are only written at synchronised
  checkpoints (`results_<b>.csv` appears once every subset has folded batch b). Chains advance
  independently -- a 1254-column subset takes far longer per link than a 174-column one -- so a
  killed run leaves points ahead of the last checkpoint by differing amounts. The export therefore
  takes the newest checkpoint and reports BOTH `n_configs` (per potential, read from its own
  accumulator: `n[('tr','E')] + n[('te','E')]`, asserted equal across folds) and `n_configs_errors`
  (what the checkpoint's RMSEs describe), printing a NOTE when they differ. Equal for a finished run.
  Snapshotting states per checkpoint is not an option: they are `n_fold * 10 * (p+1)^2 * 8` bytes --
  measured 4.1 MB at p=174, and 378 MB per subset at p=1254, i.e. ~20 GB for one snapshot of the
  HBeW grid. TRIED AND REJECTED (do not re-attempt without new evidence): having `foldfit` write the
  merged all-data beta into its per-batch `betas_{pid}.bin` (fold `-1`) would align coefficients and
  errors by construction for ANY batch and make the export near-instant, but it was IMPLEMENTED and
  MEASURED at **+23.6% fit work on GPU** (p=1254, 40k rows, 5 eweights: 5.87 s -> 7.25 s per link;
  +18.7% at p=400; ~+25% on CPU). The cost is one extra SVD per eweight and is intrinsic -- hoisting
  the eweight-independent merge out of the loop only recovered 2 points. On the 100k GRACE run that
  is +8 min on a 33 min fit tail, i.e. ~7% of every run, forever, to remove one CSV column and make
  a rare historical-batch export exact. The reconstruction below is already validated exact (1e-11
  vs a one-shot fit) and the two counts are equal for any completed run. It must NOT be
  written into `results_*.csv`, which pareto averages over folds.
- Gotchas when touching this code: `featurize()` chdirs into its feature directory and never returns
  (save/restore cwd, and keep run paths absolute); LAMMPS forces must come from `gather_atoms`
  (`extract_atom("f")` is dimensioned `[0:nmax]` and includes ghost atoms); the `.mod` refers to its
  `.yace` by bare filename, so LAMMPS must run with cwd = the potential dir.

Verification lives in `potential/verify.py` and is the definition of correct here: LAMMPS energy and
forces vs `natoms*(a_E@beta)` and `a_F@beta` from a fresh FitSNAP featurization of the same
structure. On the HBeW 5000-config run this agrees to ~1e-13 eV/atom and ~1e-12 eV/A.

**It is deliberately NOT a pipeline stage.** Two different questions get called "verification" here:
the writer's structural checks (labels vs featurization, coefficient counts, per-element blocks)
catch per-RUN problems -- an edited grid, a FitSNAP that reordered the basis -- and stay in the
export. Re-deriving energies through LAMMPS tests the writer and the installed FitSNAP/LAMMPS, which
do not vary run to run, so running it every run re-proves the same fact; it lives in
`tests/test_potential.py` (CI) and behind `--verify` for use after upgrading either package.
`--md-steps` runs a short NVE probe (energy drift).

## MD stability screening (`potmill/md/`, `[Main] md`)

A different question from verification: not "was the file written correctly?" (a property of the
writer, constant per install) but "is this FIT usable for dynamics?" -- a property of the training
coverage and hyperparameters, which is why it runs per potential and belongs next to the Pareto
table. Results land in `potentials/md.csv` (authoritative) and are joined into `index.csv`.

### `rcinner` builds an atom trap into every exported potential

**What the parameter does** (`ML-PACE/ace-evaluator/ace_radial.cpp`, `ACERadialFunctions::radbase`):
`rcinner`/`drcinner` in `FitSNAP.in [ACE]` become the yace `rcut_in`/`dcut_in`. For any pair with
`r <= rcut_in - dcut_in` the code does `gr.fill(0); dgr.fill(0)` -- EVERY radial basis function is
set to zero, so the two atoms become completely invisible to each other. Between `rcut_in - dcut_in`
and `rcut_in` a 5th-order polynomial ramps the basis back in (`cutoff_func_poly`, line 220). The
mechanism exists so ACE can hand over to a core repulsion at short range -- but that core term is
`prehc * exp(-lambdahc r^2)` and FitSNAP's `AcePot` always writes **`prehc: 0`**, so it hands over
to nothing at all.

**The consequence, measured on real exported potentials** (W-W, `rcinner = 1.0`, `drcinner = 0.01`):

| r (A) | 0.80 | 0.95 | 0.99 | 1.00 | 1.05 | 2.40 |
|---|---|---|---|---|---|---|
| E (eV) | -4.53 | -4.53 | -4.53 | +679.86 | +515.37 | -6.95 |
| force | 0.0 | 0.0 | 0.0 | -3728 | -2884 | -4.3 |

Below 0.99 A the energy is exactly the sum of the per-element `E0` constants and the force is
IDENTICALLY ZERO, bounded from inside by a ~680 eV wall. A pair that crosses can never come back and
releases that energy as heat, which pushes further pairs over. Collapse verdicts clustering just
inside `rcinner` are this trap, and a post-hoc ZBL overlay does not rescue it (tested: restores a
3275 eV wall at 0.6 A, leaves r >= 2 A untouched, MD still collapsed) because the trap sits
underneath the overlay.

**`rcinner = 0` removes it and does not cost accuracy.** With no inner cutoff the same fit produces a
genuine repulsive wall (2234 eV at 0.80 A, force -11120 eV/A pushing apart) because a third of the
training configurations already contain pairs closer than 1.0 A for it to learn from. A CONTROLLED
A/B -- same 300 configurations, same labels, same hyperparameters, only `rcinner` differing -- gives
test RMSE 1.174 eV/atom and 3.009 eV/A at `rcinner = 0` versus 1.366 and 3.062 as shipped, and
identical descriptor magnitudes (max |a| = 3.73e3 both ways). FitSNAP's own default is `0.0`.
(An earlier note here claimed `rcinner = 0` made errors ~20x worse. That was WRONG: it compared two
different pipeline runs with different random structures and never isolated the parameter.)

**Validated at full scale.** All four example `FitSNAP.in` files now ship `rcinner = 0.0`, checked by
a 100k-configuration GRACE run (4 nodes, 1 h 50 m): 28 Pareto potentials, 28/28 MD-stable with drift
2.5e-9..1.9e-7 eV/atom/ps and the closest contact holding at 0.77-0.81 of its covalent bond length,
while accuracy held (best test 0.170 eV/atom / 0.588 eV/A vs 0.165 / 0.588 for the previous
`rcinner = 0.5/1.0` GRACE 100k run, and the front's MEDIAN energy RMSE improved 0.310 -> 0.194).
Note those two are separate runs with different random structures, so the controlled evidence remains
the same-data A/B above; this run establishes that nothing regresses at scale.

**What `rcinner = 0` does NOT fix** is a weak fit: with 300 configurations the potential still drove
an H-containing pair to 0.354 A. Short-range behaviour is only as good as the data, which is why
well-trained potentials are stable either way.


- **The test structure is the whole problem.** Every configuration the pipeline generates is entropy
  MAXIMIZED -- deliberately strange and 2-25 atoms. MD from a raw one runs away regardless of fit
  quality (measured: 300 K start -> 5200 K) and says nothing. The auto path takes the run's lowest
  FORMATION energy per atom (`_recon.formation_energy` removes composition by a per-element
  reference fit; lowest total or per-atom energy would just pick the smallest cell or the
  strongest-binding composition), replicates it past twice the cutoff in every direction and up to
  `min_atoms`, and relaxes it with the potential under test. Same run, same potentials: a raw
  entropy cell "failed" at 5700 K while the prepared structure holds 300 K with 1e-7 eV/atom/ps
  drift. A user-supplied `structure` is used AS GIVEN (no replication) -- their cell, their question.
- **Lowest formation energy is the SHORTLIST, not the criterion.** Two successive in-pipeline runs
  taught this. First: all four potentials reported unstable, because the picked structure's closest
  contact was 0.885 A -- inside ACE's inner cutoff, so `pair_pace.cpp` raised "Encountered very
  small distance" and MD never started. Adding a hard floor (`max(rcinner) + 0.1`) fixed that, but
  the next run picked a merely-legal candidate at 1.47 A and again failed 4/4 -- while a 2.39 A
  candidate from the SAME run passed 4/4. The verdict was tracking the structure, not the fits. So
  the stage now shortlists the 20 lowest formation energies, drops any that cannot be evaluated, and
  takes the LEAST COMPRESSED survivor, ranked by `d/(r_i + r_j)` with Pyykko covalent radii (the
  same mendeleev table `structuregen` samples) so that compositions compare fairly. Everything about
  that choice goes to `md/structure.txt`. If nothing is evaluable the stage RAISES and asks for a
  structure -- it never tests on a cell MD cannot start from.
- **Collapse is a RATIO, never an absolute distance.** `md_compression` is the closest pair's
  distance over the sum of its covalent radii (`ase.data`), and below `COLLAPSE_COMPRESSION = 0.7`
  the run is collapsed. Both cheaper criteria produced false verdicts on real runs: a fixed 0.5 A
  floor passed a run whose closest pair was 0.877 A with a drift 100x its stable siblings, and a
  floor derived from `rcinner` was worse in both directions -- it became 0.1 A when `rcinner = 0`
  (passing a run that ended at 0.58 A) and it condemned a perfectly ordinary H-H contact at 0.9 A,
  since H2 is 0.74 A. Any absolute number is wrong for a multi-element system: 1.8 A is a squeezed
  W-W contact and a normal H-W one.
- **`compression_after_minimize` separates two diagnoses**: a potential whose own 0 K minimum is a
  collapsed cell is broken outright; one that only fails once heated is unstable in dynamics.
- **`relax_box` defaults to 0** because that is the plain reading of "minimize the structure", not
  because it is broken: on well-trained potentials it is measured to work (6/6 stable, compression
  unchanged at 0.76-0.83). On under-trained ones it collapses the cell -- correctly, since for those
  the collapsed state really is the energy minimum.
- **The timestep default is 0.5 fs, not the usual 1 fs.** These screens run stiff, immature
  potentials on hydrogen-containing systems. Measured on the same potential and structure: 1 fs lost
  an atom outright (LAMMPS "Lost atoms"), while 0.2 fs integrated cleanly (drift 1e-5 eV/atom/ps) and
  turned the failure into an honest physical verdict. A "Lost atoms" note therefore means the
  INTEGRATION failed -- try a smaller timestep before blaming the fit.
- **`md_closest_pair` names the elements that collapsed** ("H-W at 0.354 A = 0.43x"). Which
  interaction failed is what tells you where the fit needs data; a bare distance does not.
- **What a real verdict looks like.** 5000-config HBeW potentials: 3/3 stable, compression 0.76-0.81
  after minimization and unchanged after 2000 steps. 40-config toy runs: LAMMPS aborts or atoms at
  0.30-0.39x bond length. The stage is separating good fits from bad ones, which is its whole job --
  do NOT read a failure as a bug in the stage without first checking `md_compression`,
  `md_closest_pair`, the timestep and the training-set size.
- **Drift is measured on an NVE tail, never under the thermostat.** Under `nvt` the total energy is
  the thermostat's to change, so drift across the NVT leg measures nothing; the runner switches to
  NVE for `steps/10` at the end and measures there.
- **Collapse detection uses a neighbour list, not an N^2 distance matrix** (test cells can be
  thousands of atoms), and `None` means "no pair within 3 A" -- not a NaN. An early version wrote
  `inf * False = nan` and reported every stable potential as broken.
- **An unstable potential is a RESULT, not an exception**: LAMMPS aborts ("Lost atoms") are caught
  and returned as `md_ok = 0` with the reason, since finding those is the point of the stage.
- **Task count is fixed at setup** (`max_potentials`), because how many potentials the export wrote
  is a future's result and must not size the submission; task `i` claims row `i` of `index.csv` and
  returns immediately if there is no row `i`. The merge task WARNS when more potentials were
  exported than tested, so a truncated screen cannot look complete.

## Uncertainty quantification (`potmill/uq/`, `[Main] uq`)

Every exported potential gets a `<name>.uq.npz` beside its `.yace`, plus a row in
`potentials/uq.csv` (joined into `index.csv`). Users reach it through an ASE calculator --
`PotMillCalculator(pot_dir).get_uncertainty(atoms)` alongside `get_forces()` -- or the CLI
(`python -m potmill.uq <run_dir> [--predict frames.xyz]`). The method is POPS (Swinburne et al.,
npj Comput Mater 2025): the spread of the per-configuration parameter corrections that would make
the model reproduce each training point exactly. It is the right family here because our error is
dominated by MISSPECIFICATION -- a linear ACE basis cannot represent the reference surface -- not by
label noise.

- **The estimator is reimplemented (`uq/pops.py`), not delegated to `popsregression`, for two
  reasons.** SCALE: the library materializes an `n x p` matrix of pointwise corrections and reloads
  the cumulative design matrix -- at 100k configurations and p = 1254 that is neither storable nor
  the O(N) pattern the incremental fit exists to preserve. Every POPS quantity is an accumulation of
  rank-1 terms, so it streams in O(p^2): `C = Sigma_s [sum_i w_i x_i x_i^T] Sigma_s`, another
  weighted Gram. ANCHORING: the library refits BayesianRidge internally, so its error bars describe
  ITS coefficients, not the ones we ship. Given the same anchor the two agree to 1e-10
  (`tests/test_uq.py::TestAgainstTheLibrary`, run whenever `popsregression` is installed).
- **The Bayesian half is free.** Evidence maximization needs only the eigenvalues of `X^T X`, plus
  `X^T y` and `y^T y` -- exactly what the fit's augmented R-factor already carries -- so it costs one
  p x p eigendecomposition and no data pass. It must iterate on `rho = lambda / alpha` rather than on
  the two hyperparameters separately: real ACE Gram matrices reach condition ~1e18, where forming
  `alpha * eigvals` OVERFLOWS float64 (observed on a real 100k fit). Regression-tested.
- **Energy rows only.** Adding the force rows costs 42x the data and makes per-structure ranking
  WORSE (Spearman 0.395 -> 0.375 against held-out error): force residuals live on a much larger
  numerical scale and take over the correction cloud, while the question asked is about a structure's
  ENERGY.
- **Rows carry the FIT's weights** (`w = exp(-E/5)` normalized to `eweight`, as in `fitting/fit.py`).
  POPS assumes its anchor minimizes the loss whose Hessian gives `Sigma_s`, and our beta minimizes
  the weighted loss; removing that mismatch raised ranking 0.395 -> 0.431 for free. Note the weight
  CANCELS out of the pointwise correction itself (`theta_i = r_i Sigma_s x_i / (x_i^T Sigma_s x_i)`
  is invariant to scaling row i), so what the weights change is the metric `Sigma_s` and which rows
  clear the residual threshold -- and sigma stays in eV/atom, directly comparable to the held-out
  errors.
- **The shipped object is the hypercube itself, not an ensemble sampled from it.** Sampling made the
  bracket's coverage partly a statement about the sample count (91.1% at 100 samples, 95.4% at 500,
  98.2% at 10 000), whereas maximizing a LINEAR functional over a box is analytic -- pick each
  component's favourable corner -- and is the honest worst case over the set. The standard deviation
  likewise comes from the box's exact second moment (`diag((high-low)^2/12) + m m^T`): measured
  0.61197 analytic vs 0.61312 from 500 samples, i.e. sampling bought no accuracy, only a
  seed-dependent answer. `POPSPosterior.sample(n)` still materializes a committee on demand, which is
  the paper's own use, so nothing is lost by not storing one. The box is NOT the cheaper option --
  measured at p = 1236, k = 1162: `projections` is 5.75 MB against 2.47 MB for a 500-member ensemble.
  That is the price of exactness, and it sits beside a `sigma_epi` of the same order (6.11 MB), so
  the artifact is ~11 MB either way.
- **`sigma(x)` uses an eigen-factor `F F^T = Sigma_total`, not the explicit quadratic form**: 140 s
  -> 0.5 s for 100k structures at p = 759, which is the difference between screening a trajectory and
  not. Eigen, not Cholesky -- `Sigma_miss` is only positive SEMI-definite.
- **Spearman ~0.43 against held-out error is NOT a bug, and it is near the attainable ceiling. Do not
  "fix" it.** Measured on the 100k GRACE run, same data throughout: two INDEPENDENT fits of the same
  training set agree with each other about which structures are hard only at rho = 0.741 -- that is
  the ceiling for any method here, since it is how much of "difficulty" is a property of the
  structure rather than of the particular fit. POPS reaches 0.428-0.431, i.e. ~58% of it. The
  alternatives measured on the same data are all worse: `popsregression` itself scores 0.275 (it
  anchors on its own refit), leverage/`gamma` alone 0.145, a 2-model committee 0.189. The error bar
  is a RANKING tool -- which structures to label next, which predictions to distrust -- not a
  per-structure error prediction.
- **`percentile_clipping` defaults to 0.0 (the library default) because clipping trades away exactly
  what the uncertainty is for.** Same potential, same 100k configurations, only the clipping percentile
  differing (mean held-out |error| 0.0622 eV/atom):

  | clip % | sigma (eV/atom) | q68 | Spearman | bracket half-width | bracket coverage | fit time |
  |---|---|---|---|---|---|---|
  | 0.0 | 0.0488 | 1.40 | **0.428** | 1.806 | 100.0% | 7.9 s |
  | 0.5 | 0.0160 | 4.29 | 0.280 | 0.438 | 100.0% | 15.0 s |
  | 1.0 | 0.0144 | 4.84 | 0.222 | 0.336 | 100.0% | 14.0 s |
  | 5.0 | 0.0124 | 5.82 | 0.102 | 0.150 | 95.1% | 13.6 s |

  Clipping does buy a far tighter bracket that still covered every held-out residual on this run, so
  it is a reasonable choice for someone who wants a usable band rather than a worst case -- but it
  degrades the ranking monotonically, and ranking is the primary product. It also costs ~2x the fit
  time, since a quantile box needs the projected corrections materialized while min/max is a running
  extremum.
- **Calibration is split-conformal against genuinely HELD-OUT predictions**: every configuration is
  predicted by the k-fold model that did not train on it (the same fixed `config_fold` partition the
  fit uses), and `calib_q68`/`calib_q95` are quantiles of `|error| / sigma` over those. Measured q68
  1.09-1.46 across the 32-potential GRACE front (1.40-1.82 on an earlier run), i.e. the raw POPS
  spread is 10-45% too small and covers only 51-64% of held-out errors before calibration -- which is
  why `get_uncertainty()` returns the CALIBRATED number by default (`level=None` gives the raw
  spread). It accepts one mild mismatch knowingly: the errors come from the FOLD models while sigma
  describes the ALL-DATA model that ships, which makes the factor slightly conservative. That is the
  right direction and the only honest option -- the shipped model has seen every configuration, so
  its own residuals cannot measure how it does on unseen ones.
- **The artifact is float32 and ~11 MB at p = 1236** (`sigma_epi` is p x p = 6.11 MB, `projections`
  p x k = 5.74 MB). `beta` is stored in float64 -- it must match the `.yace` exactly. The file also
  carries the run's `FitSNAP.in` verbatim, so a potential directory copied to another machine still
  knows how to featurize a new structure. `sigma_epi` is KEPT even though it is over half the file:
  the parameter term carries 1.7-15.9% of the VARIANCE across the measured fronts, i.e. dropping it
  would shrink sigma by 0.9-8.3% -- small at 100k configurations but not a rounding error, and it is
  the DOMINANT term at small training-set sizes, which the pipeline also runs at.
- **The ASE calculator runs two engines and cross-checks them ONCE.** Energies/forces come from
  LAMMPS with the exported `.mod` (0.27 s per structure, the same evaluator production MD uses),
  while an uncertainty needs the descriptor ROW, i.e. a FitSNAP featurization (0.39 s per structure).
  The first time both exist for the same structure it asserts `E_lammps == natoms * (x_E @ beta)` --
  measured agreement 6.6e-11 eV/atom and 2.2e-10 eV/A -- because a `.uq.npz` and a `.yace` that are
  not the same potential would otherwise attach an error bar to the wrong model, silently. It is a
  screening calculator, not an MD engine: it builds a LAMMPS instance per call.
- **Cost: 96 s per potential** at 100k configurations, p = 1236, 8 threads (dominated by reading ~1 GB
  of energy rows and three O(n p^2) passes). That is minutes of CPU after everything else finishes,
  so unlike the potential export the UQ stage DOES get a monitor Gantt row.
- **The UQ merge takes the MD merge as a dependency.** Both do a read-modify-write of
  `potentials/index.csv`, so running them concurrently would let one clobber the other's columns.
  Only the one-line merge waits; the per-potential fits overlap the MD screen freely. `uq.csv` is
  authoritative, and it UPDATES rather than replaces, so `--potential <name>` from the CLI does not
  drop the other rows.
- **Validated at full scale.** A 100k-configuration GRACE run (4 nodes, 1 h 47 m) exported 54 Pareto
  potentials and the stage produced 32 of them (`max_potentials`), 32/32 written with no failures,
  while the MD screen ran 32/32 stable on the same potentials. Spearman held at 0.419-0.433 across
  the whole front (0.428 measured standalone on the previous run's potential), sigma 0.047-0.072
  eV/atom against a held-out mean |error| of 0.062, and `uq_n_modes` tracked the basis size
  (298-1168). Across the front, `uq_sigma_mean` correlates with `test_e_rmse` at 0.70, i.e. the error
  bar also knows which POTENTIAL is better. Both merges wrote `index.csv` without clobbering each
  other -- it carries the full `md_*` and `uq_*` column sets -- and both printed the truncation
  WARNING, since the front (54) was larger than `max_potentials` (32).
- **`[Main] pops` is the LEGACY diagnostic and is unrelated** -- it runs `popsregression` per
  hyperparameter point, reloading the whole cumulative design matrix each time, and prints. It ships
  nothing and does not scale; `[Main] uq` is the one that produces usable uncertainties.

## Dependencies

- executorlib, flux-core, flux-sched (HPC scheduling)
- LAMMPS with MLIAP/SNAP support
- FitSNAP (featurization)
- JAX (entropy model gradients)
- ASE, ase-ga (atomic simulation)
- fairchem-core (UMA labeling) or VASP
- spglib, mendeleev (crystal/element utilities)
- POPSRegression (uncertainty quantification)
- numpy, scipy, pandas, scikit-learn

## History

See `CHANGELOG.md` for the development history (structuregen unification, entropy speedups,
the incremental R-collecting fit, batched UMA labeling, and the release cleanup/modularization).
