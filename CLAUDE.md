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
6. **Uncertainty quantification** - POPSRegression for prediction intervals
7. **LAMMPS potential export** - writes the selected fits as ready-to-run LAMMPS potentials

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
- `[Main]`: pipeline stage toggles (`entropy`/`labeling`/`featurize`/`fit`/`pareto`/`pops`), `nconfigurations`, `batch_size`, and `device` = `cuda` | `cpu` (drives labeling + fitting placement)
- `[FitSNAP]`: MLIP type (ACE/SNAP), element specification, FitSNAP.in filename
- `[ourStructureGen]`: structure generation method and parameters (defaults resolved in `structuregen`), plus `entropy_jobs_per_node`, `entropy_cores_per_job`
- `[ourLabeling]`: `calculator` = `FAIRChemCalculator` | `Vasp` | `LAMMPS`, plus `label_batch_size`, `labeling_jobs_per_node`, `labeling_cores_per_job`
- `[ourFeaturization]`: `featurize_jobs_per_node`, `featurize_cores_per_job`
- `[ourFit]`: `fit_jobs_per_node`, `fit_cores_per_job`, `fit_method`, `n_fold`, `fit_engine`
- Per-stage layout is uniform: each stage has `<stage>_jobs_per_node` concurrent jobs of `<stage>_cores_per_job` cores. In `cuda` mode each labeling/fit job takes one GPU; in `cpu` mode each takes its cores and `worker_layout` checks the per-node sum leaves cores free for the dynamic executor.
- `[ourPotential]`: `which` = `none` | `knee` | `pareto` (default) | `all` (its only key -- the written `.mod` chooses its pace evaluator at runtime, see below)
- `[ourMD]`: MD screening of the written potentials -- `structure` (`auto` | path), `min_atoms`, `minimize`, `ensemble`, `temperature`, `timestep`, `steps`, `max_potentials`, `md_cores_per_job`
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
  HBeW grid. FUTURE (planned, own commit): have `foldfit` also write the merged all-data beta into
  its per-batch `betas_{pid}.bin` (fold `-1`) -- tiny, makes coefficients and errors align by
  construction for ANY batch, and costs ~3% more fit work at production batch size. It must NOT be
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
