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
`--md-steps` runs a short NVE probe (energy drift). A real MD STABILITY stage -- a different
question entirely, a property of the fit rather than of the file -- is planned separately as
`[Main] md` / `[ourMD]`: lowest-formation-energy structure (reuse `_recon.formation_energy`),
replicate to a few hundred atoms, minimize, then NVT/NVE per potential as parallel tasks.

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
