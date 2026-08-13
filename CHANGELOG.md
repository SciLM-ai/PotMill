# Changelog

## Unreleased — LAMMPS potential export

- Added the `potmill.potential` stage (`[Main] potential`, `[ourPotential]`): the selected fits are
  written as ready-to-run LAMMPS potentials (`potentials/<point>/<point>.yace` + `.mod`, plus one
  `index.csv` with hyperparameters, k-fold CV RMSEs and cost). `which` = `none | knee | pareto | all`.
  The same code runs as a CLI for finished runs: `python -m potmill.potential <run_dir>`.
- Shipped coefficients are the ALL-DATA fit, recovered exactly from the per-fold R-factors already
  saved by `foldfit` (each row is in exactly `k-1` training sets, so QR-merging the k solve R's gives
  the full design matrix's R up to a scale that cancels in the LS solve) — no extra work during the
  run and no cumulative-design-matrix reload. Validated against a one-shot `lstsq` in
  `tests/test_potential.py`; `fit_engine = rows` computes the same estimator directly.
- Each written potential carries only its own (nmax, lmax) basis rather than the full swept basis
  padded with zeros, with coefficients attached by symbolic descriptor label and the label sets
  asserted equal per element first.
- Rewrote the `.yace` `E0:` line at full precision — `AcePot.write_pot` formats it with `'%f'`, which
  alone shifted energies by ~2e-7 eV/atom. LAMMPS then reproduces the fitted model to ~1e-13 eV/atom
  and ~1e-12 eV/A (verified on the HBeW 5000-config run, and per-basis in the test suite).
- The `.mod` picks its pace evaluator at runtime (`is_active(package,kokkos)` → `product`, else
  `recursive`). The two are identical numerically (<1e-15 eV/atom, ranks 1..4) but `recursive` is
  ~18% faster on CPU at a production basis while KOKKOS rejects it outright
  (`pair_pace_kokkos.cpp:570`), so hardcoding either would cost speed or abort every GPU run. GPU
  needs no different file — `-sf kk` selects `pace/kk`. A non-`zero` `[REFERENCE] pair_style` is
  passed through for `hybrid` and raises otherwise.
- Interrupted runs export from the latest completed checkpoint. Because each hyperparameter point's
  fit accumulates continuously while errors are only written at synchronised checkpoints, `index.csv`
  reports `n_configs` (what each potential's coefficients saw, read from its own fit state) next to
  `n_configs_errors` (what its RMSEs describe), with a NOTE when they differ; equal for a finished run.
- The LAMMPS energy/force cross-check is not part of the pipeline: it tests the writer and the
  installed FitSNAP/LAMMPS rather than the run. It stays in `tests/test_potential.py` and behind
  `python -m potmill.potential <run> --verify N` (worth running after upgrading either package).

## Unreleased — CPU + VASP full-pipeline path

- Added a `[Main] device = cuda | cpu` switch and a uniform per-stage layout scheme
  (`<stage>_jobs_per_node` + `<stage>_cores_per_job` for entropy/labeling/featurize/fit), replacing
  the GPU-only knobs (`fit_gpus_per_node`, `featurize_workers_per_node`, `n_entropy_workers`,
  `ncores_per_*`, `fit_device`). `resources.worker_layout` is now device-aware: cuda keeps the
  GPU-per-job behavior; cpu budgets cores per node and leaves cores free for the dynamic executor
  (combine_b/cost/pareto) so no stage stalls. Strict entropy auto-runs as a single serial worker.
- VASP labeling backend now applies the `vasp-ase-sp.py` single-point DFT settings as overridable
  defaults (encut 500, ismear 0, ediff 1e-6, kspacing 0.125, prec Accurate, ...), sets per-atom
  MAGMOMs for any element (unless `ispin = 1`), parses `setups` from a string, and rejoins a
  spaced `command`. The incremental R-collecting fit runs unchanged on CPU (`device = cpu`).
- Added the `examples/WBe/CPU_Vasp` example (Cray-MPICH `vasp_std_pm_cpu_01`, launched flux-natively
  with `flux run -n N -o cpu-affinity=per-task`; 4 VASP jobs/node x 24 cores; `m4884` CPU sbatch).
- Migrated the `HBeW`/`WRe` GPU examples and the unit tests to the new schema (GPU behavior unchanged).

## Unreleased — cleanup & modularization for release

- Removed the dead/broken `unary.py` entry point and the legacy `binary_entropy`/`multi_element_entropy`
  packages (recoverable from git history).
- Added conda-friendly packaging: `pyproject.toml` (hatchling), `LICENSE` (BSD-3), ruff/black/mypy +
  pre-commit config, a conda-forge `meta.yaml` scaffold, and a CI workflow.
- Added `potmill.config.ConfigManager` (centralized `DEFAULTS`, type coercion, unknown-key warnings,
  `validate()`, passthrough external-calculator sections) and renamed `inputfile` -> `config.ini`.
- Moved the labeling backends into `potmill.labeling` with a config-driven `make_labeling()` selector
  and `[ourLabeling]` + passthrough `[FAIRChemCalculator]` / `[Vasp]` / `[LAMMPS]` sections.
- Grouped pipeline stages into `potmill.featurization`, `potmill.fitting`, and `potmill.analysis`.
- Decomposed `__main__` (Flux/worker math -> `resources.py`; helpers, run-dir setup, progress reporting
  -> `pipeline.py`), leaving `main()` as the executor/submission/polling skeleton.
- Centralized the labeling b-file format in `potmill.bfile`; collapsed `fit()` to a single torch path
  with a configurable `fit_device` (cpu/cuda); deduplicated `tools.py` helpers and `_feature_indices`.
- Fixed ACE beta-coefficient filenames to use `lmax` (was `nmax`).
- Added a `unittest` suite (config, tools, b-file, labeling selection, resource layout, and a
  `fit`-vs-`foldfit` numerical equivalence test).

## Earlier milestones

- **Unified `structuregen/`** combining the binary and multi_element entropy methods behind one
  `config['method']` dispatch, with the executorlib `init_function` closure + worker-id injection pattern.
- **Entropy performance**: OMP threads set before LAMMPS/JAX import; reused LAMMPS calculators and JAX
  model state (`update_state`) across MC iterations.
- **Multi-element entropy speedup**: pure-Python `SoftRepulsionCalculator` + early distance check +
  skip-when-inactive, raising acceptance from ~0.4% to ~95% (math preserved; entropy still decreases).
- **Incremental R-collecting fit** (`fit_engine = incremental`): O(N) per-fold augmented-QR state,
  validated against the row-based engine to ~1e-9.
- **Featurize cutoff fix**: per-task `pair_style` override + `restart_limit=3` on block executors to
  avoid the LAMMPS `compute pace cutoff > pairwise cutoff` abort.
- **Batched UMA labeling** (`label_batch_size>1`): amortizes the fixed forward overhead so one labeling
  GPU/node keeps up with entropy generation.
