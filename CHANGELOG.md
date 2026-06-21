# Changelog

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
