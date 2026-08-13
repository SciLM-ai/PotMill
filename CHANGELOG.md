# Changelog

## Unreleased — MD stability screening

- `rcinner = 0.0` in all four example `FitSNAP.in` files (HBeW/ACE, HBeW/GRACE, WRe/ACE,
  WBe/CPU_Vasp), matching FitSNAP's own default. **Validated by a full 100k-configuration GRACE
  pipeline run** (4 nodes, 1 h 50 m): 28 Pareto potentials exported, **28/28 stable in MD** with
  energy drift 2.5e-9 to 1.9e-7 eV/atom/ps, final temperatures 260–324 K and the closest contact
  holding at 0.77–0.81 of its covalent bond length. Accuracy did not regress — best test RMSE
  0.170 eV/atom and 0.588 eV/Å versus 0.165 and 0.588 for the previous `rcinner = 0.5/1.0` GRACE
  100k run, with the Pareto front's *median* energy RMSE improving from 0.310 to 0.194.
- `[Main] md` appears in `pipeline_monitor.csv` and as the last row of the monitor Gantt (it runs for
  minutes on CPU with the GPUs idle). The potential export is deliberately NOT tracked: measured at
  36 s for 28 potentials across 15 subsets at production scale, it is over before a bar would be
  legible.

- Added the `potmill.md` stage (`[Main] md`, `[ourMD]`): every exported potential gets a short MD
  trajectory of its own, in parallel, and the outcome (`md_ok`, NVE energy drift, final temperature,
  closest approach, and the reason for any failure) lands in `potentials/md.csv` and is joined into
  `index.csv`. Also runs standalone: `python -m potmill.md <run_dir>`.
- The test structure defaults to the **least compressed** of the run's 20 lowest formation-energy
  configurations (composition removed by a per-element reference fit), replicated past twice the
  potential cutoff and up to `min_atoms`, then relaxed with the potential under test. Every
  configuration PotMill generates is entropy-*maximized*, and each part of that rule was forced by a
  real run: MD from a raw configuration runs away regardless of fit quality (300 K start reaching
  5700 K); the lowest-energy one routinely has a contact inside ACE's inner cutoff (0.885 Å), so
  `pair_pace` refuses to evaluate it and every potential looks unstable; and picking on energy alone
  made the same four potentials come back 4/4 stable from one candidate and 0/4 from another.
  Compression is measured as `d/(r_i + r_j)` with Pyykkö covalent radii so compositions compare
  fairly, and the full choice is written to `md/structure.txt`. A user-supplied `structure` is used
  exactly as given.
- Collapse is judged as `md_compression` — the closest pair over the sum of its covalent radii —
  below 0.7, never as an absolute distance: no single distance works across elements (1.8 Å is a
  squeezed W–W contact and a normal H–W one), and both cheaper criteria gave false verdicts on real
  runs (a 0.5 Å floor passed a run whose closest pair was 0.877 Å; an `rcinner`-derived floor became
  0.1 Å when `rcinner = 0` and separately condemned an ordinary 0.9 Å H–H contact). The stage also
  reports `compression_after_minimize`, which distinguishes a potential whose 0 K minimum is already
  collapsed from one that only fails when heated.
- With that criterion the verdicts are physical: the 5000-configuration HBeW potentials are stable
  (3/3, compression 0.76–0.81 unchanged through 2000 steps, and 6/6 including with `relax_box = 1`),
  while 40-configuration toy runs collapse. `relax_box` defaults to 0 — it is measured to work on
  well-trained potentials, so the default is simply the plain reading of "minimize the structure".
- `timestep` defaults to 0.5 fs rather than 1 fs, and `md_closest_pair` records which elements
  collapsed. Measured: at 1 fs LAMMPS lost an atom outright on a stiff potential, while 0.2 fs on the
  same potential and structure integrated cleanly (drift 1e-5 eV/atom/ps) — so a "Lost atoms" result
  is an integration failure to retry smaller, not a verdict on the fit.
- Documented a property of every FitSNAP-generated ACE potential found while validating this stage:
  `rcinner` zeroes **all** radial basis functions below the inner cutoff
  (`ace_radial.cpp: gr.fill(0)`), and FitSNAP always writes `prehc: 0`, so there is nothing
  underneath — measured on a real potential, W–W energy is a flat −4.53 eV with **exactly zero
  force** below 0.99 Å behind a ~680 eV wall, i.e. an atom trap. A controlled A/B (same 300
  configurations, same labels, only `rcinner` differing) shows `rcinner = 0` both removes the trap
  (a genuine 2234 eV repulsive wall at 0.80 Å) and slightly *improves* errors (1.174 vs 1.366
  eV/atom), with FitSNAP's own default being 0. An earlier CHANGELOG entry claiming `rcinner = 0`
  cost ~20× the energy RMSE was wrong — that comparison was between two different pipeline runs with
  different random structures and never isolated the parameter.
- Drift is measured on an NVE tail rather than across the NVT leg (where the thermostat, not the
  potential, sets the total energy); collapse is detected with a neighbour list; an unstable
  potential is reported as a result (`md_ok = 0` with a reason), never raised.

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
