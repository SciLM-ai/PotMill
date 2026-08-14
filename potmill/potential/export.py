"""Select fitted hyperparameter points and write them out as LAMMPS-ready potentials.

Layout produced under a run directory::

    potentials/
      index.csv                                   # every selected point + CV RMSEs, cost, status
      rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0/
          rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0.yace
          rcut_5.0__nmax_9_4__lmax_0_4__eweight_10.0.mod

The coefficients written are the ALL-DATA fit (see ``potmill.potential.betas``), while the errors
recorded in ``index.csv`` remain the honest k-fold cross-validated RMSEs from the Pareto results.

INTERRUPTED RUNS are supported and are the reason ``index.csv`` carries two configuration counts.
Each hyperparameter point's fit is an accumulator that eats one batch at a time and is overwritten
in place, while errors and the Pareto ranking are written only at SYNCHRONIZED checkpoints (a
``results_<b>.csv`` appears once every point has eaten batch ``b``).  Between checkpoints the
accumulators keep running, at different speeds -- a 1254-column basis takes longer per batch than a
174-column one -- so a run killed mid-batch leaves points that have eaten MORE data than the last
checkpoint's errors describe, by differing amounts.  Exporting then is fine (extra data only helps,
and the ranking is still apples-to-apples because every point was compared at the same checkpoint),
but it must not be silent: ``n_configs`` is what each potential's coefficients actually saw (read
from its own accumulator) and ``n_configs_errors`` is what the quoted RMSEs describe.  They are
equal for a run that finished.

Each potential is written independently: a failure raises for that point only, prints an
unmissable ERROR naming the point and the reason, is recorded in ``index.csv``, and the remaining
potentials are still written.
"""

import glob
import os
import re
import traceback

import numpy as np
import pandas as pd

from potmill.config import ConfigManager, load_fitsnap_config
from potmill.potential.ace import write_yace
from potmill.potential.betas import all_data_beta, all_data_beta_rows, state_n_configs
from potmill.potential.mod import write_mod
from potmill.tools import combined_ace_hyperparameters, rcuts_to_string

WHICH_CHOICES = ("none", "knee", "pareto", "all")


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"REQUIRED FILE MISSING (stop, do not guess): {path}")
    return path


def _numbered(columns, prefix):
    """Columns named ``<prefix><n>`` in NUMERIC order (rcut0, rcut1, ... nmax1, nmax2, ...)."""
    hits = [c for c in columns if re.fullmatch(prefix + r"\d+", c)]
    return sorted(hits, key=lambda c: int(c[len(prefix) :]))


def combo_from_row(row, columns):
    """A hyperparameter point from a Pareto-results row, keeping EVERY rcut/nmax/lmax column."""
    return {
        "rcuts": [float(row[c]) for c in _numbered(columns, "rcut")],
        "nmaxes": [int(row[c]) for c in _numbered(columns, "nmax")],
        "lmaxes": [int(row[c]) for c in _numbered(columns, "lmax")],
        "eweight": float(row["eweight"]),
    }


def combo_dirname(combo):
    """Readable, round-trippable directory name for a hyperparameter point."""
    return (
        "rcut_"
        + "_".join(str(float(r)) for r in combo["rcuts"])
        + "__nmax_"
        + "_".join(str(int(n)) for n in combo["nmaxes"])
        + "__lmax_"
        + "_".join(str(int(v)) for v in combo["lmaxes"])
        + "__eweight_"
        + str(float(combo["eweight"]))
    )


def final_batch(start_path):
    idx = [
        int(re.search(r"results_(\d+)\.csv", p).group(1))
        for p in glob.glob(start_path + "pareto-front/results_*.csv")
    ]
    if not idx:
        raise FileNotFoundError(
            f"No pareto-front/results_*.csv under {start_path} -- potentials need the Pareto "
            f"results to select and annotate hyperparameter points (stop)"
        )
    return max(idx)


def n_configs(start_path):
    """Number of labeled configurations behind the fit, from the cumulative b-file name."""
    bfiles = glob.glob(start_path + "features/b*.csv")
    sizes = [int(m.group(1)) for m in (re.search(r"b(\d+)\.csv", p) for p in bfiles) if m]
    return max(sizes) if sizes else -1


def configs_through_batch(start_path, batch):
    """Configurations covered by checkpoint ``batch`` -- i.e. what its errors describe.

    Counted from the per-batch b files (one energy row, ``local_index == 0``, per configuration)
    rather than assumed to be ``batch_size * (batch + 1)``: a batch can lose configurations to
    failed labeling, and combine_b writes only the survivors.
    """
    total = 0
    for bid in range(batch + 1):
        path = _need(f"{start_path}features/{bid}/b_batch.csv")
        local_idx = pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0].to_numpy()
        total += int((local_idx == 0).sum())
    return total


KNEE_COLUMNS = ("test_e_rmse_weighted", "test_f_rmse_weighted")


def select_rows(df, which):
    """Rows of the Pareto results to export, and the knee row's index (or None).

    The knee is chosen on the WEIGHTED errors, because that is the metric the stored
    ``pareto_front`` itself was computed on -- picking the knee on the unweighted errors would
    single out a point by a different criterion than the front it is being selected from, and would
    disagree with what ``plot_pareto``/``plot_errors`` draw. They coincide when the weighting does
    not change the ranking (every GRACE run measured), and differ when it does: on a 100k UMA run
    the weighted knee is rcut 5.5 / nmax 8,4 / lmax 0,4 while the unweighted one is 5 / 5,4 / 0,3.
    """
    from potmill.analysis._recon import select_knee

    knee_idx = None
    if len(df):
        missing = [c for c in KNEE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"pareto results are missing {missing}, so the knee cannot be selected on the same "
                f"metric as the Pareto front (stop, do not fall back to a different criterion)"
            )
        knee_idx = select_knee(df, *KNEE_COLUMNS).name
    if which == "none":
        return df.iloc[0:0], knee_idx
    if which == "all":
        return df, knee_idx
    if which == "pareto":
        return df[df["pareto_front"] == 1], knee_idx
    if which == "knee":
        return df.loc[[knee_idx]], knee_idx
    raise ValueError(f"[ourPotential] which = '{which}' must be one of {WHICH_CHOICES} (stop)")


def _subset_index(subsets, combo):
    """Index of this point's (rcut, nmax, lmax) subset in the swept grid -- the fit-state chain id."""
    hits = [
        s
        for s, (rcuts, nmaxes, lmaxes) in enumerate(subsets)
        if list(map(int, nmaxes)) == combo["nmaxes"]
        and list(map(int, lmaxes)) == combo["lmaxes"]
        and len(rcuts) == len(combo["rcuts"])
        and all(abs(float(a) - b) < 1e-9 for a, b in zip(rcuts, combo["rcuts"], strict=True))
    ]
    if len(hits) != 1:
        raise ValueError(
            f"hyperparameter point rcut={combo['rcuts']} nmax={combo['nmaxes']} "
            f"lmax={combo['lmaxes']} matches {len(hits)} swept subsets (expected exactly 1) -- "
            f"config.ini and the Pareto results disagree (stop, do not guess)"
        )
    return hits[0]


def _line_count(path):
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))


def _cumulative_b(start_path, n_rows):
    """The cumulative b file that is ROW-ALIGNED with a given number of design-matrix rows.

    The b files are named by configuration count (``b<N>.csv``), not by batch, and a batch can lose
    configurations to failed labeling -- so the alignment is established by matching row counts (the
    same pairing the fit itself relies on), never by assuming ``batch_size * (batch + 1)``.
    """
    candidates = []
    for path in glob.glob(start_path + "features/b*.csv"):
        m = re.search(r"b(\d+)\.csv$", path)
        if m:
            candidates.append((int(m.group(1)), path))
    for size, path in sorted(candidates):
        if _line_count(path) == n_rows:
            return path, size
    raise ValueError(
        f"no features/b<N>.csv has {n_rows} rows to match the loaded design matrix "
        f"(checked {len(candidates)} files) -- targets and descriptors would be misaligned (stop)"
    )


def _rows_engine_beta(start_path, combo, batch, feature_indices, eweight):
    """All-data coefficients for ``fit_engine = rows``: the row engine keeps no accumulated state,
    but it already reloads the cumulative design matrix, so the same estimator is computed here
    directly from it (identical weighting, train = every row)."""
    from potmill.bfile import read_b

    rcut_str = rcuts_to_string(combo["rcuts"], delimiter="_")
    parts = []
    for bid in range(batch + 1):
        a_map = np.load(_need(f"{start_path}features/{bid}/{rcut_str}/a.npy"), mmap_mode="r")
        parts.append(np.ascontiguousarray(a_map[:, feature_indices]))
    a_matrix = np.concatenate(parts)
    b_path, size = _cumulative_b(start_path, a_matrix.shape[0])
    local_idx, _, b_values = read_b(b_path)
    if a_matrix.shape[0] != len(b_values):
        raise ValueError(f"design matrix rows {a_matrix.shape[0]} != b rows {len(b_values)} (stop)")
    return all_data_beta_rows(a_matrix, b_values, local_idx == 0, eweight), size


def export_potentials(start_path, which="pareto", batch=None, feature_names=None, verbose=True):
    """Write LAMMPS potentials for the selected fitted hyperparameter points.

    ``feature_names`` is the descriptor-label list ``featurize`` returned for the full swept basis
    (the pipeline passes its featurization future's result). When omitted it is regenerated by
    featurizing one labeled structure from the run -- it is the ground truth every basis
    reconstruction is asserted against, so it is never simply assumed.
    """
    # Absolute: featurize() chdirs into its feature directory and never returns, so any relative
    # run path would stop resolving the moment descriptor labels are regenerated below.
    start_path = os.path.abspath(start_path) + "/"
    config = ConfigManager(_need(start_path + "config.ini"))
    fitsnap_name = config["FitSNAP"]["filename"]
    fitsnap_config = load_fitsnap_config(_need(start_path + fitsnap_name))
    mlip = config["FitSNAP"]["mlip"]
    if mlip != "ACE":
        raise NotImplementedError(
            f"[FitSNAP] mlip = {mlip}: the LAMMPS potential writer currently supports ACE only "
            f"(set [Main] potential = 0 to skip this stage) (stop)"
        )
    if which not in WHICH_CHOICES:
        raise ValueError(f"[ourPotential] which = '{which}' must be one of {WHICH_CHOICES} (stop)")
    if which == "none":
        return {"written": [], "failed": [], "out_dir": None}

    # The newest checkpoint on disk, which is NOT the same as "the run finished": an interrupted
    # run is exported from its last completed checkpoint. Whether each potential's coefficients
    # actually stop there is measured per potential below, not assumed.
    batch = final_batch(start_path) if batch is None else batch
    results_path = _need(start_path + f"pareto-front/results_{batch}.csv")
    df = pd.read_csv(results_path)
    rows, knee_idx = select_rows(df, which)

    hp = config["ourHyperparameters"]
    full_nmax = hp["max_nmax"] if isinstance(hp["max_nmax"], list) else [hp["max_nmax"]]
    full_lmax = hp["max_lmax"] if isinstance(hp["max_lmax"], list) else [hp["max_lmax"]]
    subsets = combined_ace_hyperparameters(config, w_eweight=False)
    ace_section = fitsnap_config["ACE"]
    reference_section = fitsnap_config.get("REFERENCE", {})
    elements = str(ace_section["type"]).split()
    fit_engine = config["ourFit"]["fit_engine"]
    nconf_errors = configs_through_batch(start_path, batch)

    if fit_engine == "incremental":
        # The subset index is positional in combined_ace_hyperparameters(config), so a config.ini
        # whose grid was edited after the run would map points onto the WRONG fit-state chain --
        # and a same-width mismatch (e.g. a different rcut) would not be caught downstream.
        on_disk = len(glob.glob(start_path + "fits/_state/subset_*"))
        if on_disk != len(subsets):
            raise ValueError(
                f"config.ini describes {len(subsets)} swept (rcut, nmax, lmax) subsets but the run "
                f"has {on_disk} fit-state chains on disk -- the grid in config.ini is not the grid "
                f"this run fitted, so subset indices cannot be trusted (stop, do not guess)"
            )

    if feature_names is None:
        from potmill.analysis._recon import feature_names as recon_feature_names
        from potmill.analysis._recon import load_run

        if verbose:
            print(
                "POTENTIAL: regenerating descriptor labels by featurizing one structure...",
                flush=True,
            )
        cwd = os.getcwd()
        try:
            feature_names = recon_feature_names(load_run(start_path))
        finally:
            os.chdir(cwd)  # featurize() chdirs into its feature directory and does not return

    out_dir = start_path + "potentials/"
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(
            f"POTENTIAL: writing {len(rows)} '{which}' potential(s) from batch {batch} "
            f"(all-data coefficients) -> {out_dir}",
            flush=True,
        )

    index_rows, written, failed = [], [], []
    seen_names = {}
    for row_idx, row in rows.iterrows():
        combo = combo_from_row(row, df.columns)
        name = combo_dirname(combo)
        record = {
            "dir": name,
            "rcut": rcuts_to_string(combo["rcuts"], delimiter=" "),
            "nmax": " ".join(str(n) for n in combo["nmaxes"]),
            "lmax": " ".join(str(v) for v in combo["lmaxes"]),
            "eweight": combo["eweight"],
            "train_e_rmse": row.get("train_e_rmse"),
            "train_f_rmse": row.get("train_f_rmse"),
            "test_e_rmse": row.get("test_e_rmse"),
            "test_f_rmse": row.get("test_f_rmse"),
            "cost": row.get("cost"),
            "pareto_front": int(row.get("pareto_front", 0)),
            "knee": int(row_idx == knee_idx),
            # n_configs: what THIS potential's coefficients were fitted on (from its own fit state).
            # n_configs_errors: what the RMSE/cost columns above describe (the checkpoint).
            # They differ only when a run was interrupted between checkpoints -- see the module docstring.
            "n_configs": None,
            "n_configs_errors": nconf_errors,
            "batch": batch,
            "status": "ok",
        }
        try:
            if name in seen_names:
                raise ValueError(
                    f"directory name '{name}' collides with hyperparameter point "
                    f"{seen_names[name]} (stop, do not overwrite)"
                )
            seen_names[name] = combo

            subset = _subset_index(subsets, combo)
            if fit_engine == "incremental":
                state_path = _need(f"{start_path}fits/_state/subset_{subset}/state.pt")
                beta = all_data_beta(state_path, combo["eweight"])
                record["n_configs"] = state_n_configs(state_path)
            else:
                from potmill.fitting.fit import _feature_indices

                indices = _feature_indices(
                    mlip, feature_names, [combo["rcuts"], combo["nmaxes"], combo["lmaxes"]]
                )
                beta, used = _rows_engine_beta(start_path, combo, batch, indices, combo["eweight"])
                record["n_configs"] = used

            pot_dir = out_dir + name
            os.makedirs(pot_dir, exist_ok=True)
            write_yace(
                f"{pot_dir}/{name}",
                ace_section,
                combo["rcuts"],
                full_nmax,
                full_lmax,
                combo["nmaxes"],
                combo["lmaxes"],
                beta,
                feature_names,
            )
            write_mod(
                f"{pot_dir}/{name}.mod",
                f"{name}.yace",
                elements,
                reference_section,
                meta=[
                    f"run: {start_path}  checkpoint (batch): {batch}",
                    f"rcut {record['rcut']} | nmax {record['nmax']} | lmax {record['lmax']} "
                    f"| eweight {record['eweight']}",
                    f"{len(beta)} coefficients fitted on all {record['n_configs']} configurations; "
                    f"{config['ourFit']['n_fold']}-fold CV test RMSE (over "
                    f"{record['n_configs_errors']} configurations): "
                    f"E {record['test_e_rmse']} / F {record['test_f_rmse']}",
                ],
                source=start_path + fitsnap_name,
            )
            written.append(
                {"dir": pot_dir, "name": name, "combo": combo, "beta": beta, "elements": elements}
            )
            if verbose:
                print(f"POTENTIAL: wrote {name} ({len(beta)} coefficients)", flush=True)
        except Exception as exc:  # noqa: BLE001 -- one bad point must not lose the others
            record["status"] = f"FAILED: {type(exc).__name__}: {exc}"
            failed.append((name, exc))
            print(
                f"ERROR: potential export FAILED for {name}:\n"
                f"  {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
        index_rows.append(record)

    pd.DataFrame(index_rows).to_csv(out_dir + "index.csv", index=False)
    # An interrupted run leaves accumulators ahead of the last checkpoint, by differing amounts.
    # That is fine (extra data only helps) but it must be stated, not left for the reader to notice.
    ahead = [r for r in index_rows if r["n_configs"] and r["n_configs"] != r["n_configs_errors"]]
    if ahead:
        counts = sorted({r["n_configs"] for r in ahead})
        print(
            f"NOTE: this run was interrupted between checkpoints -- {len(ahead)} of "
            f"{len(index_rows)} potential(s) were fitted on more configurations "
            f"({counts if len(counts) < 5 else f'{min(counts)}..{max(counts)}'}) than the batch-"
            f"{batch} errors describe ({nconf_errors}). The extra data only improves them and the "
            f"Pareto ranking is unaffected (every point was ranked at the same checkpoint); both "
            f"counts are in index.csv as n_configs / n_configs_errors.",
            flush=True,
        )
    # index.csv describes THIS export. Directories from an earlier, wider export are still on disk
    # and are no longer described by it -- say so rather than leave the user to notice.
    stale = sorted(
        os.path.basename(d.rstrip("/"))
        for d in glob.glob(out_dir + "*/")
        if os.path.basename(d.rstrip("/")) not in seen_names
    )
    if stale:
        print(
            f"WARNING: {len(stale)} potential director{'y' if len(stale) == 1 else 'ies'} in "
            f"{out_dir} are left over from an earlier export and are NOT in index.csv "
            f"(e.g. {stale[0]}). Delete them, or re-export with the selection that produced them.",
            flush=True,
        )
    if verbose:
        print(
            f"POTENTIAL: {len(written)} written, {len(failed)} failed -> {out_dir}index.csv",
            flush=True,
        )
    return {"written": written, "failed": failed, "out_dir": out_dir}


def export_potentials_task(
    start_path,
    which,
    feature_names,
    *dependencies,  # noqa: ARG001 -- futures only; executorlib resolves them before this runs
):
    """executorlib task wrapper for the in-pipeline ``[Main] potential`` stage.

    ``dependencies`` are the final pareto + fitting futures: they carry no data this needs, they
    exist so executorlib only starts the export once the Pareto results and the last fit-state
    checkpoint are on disk. Never raises -- the run's expensive artifacts are already safe, so a
    failure here is reported as an unmissable ERROR and returned, not turned into a dead future.

    The writer's own structural checks (regenerated labels vs featurization, coefficient counts,
    per-element blocks) run here as always -- those catch per-RUN problems. Re-deriving energies
    through LAMMPS is deliberately NOT part of the pipeline: it tests the writer and the installed
    FitSNAP/LAMMPS, which do not vary from run to run. That check lives in tests/test_potential.py
    and behind ``python -m potmill.potential <run> --verify N`` (see potmill/potential/verify.py).
    """
    try:
        result = export_potentials(start_path, which=which, feature_names=feature_names)
        return {
            "written": [entry["name"] for entry in result["written"]],
            "failed": [name for name, _ in result["failed"]],
            "out_dir": result["out_dir"],
        }
    except Exception as exc:  # noqa: BLE001 -- the pipeline must not die on the export stage
        print(
            f"ERROR: LAMMPS potential export FAILED ({type(exc).__name__}: {exc})\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        return {"written": [], "failed": [f"{type(exc).__name__}: {exc}"], "out_dir": None}
