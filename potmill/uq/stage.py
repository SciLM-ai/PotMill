"""The ``[Main] uq`` stage: a POPS uncertainty model for every exported potential.

Same shape as the MD stage -- a FIXED number of tasks submitted upfront (the count cannot depend on
how many potentials the export wrote, since that is a future's result), each claiming one row of
``potentials/index.csv``, plus a merge that writes ``potentials/uq.csv`` and joins its columns into
``index.csv``.  Each task writes ``potentials/<name>/<name>.uq.npz`` next to the ``.yace``/``.mod``.
``uq.csv`` is authoritative and ``index.csv`` the convenience join -- and since the MD merge does the
same read-modify-write on ``index.csv``, the pipeline hands this merge the MD merge future as a
dependency so the two can never interleave. Only the one-line merge waits; the per-potential fits run
alongside the MD screen.

Three choices here were settled by measurement on the 100k GRACE run, not by argument:

* **Energy rows only.**  Including the force rows costs 42x the data and makes per-structure ranking
  WORSE (Spearman 0.395 -> 0.375), because force residuals live on a much larger numerical scale and
  dominate the correction cloud while the question being asked is about a structure's ENERGY.
* **In the fit's own weighted metric.**  POPS assumes the anchor minimizes the loss whose Hessian
  gives Sigma_s.  Our beta minimizes the fit's WEIGHTED loss (``w = exp(-E/5)``, normalized to
  ``eweight``), so the rows fed to POPS carry the same weights.  Removing that mismatch raised
  ranking from 0.395 to 0.431 for free.  Note the weights cancel out of the pointwise correction
  itself -- ``theta_i = r_i Sigma_s x_i / (x_i^T Sigma_s x_i)`` is unchanged by scaling row ``i`` --
  so what the weights actually change is the METRIC ``Sigma_s`` and which rows clear the residual
  threshold.  Predictions therefore stay in eV/atom, comparable to the held-out errors below.
* **Calibrated against genuinely held-out predictions.**  Every configuration is predicted by the
  k-fold model that did NOT train on it (the same fixed ``config_fold`` partition the fit uses), so
  ``calib_q68`` is a split-conformal factor measured on unseen data, not on the training residuals
  the POPS model was built from.  Note the mild mismatch this accepts: the errors come from the FOLD
  models (each trained on ``(k-1)/k`` of the data) while sigma describes the ALL-DATA model that is
  actually shipped.  That makes the calibration slightly CONSERVATIVE, which is the right direction,
  and it is the only honest option -- the shipped model has seen every configuration, so its own
  residuals cannot measure how it does on unseen ones.
"""

import os
import traceback

import numpy as np
import pandas as pd

from potmill.config import ConfigManager, load_fitsnap_config

UQ_COLUMNS = (
    "uq_sigma_mean",
    "uq_q68",
    "uq_q95",
    "uq_raw_coverage",
    "uq_spearman",
    "uq_epistemic_share",
    "uq_n_modes",
    "uq_note",
)


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"REQUIRED FILE MISSING (stop, do not guess): {path}")
    return path


def energy_rows(start_path, rcut_str, columns, batch):
    """Every configuration's energy row through checkpoint ``batch``: ``(rows, targets, job_ids)``.

    One row per configuration -- FitSNAP's ``local_index == 0`` row, whose target is the per-atom
    energy -- column-filtered to this potential's basis.  The force rows are skipped entirely (see
    the module docstring), which is also what keeps this a ~1 GB read at 100k configurations
    instead of a ~40 GB one.
    """
    rows, targets, jobs = [], [], []
    for bid in range(batch + 1):
        b_batch = pd.read_csv(_need(f"{start_path}features/{bid}/b_batch.csv"), header=None)
        b_batch = b_batch.to_numpy()
        is_energy = np.where(b_batch[:, 0] == 0)[0]
        matrix = np.load(_need(f"{start_path}features/{bid}/{rcut_str}/a.npy"), mmap_mode="r")
        if matrix.shape[0] != b_batch.shape[0]:
            raise ValueError(
                f"batch {bid}: design matrix has {matrix.shape[0]} rows but b_batch.csv has "
                f"{b_batch.shape[0]} -- descriptors and targets would be misaligned (stop)"
            )
        rows.append(np.ascontiguousarray(matrix[is_energy][:, columns]))
        targets.append(b_batch[is_energy, 2])
        jobs.append(b_batch[is_energy, 1])
    return (
        np.concatenate(rows),
        np.concatenate(targets).astype(float),
        np.concatenate(jobs).astype(int),
    )


def fit_weights(targets, eweight):
    """The fit's own energy-row weights: ``exp(-E/5)`` normalized to sum to ``eweight``.

    Mirrors ``fitting/fit.py`` (``eweights_train``) exactly; the normalization is over all rows here
    because these coefficients are the ALL-DATA fit, not one fold's.
    """
    weights = np.exp(-np.asarray(targets, float) / 5.0)
    return weights / weights.sum() * float(eweight)


class WeightedChunks:
    """Re-iterable weighted views of the resident rows -- POPS makes three passes over them.

    The weighting is applied per chunk rather than once over the whole matrix: at 100k x 1254 that
    would double a 1 GB array for no reason, and the chunk temporaries are ~50 MB.
    """

    def __init__(self, rows, targets, weights, n_chunks=20):
        self.rows, self.targets, self.weights = rows, targets, weights
        self.slices = np.array_split(np.arange(rows.shape[0]), max(1, n_chunks))

    def __iter__(self):
        for sel in self.slices:
            w = self.weights[sel]
            yield self.rows[sel] * w[:, None], self.targets[sel] * w


def _weighted_statistics(chunks):
    """``(X^T X, X^T y, y^T y)`` of the weighted energy rows -- all the Bayesian half of POPS needs."""
    p = chunks.rows.shape[1]
    gram, xty, yty = np.zeros((p, p)), np.zeros(p), 0.0
    for rows, targets in chunks:
        gram += rows.T @ rows
        xty += rows.T @ targets
        yty += float(targets @ targets)
    return gram, xty, yty


def uq_for_potential(rows, targets, jobs, eweight, beta, settings, fold_betas, n_fold):
    """POPS posterior + calibration for one potential. Returns ``(posterior, calibration, stats)``."""
    from scipy.stats import spearmanr

    from potmill.fitting.fit import config_fold
    from potmill.uq.artifact import calibrate
    from potmill.uq.pops import POPSPosterior, evidence_ridge_from_gram, fit_pops_streaming

    chunks = WeightedChunks(rows, targets, fit_weights(targets, eweight))
    gram, xty, yty = _weighted_statistics(chunks)
    _, sigma_epi, alpha, _ = evidence_ridge_from_gram(gram, xty, yty, rows.shape[0])
    posterior = fit_pops_streaming(
        chunks,
        beta,
        sigma_epi,
        alpha,
        posterior=str(settings["posterior"]),
        minimum_relative_error=float(settings["minimum_relative_error"]),
        percentile_clipping=float(settings["percentile_clipping"]),
    )

    # Calibration against genuinely unseen predictions: each configuration through the fold model
    # that did not train on it.
    sigma = posterior.std(rows)
    partition = np.array([config_fold(j, n_fold) for j in jobs])
    held_out = np.empty(rows.shape[0])
    for fold in range(n_fold):
        mask = partition == fold
        if not np.any(mask):
            raise ValueError(f"fold {fold} of {n_fold} has no configurations (stop)")
        held_out[mask] = rows[mask] @ fold_betas[fold]
    errors = np.abs(held_out - targets)
    calibration = calibrate(sigma, errors)

    # How much of the variance the parameter (Bayesian) term carries. POPS's premise is that
    # misspecification dominates; if this is not small the fit is noise-limited instead, which
    # changes how the error bar should be read -- so it is recorded rather than assumed.
    epi = POPSPosterior(
        posterior.sigma_epi, None, None, None, np.zeros_like(posterior.sigma_epi), "ensemble", 0, 0
    ).std(rows)
    usable = sigma > 0
    stats = {
        "uq_sigma_mean": float(sigma[usable].mean()),
        "uq_spearman": float(spearmanr(sigma, errors).statistic),
        "uq_epistemic_share": float(np.mean((epi[usable] / sigma[usable]) ** 2)),
        "uq_n_modes": int(0 if posterior.low is None else posterior.low.size),
        "held_out_mean_error": float(errors.mean()),
        "n_configs": int(rows.shape[0]),
    }
    return posterior, calibration, stats


def potential_context(start_path, name, config, batch):
    """Everything about one exported potential the UQ needs: its combo, columns, beta, fold betas."""
    from potmill.analysis._recon import feature_names as recon_feature_names
    from potmill.analysis._recon import load_run
    from potmill.fitting.fit import _feature_indices, read_beta
    from potmill.potential.betas import all_data_beta
    from potmill.potential.export import _subset_index, combo_dirname, combo_from_row
    from potmill.tools import (
        combined_ace_hyperparameters,
        hyperparameters_to_string,
        rcuts_to_string,
    )

    results = pd.read_csv(_need(f"{start_path}pareto-front/results_{batch}.csv"))
    combos = [combo_from_row(row, results.columns) for _, row in results.iterrows()]
    matching = [c for c in combos if combo_dirname(c) == name]
    if not matching:
        raise ValueError(
            f"potential '{name}' has no row in pareto-front/results_{batch}.csv, so its "
            f"hyperparameters cannot be recovered (stop, do not guess)"
        )
    combo = matching[0]

    cwd = os.getcwd()
    try:
        names = recon_feature_names(load_run(start_path))
    finally:
        os.chdir(cwd)  # featurize() chdirs into its feature directory and does not return
    columns = np.array(
        _feature_indices("ACE", names, [combo["rcuts"], combo["nmaxes"], combo["lmaxes"]])
    )

    subsets = combined_ace_hyperparameters(config, w_eweight=False)
    state = _need(f"{start_path}fits/_state/subset_{_subset_index(subsets, combo)}/state.pt")
    beta = all_data_beta(state, combo["eweight"])
    if len(beta) != len(columns):
        raise ValueError(
            f"potential '{name}': {len(beta)} coefficients but {len(columns)} selected descriptor "
            f"columns (stop, do not guess the alignment)"
        )
    combo_string = hyperparameters_to_string(
        "ACE", [combo["rcuts"], combo["nmaxes"], combo["lmaxes"], combo["eweight"]], delimiter="_"
    )
    n_fold = int(config["ourFit"]["n_fold"])
    fold_betas = [
        np.asarray(read_beta(f"{start_path}fits/{batch}", combo_string, f), float)
        for f in range(n_fold)
    ]
    return {
        "combo": combo,
        "columns": columns,
        "beta": beta,
        "fold_betas": fold_betas,
        "n_fold": n_fold,
        "rcut_str": rcuts_to_string(combo["rcuts"], delimiter="_"),
    }


def write_uq(start_path, name, settings, batch=None, verbose=True):
    """Fit and write ``potentials/<name>/<name>.uq.npz``. Returns the index record for ``name``."""
    from potmill.potential.export import final_batch
    from potmill.tools import rcuts_to_string
    from potmill.uq.artifact import save_uq

    start_path = os.path.abspath(start_path) + "/"
    config = ConfigManager(_need(start_path + "config.ini"))
    if config["FitSNAP"]["mlip"] != "ACE":
        raise NotImplementedError(
            f"[FitSNAP] mlip = {config['FitSNAP']['mlip']}: the UQ stage supports ACE only "
            f"(set [Main] uq = 0 to skip it) (stop)"
        )
    fitsnap_name = config["FitSNAP"]["filename"]
    fitsnap_config = load_fitsnap_config(_need(start_path + fitsnap_name))
    batch = final_batch(start_path) if batch is None else batch

    ctx = potential_context(start_path, name, config, batch)
    rows, targets, jobs = energy_rows(start_path, ctx["rcut_str"], ctx["columns"], batch)
    posterior, calibration, stats = uq_for_potential(
        rows,
        targets,
        jobs,
        ctx["combo"]["eweight"],
        ctx["beta"],
        settings,
        ctx["fold_betas"],
        ctx["n_fold"],
    )

    hp = config["ourHyperparameters"]
    full_nmax = hp["max_nmax"] if isinstance(hp["max_nmax"], list) else [hp["max_nmax"]]
    full_lmax = hp["max_lmax"] if isinstance(hp["max_lmax"], list) else [hp["max_lmax"]]
    provenance = {
        "run_dir": start_path,
        "batch": np.int64(batch),
        "rcut": rcuts_to_string(ctx["combo"]["rcuts"], delimiter=" "),
        "nmax": " ".join(str(v) for v in ctx["combo"]["nmaxes"]),
        "lmax": " ".join(str(v) for v in ctx["combo"]["lmaxes"]),
        "full_nmax": " ".join(str(v) for v in full_nmax),
        "full_lmax": " ".join(str(v) for v in full_lmax),
        "eweight": np.float64(ctx["combo"]["eweight"]),
        "rows": "energy rows only, weighted as the fit weights them",
        "minimum_relative_error": np.float64(settings["minimum_relative_error"]),
        "percentile_clipping": np.float64(settings["percentile_clipping"]),
        "n_configs": np.int64(stats["n_configs"]),
        "n_fold": np.int64(ctx["n_fold"]),
        "held_out_mean_error": np.float64(stats["held_out_mean_error"]),
        "epistemic_share": np.float64(stats["uq_epistemic_share"]),
        "spearman_vs_held_out": np.float64(stats["uq_spearman"]),
        # The calculator featurizes new structures itself, so the artifact carries the FitSNAP input
        # it must reproduce: a potential directory copied elsewhere stays usable on its own.
        "fitsnap_in": _fitsnap_text(start_path + fitsnap_name),
        "mlip": "ACE",
        "elements": str(fitsnap_config["ACE"]["type"]),
        "units": str(fitsnap_config.get("REFERENCE", {}).get("units", "metal")).strip(),
        "atom_style": str(fitsnap_config.get("REFERENCE", {}).get("atom_style", "atomic")).strip(),
    }
    pot_dir = _need(f"{start_path}potentials/{name}")
    path = f"{pot_dir}/{name}.uq.npz"
    save_uq(path, posterior, ctx["beta"], ctx["columns"], calibration, provenance)

    record = {"dir": name, "uq_note": ""}
    record |= {k: v for k, v in stats.items() if k.startswith("uq_")}
    record["uq_q68"] = float(calibration["calib_q68"])
    record["uq_q95"] = float(calibration["calib_q95"])
    record["uq_raw_coverage"] = float(calibration["raw_coverage"])
    if verbose:
        print(
            f"UQ {name}: sigma {stats['uq_sigma_mean']:.4f} eV/atom vs held-out |err| "
            f"{stats['held_out_mean_error']:.4f} eV/atom | q68 {record['uq_q68']:.2f} "
            f"| spearman {stats['uq_spearman']:.3f} | {stats['uq_n_modes']} modes "
            f"| epistemic {stats['uq_epistemic_share']:.1%} "
            f"| {os.path.getsize(path) / 1e6:.1f} MB -> {path}",
            flush=True,
        )
    return record


def _fitsnap_text(path):
    with open(path) as f:
        return f.read()


def uq_task(start_path, position, *dependencies, overrides=None):  # noqa: ARG001 -- futures only
    """Write the UQ artifact for row ``position`` of ``potentials/index.csv`` (or return None)."""
    try:
        start_path = os.path.abspath(start_path) + "/"
        index_path = start_path + "potentials/index.csv"
        if not os.path.exists(index_path):
            return None
        index = pd.read_csv(index_path)
        index = index[index["status"] == "ok"].reset_index(drop=True)
        if position >= len(index):
            return None  # fewer potentials than tasks -- the normal case, nothing to report
        name = str(index.iloc[position]["dir"])
        config = ConfigManager(_need(start_path + "config.ini"))
        settings = dict(config["ourUQ"])
        settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return write_uq(start_path, name, settings)
    except Exception as exc:  # noqa: BLE001 -- one potential must not lose the others
        print(
            f"ERROR: UQ task {position} FAILED ({type(exc).__name__}: {exc})\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        return {"dir": f"<task {position}>", "uq_note": f"{type(exc).__name__}: {exc}"}


def merge_uq_task(start_path, *records):
    """Collect the UQ records into ``potentials/uq.csv`` and join them into ``index.csv``.

    Records UPDATE the table rather than replace it, so running the CLI for a single potential does
    not drop the other rows -- while entries for potentials the current export no longer contains
    are dropped, since ``index.csv`` is what describes this export.
    """
    try:
        start_path = os.path.abspath(start_path) + "/"
        rows = [r for r in records if isinstance(r, dict)]
        out_dir = start_path + "potentials/"
        if not rows:
            print("UQ: no potentials were processed (nothing written).", flush=True)
            return None
        frame = pd.DataFrame(rows)
        uq_path = out_dir + "uq.csv"
        index_path = out_dir + "index.csv"
        exported = None
        if os.path.exists(index_path):
            index = pd.read_csv(index_path)
            exported = set(index[index["status"] == "ok"]["dir"].astype(str))
        if os.path.exists(uq_path):
            previous = pd.read_csv(uq_path)
            previous = previous[~previous["dir"].astype(str).isin(set(frame["dir"].astype(str)))]
            if exported is not None:
                previous = previous[previous["dir"].astype(str).isin(exported)]
            frame = pd.concat([previous, frame], ignore_index=True)
        frame.to_csv(uq_path, index=False)

        if exported is not None:
            index = index.drop(columns=[c for c in UQ_COLUMNS if c in index.columns])
            index.merge(frame, on="dir", how="left").to_csv(index_path, index=False)
            covered = int(frame["dir"].astype(str).isin(exported).sum())
            if len(exported) > covered:
                print(
                    f"WARNING: {len(exported)} potentials were exported but only {covered} carry "
                    f"an uncertainty model. In the pipeline this means [ourUQ] max_potentials caps "
                    f"the number of UQ tasks -- raise it to cover them all; from the CLI, drop "
                    f"--potential to do the rest.",
                    flush=True,
                )
        failed = [r["dir"] for r in rows if r.get("uq_note")]
        print(
            f"UQ: {len(rows) - len(failed)}/{len(rows)} potentials processed this time "
            f"({len(frame)} in total) -> {uq_path}"
            + (f"; FAILED: {', '.join(failed)}" if failed else ""),
            flush=True,
        )
        return uq_path
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: UQ merge FAILED ({type(exc).__name__}: {exc})\n{traceback.format_exc()}",
            flush=True,
        )
        return None
