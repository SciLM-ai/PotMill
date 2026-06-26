"""Post-hoc reconstruction layer for the analysis plots.

Everything here is PURE INFERENCE from the artifacts the pipeline already saved
(per-batch coefficients, design matrices, b-files, structures) -- no re-fitting.
The reconstruction is validated to match the pipeline's stored RMSE to ~1e-8.

NOTE (consistency rule): every required file is asserted via _need(); if anything
is missing we raise loudly rather than guessing or falling back to a different method.
"""

import glob
import os
import re
import tempfile
from collections import Counter

import numpy as np

from potmill.bfile import read_b
from potmill.config import ConfigManager, load_fitsnap_config
from potmill.fitting.fit import _feature_indices, config_fold
from potmill.tools import lmaxes_to_string, nmaxes_to_string, rcuts_to_string


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"REQUIRED FILE MISSING (stop, do not guess): {path}")
    return path


# --------------------------------------------------------------------------- config
def load_run(run_dir):
    run_dir = run_dir.rstrip("/") + "/"
    config = ConfigManager(_need(run_dir + "config.ini"))
    fitsnap_config = load_fitsnap_config(_need(run_dir + config["FitSNAP"]["filename"]))
    mlip = config["FitSNAP"]["mlip"]
    n_fold = int(config["ourFit"]["n_fold"])
    elements = fitsnap_config["ACE" if mlip == "ACE" else "BISPECTRUM"]["type"].split()
    return {
        "run_dir": run_dir,
        "config": config,
        "fitsnap_config": fitsnap_config,
        "mlip": mlip,
        "n_fold": n_fold,
        "elements": elements,
    }


def _one_structure(run_dir):
    from ase.io import read

    for pat in ("labeling/*/atoms_*.traj", "labeling/labeled_*.traj"):
        hits = sorted(glob.glob(run_dir + pat))
        if hits:
            a = read(_need(hits[0]), index=0)
            a.calc = (
                None  # featurize needs only geometry; a labeled calc lacks stress -> scraper errors
            )
            return a
    raise FileNotFoundError(
        f"No structure traj found under {run_dir}labeling/ (stop, do not guess)"
    )


def feature_names(run):
    """Descriptor labels (geometry/rcut-independent) via featurize on one structure."""
    from potmill.featurization.featurize import featurize

    cfg, fc, mlip = run["config"], run["fitsnap_config"], run["mlip"]
    hp = cfg["ourHyperparameters"]
    if mlip == "ACE":
        fc["ACE"]["nmax"] = nmaxes_to_string(hp["max_nmax"])
        fc["ACE"]["lmax"] = lmaxes_to_string(hp["max_lmax"])
    else:
        fc["BISPECTRUM"]["twojmax"] = str(hp["max_twojmax"][0])
    rcut0 = float(re.findall(r"[\d.]+", str(hp["min_rcut"]))[0])
    tmpd = tempfile.mkdtemp() + "/"
    return featurize([_one_structure(run["run_dir"])], cfg, fc, [rcut0], tmpd, only_cost=False)


# --------------------------------------------------------------------------- combos
def final_batch(run_dir):
    idx = [
        int(re.search(r"results_(\d+)\.csv", p).group(1))
        for p in glob.glob(run_dir + "pareto-front/results_*.csv")
    ]
    if not idx:
        raise FileNotFoundError(f"No pareto-front/results_*.csv under {run_dir} (stop)")
    return max(idx)


def combo_from_potname(potpath):
    m = re.search(
        r"rcut_([\d.]+)__nmax_([\d_]+)__lmax_([\d_]+)__eweight_([\d.]+)", os.path.basename(potpath)
    )
    return {
        "rcut": float(m.group(1)),
        "nmaxes": [int(x) for x in m.group(2).split("_")],
        "lmaxes": [int(x) for x in m.group(3).split("_")],
        "eweight": float(m.group(4)),
    }


def list_combos(run_dir, batch):
    out = []
    for cd in sorted(glob.glob(run_dir + f"fits/{batch}/*")):
        pots = sorted(glob.glob(cd + "/pot__*fold_0.csv"))
        if not pots:
            continue
        c = combo_from_potname(pots[0])
        c["dir"] = cd
        out.append(c)
    return out


def cumulative_b_path(run_dir):
    bfiles = glob.glob(run_dir + "features/b*.csv")
    return max(bfiles, key=lambda p: int(re.search(r"b(\d+)\.csv", p).group(1)))


def combo_from_row(row):
    """Build a combo dict from a pareto-front results row (handles any #rcut/#nmax/#lmax)."""
    rc = sorted(c for c in row.index if re.fullmatch(r"rcut\d+", c))
    nm = sorted(c for c in row.index if re.fullmatch(r"nmax\d+", c))
    lm = sorted(c for c in row.index if re.fullmatch(r"lmax\d+", c))
    return {
        "rcut": float(row[rc[0]]),
        "nmaxes": [int(row[c]) for c in nm],
        "lmaxes": [int(row[c]) for c in lm],
        "eweight": float(row["eweight"]),
    }


def load_beta(run, combo, batch, fold):
    """Saved fold coefficients. VASP: per-combo pot CSV. UMA: the binary betas container
    (reusing fit.read_beta). Raises if neither is present -- never guesses a format."""
    from potmill.tools import hyperparameters_to_string

    run_dir, mlip = run["run_dir"], run["mlip"]
    hp = (
        ([combo["rcut"]], combo["nmaxes"], combo["lmaxes"], combo["eweight"])
        if mlip == "ACE"
        else ([combo["rcut"]], combo["nmaxes"], combo["eweight"])
    )
    cs = hyperparameters_to_string(mlip, hp, delimiter="_")
    batch_dir = run_dir + f"fits/{batch}"
    pots = glob.glob(f"{batch_dir}/{cs}/pot__*fold_{fold}.csv")
    if pots:
        return np.loadtxt(pots[0])
    if glob.glob(f"{batch_dir}/betas_*.idx"):
        from potmill.fitting.fit import read_beta

        return np.asarray(read_beta(batch_dir, cs, fold), dtype=float)
    raise FileNotFoundError(f"No coefficients for combo '{cs}' fold {fold} in {batch_dir} (stop)")


# --------------------------------------------------------------- predictions from saved coefficients
def reconstruct_cv(run, combo, batch, fnames, b_path=None):
    """Cross-validated residuals for EVERY row: each config predicted by the fold model that held
    it out (A[:, idx] @ beta_fold). Streams the design matrix batch-by-batch -- it never holds the
    full cumulative A in memory (which is GB-to-tens at 100k scale and OOMs a login node)."""
    run_dir, n_fold, mlip = run["run_dir"], run["n_fold"], run["mlip"]
    b_path = b_path or cumulative_b_path(run_dir)
    hp_noe = (
        ([combo["rcut"]], combo["nmaxes"], combo["lmaxes"])
        if mlip == "ACE"
        else ([combo["rcut"]], combo["nmaxes"])
    )
    fidx = _feature_indices(mlip, fnames, list(hp_noe))
    rdir = rcuts_to_string([combo["rcut"]], delimiter="_")
    local_idx, job_id, b_values = read_b(b_path)
    is_energy = local_idx == 0
    part = np.array([config_fold(j, n_fold) for j in job_id])
    betas = [load_beta(run, combo, batch, fold) for fold in range(n_fold)]
    for beta in betas:
        if len(beta) != len(fidx):
            raise ValueError(f"beta len {len(beta)} != feature cols {len(fidx)} (stop)")
    pred = np.empty(len(b_values))
    row0 = 0
    for b in range(batch + 1):
        ab = np.load(_need(run_dir + f"features/{b}/{rdir}/a.npy"), mmap_mode="r")[:, fidx]
        idx = np.arange(row0, row0 + ab.shape[0])
        p = part[idx]
        for fold, beta in enumerate(betas):
            m = p == fold
            if m.any():
                pred[idx[m]] = np.ascontiguousarray(ab[m]) @ beta
        row0 += ab.shape[0]
    if row0 != len(b_values):
        raise ValueError(f"summed A rows {row0} != b rows {len(b_values)} (stop)")
    return {"resid": pred - b_values, "ref": b_values, "is_energy": is_energy, "job_id": job_id}


# --------------------------------------------------------------- formation energy
def _composition_by_jobid(run_dir, want, elements):
    """job_id -> (composition vector, n_atoms). VASP: per-config atoms_<id>.traj (filename).
    Per-worker labeled_*.traj: keyed on atoms.info['job_id']. RAISES if a needed job_id is absent
    or a frame lacks job_id -- never energy-matches or guesses."""
    from ase.io import read

    comp, nat = {}, {}
    sample = next(iter(want))
    if os.path.exists(run_dir + f"labeling/{sample}/atoms_{sample}.traj"):  # VASP: file per job_id
        for jid in want:
            c = Counter(
                read(
                    _need(run_dir + f"labeling/{jid}/atoms_{jid}.traj"), index=0
                ).get_chemical_symbols()
            )
            comp[jid] = [c.get(el, 0) for el in elements]
            nat[jid] = sum(c.values())
        return comp, nat
    trajs = sorted(
        glob.glob(run_dir + "labeling/labeled_*.traj")
    )  # per-worker: key on info['job_id']
    if not trajs:
        raise FileNotFoundError(
            f"No atoms_<id>.traj nor labeled_*.traj under {run_dir}labeling/ (stop)"
        )
    for f in trajs:
        for a in read(f, index=":"):
            if "job_id" not in a.info:
                raise ValueError(
                    f"labeled traj {f} has a frame without info['job_id'] -- rerun with the "
                    f"job_id-writing labeling backend; NOT falling back to energy-matching. (stop)"
                )
            jid = int(a.info["job_id"])
            if jid in want and jid not in comp:
                c = Counter(a.get_chemical_symbols())
                comp[jid] = [c.get(el, 0) for el in elements]
                nat[jid] = sum(c.values())
    missing = want - set(comp)
    if missing:
        raise ValueError(
            f"{len(missing)} b-file job_ids missing from labeled trajs, e.g. "
            f"{sorted(missing)[:5]} (stop, do not guess)"
        )
    return comp, nat


def formation_energy(run, b_path=None, cache=True):
    """Per-config ΔE_form/atom above the lowest sampled config (composition removed by
    a least-squares per-element reference fit). Returns dict keyed by job_id + refs."""
    run_dir, elements = run["run_dir"], run["elements"]
    b_path = b_path or cumulative_b_path(run_dir)
    cachef = run_dir + "features/_formation_energy.npz"
    if cache and os.path.exists(cachef):
        z = np.load(cachef, allow_pickle=True)
        return {
            "dE": z["dE"].item(),
            "refs": dict(zip(elements, z["refs"], strict=False)),
            "elements": elements,
        }

    local_idx, job_id, vals = read_b(b_path)
    e_rows = local_idx == 0
    jids = job_id[e_rows].astype(int)
    epa = vals[e_rows]  # energy per atom

    cmap, nmap = _composition_by_jobid(run_dir, set(jids.tolist()), elements)
    comp = np.array([cmap[j] for j in jids], dtype=float)
    nat = np.array([nmap[j] for j in jids], dtype=float)
    E_total = epa * nat
    refs, *_ = np.linalg.lstsq(comp, E_total, rcond=None)
    e_form = (E_total - comp @ refs) / nat
    dE = e_form - e_form.min()
    dE_map = {int(j): float(d) for j, d in zip(jids, dE, strict=False)}
    if cache:
        np.savez(cachef, dE=np.array(dE_map, dtype=object), refs=refs)
    return {"dE": dE_map, "refs": dict(zip(elements, refs, strict=False)), "elements": elements}


def dE_per_row(rec, dE_map):
    return np.array([dE_map[int(j)] for j in rec["job_id"]])


# --------------------------------------------------------------- metrics
def _rmse(r):
    return float(np.sqrt(np.mean(r**2))) if len(r) else np.nan


def _mae(r):
    return float(np.mean(np.abs(r))) if len(r) else np.nan


def windowed_metrics(rec, dE_row, windows):
    """For each cutoff W (configs with ΔE_form < W): RMSE & MAE of energy and force."""
    out = []
    e, f = rec["is_energy"], ~rec["is_energy"]
    r = rec["resid"]
    for W in windows:
        sel = dE_row < W
        out.append(
            {
                "W": W,
                "n_cfg": int((sel & e).sum()),
                "E_rmse": _rmse(r[sel & e]),
                "E_mae": _mae(r[sel & e]),
                "F_rmse": _rmse(r[sel & f]),
                "F_mae": _mae(r[sel & f]),
            }
        )
    return out


def boltzmann_metrics(rec, dE_row, dE_cfg, kTs):
    """Boltzmann-weighted (exp(-ΔE_form/kT)) RMSE & MAE + config-level n_eff, per kT."""
    out = []
    e, f = rec["is_energy"], ~rec["is_energy"]
    r = rec["resid"]
    for kT in kTs:
        w = np.exp(-dE_row / kT)
        we, wf = w[e], w[f]
        wc = np.exp(-dE_cfg / kT)
        n_eff = float(wc.sum() ** 2 / np.sum(wc**2))
        out.append(
            {
                "kT": kT,
                "n_eff": n_eff,
                "E_rmse": float(np.sqrt(np.sum(we * r[e] ** 2) / we.sum())),
                "E_mae": float(np.sum(we * np.abs(r[e])) / we.sum()),
                "F_rmse": float(np.sqrt(np.sum(wf * r[f] ** 2) / wf.sum())),
                "F_mae": float(np.sum(wf * np.abs(r[f])) / wf.sum()),
            }
        )
    return out


# --------------------------------------------------------------- pareto knee
def select_knee(df, e_col, f_col, cost_col="cost", accuracy_weight=2.0):
    """Knee on the Pareto front, favouring accuracy over cost: min weighted distance
    to the ideal corner after [0,1] normalisation. Returns the row (Series)."""
    front = df[df["pareto_front"] == 1].copy()
    if front.empty:
        front = df.copy()

    def norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0.0

    d = (
        accuracy_weight * norm(front[e_col]) ** 2
        + accuracy_weight * norm(front[f_col]) ** 2
        + norm(front[cost_col]) ** 2
    )
    return front.loc[d.idxmin()]
