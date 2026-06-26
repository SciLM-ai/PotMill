#!/usr/bin/env python
"""Error/accuracy plots for a finished PotMill run (post-hoc; pure inference from
saved coefficients -- validated to match the pipeline RMSE to ~1e-8).

Produces four figures:
  convergence.pdf   - per-combo learning curves (spaghetti), chosen knee highlighted
  parity.pdf        - knee model: predicted vs reference E & F, coloured by ΔE_form
  error_window.pdf  - knee model: RMSE & MAE vs energy window (ΔE_form above min)
  error_kT.pdf      - knee model: Boltzmann-weighted RMSE & MAE vs kT, with n_eff

ΔE_form is the per-config formation energy above the LOWEST SAMPLED configuration
(composition removed by a per-element reference fit) -- NOT above the true ground
state, since the entropy-max set does not sample near equilibrium.

Usage:
    python -m potmill.analysis.plot_errors <run_dir> [-o out_dir]
"""

import argparse
import glob
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from potmill.analysis import _recon as R  # noqa: E402

EC, FC = "#0072B2", "#D55E00"  # energy / force colors


# --------------------------------------------------------------- 1) convergence
def convergence(run, batch, knee, out):
    rd = run["run_dir"]
    bs = int(run["config"]["Main"]["batch_size"])
    files = sorted(
        glob.glob(rd + "pareto-front/results_*.csv"),
        key=lambda p: int(re.search(r"results_(\d+)", p).group(1)),
    )
    cols = list(pd.read_csv(files[0]).columns)
    hp_cols = cols[: cols.index("train_e_rmse")]
    series = {}
    for f in files:
        i = int(re.search(r"results_(\d+)", f).group(1))
        d = pd.read_csv(f)
        x = (i + 1) * bs
        for _, row in d.iterrows():
            key = tuple(round(float(row[c]), 4) for c in hp_cols)
            series.setdefault(key, []).append((x, row["test_e_rmse"], row["test_f_rmse"]))
    knee_key = tuple(round(float(knee[c]), 4) for c in hp_cols)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for key, pts in series.items():
        if key == knee_key:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ax[0].plot(xs, [p[1] for p in pts], color="0.4", alpha=0.12, lw=0.5, rasterized=True)
        ax[1].plot(xs, [p[2] for p in pts], color="0.4", alpha=0.12, lw=0.5, rasterized=True)
    kp = sorted(series[knee_key])
    xs = [p[0] for p in kp]
    ax[0].plot(xs, [p[1] for p in kp], color="red", lw=2.0, label="chosen knee", zorder=5)
    ax[1].plot(xs, [p[2] for p in kp], color="red", lw=2.0, zorder=5)
    ax[0].set_ylabel("Energy test RMSE (eV/atom)")
    ax[1].set_ylabel("Force test RMSE (eV/Å)")
    ax[1].set_xlabel("Number of structures")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.tick_params(labelsize=9)
    ax[0].legend(fontsize=9)
    ax[0].set_title(f"Convergence — {len(series)} models (faint), chosen knee in red", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}  ({len(series)} combos)")


# --------------------------------------------------------------- 2) parity
def _ann(ax, r):
    e = "  ".join([f"RMSE {R._rmse(r):.4g}", f"MAE {R._mae(r):.4g}"])
    ax.text(
        0.04,
        0.92,
        e,
        transform=ax.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8},
    )


def parity(rec, dE_row, out, hp):
    e, f = rec["is_energy"], ~rec["is_energy"]
    pred = rec["ref"] + rec["resid"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for k, (m, lab, unit) in enumerate([(e, "Energy", "eV/atom"), (f, "Force", "eV/Å")]):
        x, y, c = rec["ref"][m], pred[m], dE_row[m]
        hb = ax[k].hexbin(
            x, y, C=c, reduce_C_function=np.mean, gridsize=180, cmap="viridis", mincnt=1
        )
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax[k].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
        ax[k].set_xlabel(f"Reference {lab} ({unit})")
        ax[k].set_ylabel(f"Predicted {lab} ({unit})")
        ax[k].set_title(f"{lab} parity")
        _ann(ax[k], rec["resid"][m])
        cb = fig.colorbar(hb, ax=ax[k])
        cb.set_label("ΔE_form above min (eV/atom)", fontsize=8)
    fig.suptitle(f"Parity — chosen knee: {hp}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


# --------------------------------------------------------------- 3) error vs window
def error_window(wm, out, hp, brk=6.0):
    """Error vs energy window on a BROKEN x-axis: the informative low-energy region (0.3 -> brk) is
    drawn at full width, then an axis break (//), then the long high-energy tail (brk -> max) is
    compressed into a narrow panel -- so the endpoint is still shown without a giant empty gap
    squashing the bulk. A grey secondary axis shows the config count per window (trust where > 100)."""
    W = np.array([m["W"] for m in wm])
    ncfg = np.array([m["n_cfg"] for m in wm])
    fig = plt.figure(figsize=(13, 5))
    subfigs = fig.subfigures(1, 2, wspace=0.12)
    specs = [("E_rmse", "E_mae", "Energy", "eV/atom", EC), ("F_rmse", "F_mae", "Force", "eV/Å", FC)]
    for sf, (rk, mk, lab, unit, col) in zip(subfigs, specs, strict=True):
        axm, axt = sf.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.06}
        )
        rmse = np.array([m[rk] for m in wm])
        mae = np.array([m[mk] for m in wm])
        lo, hi = brk >= W, brk <= W  # brk is exactly a sampled window, so it joins both panels
        for ax, msk in ((axm, lo), (axt, hi)):
            ax.plot(W[msk], rmse[msk], "-o", ms=3, color=col, label="RMSE")
            ax.plot(W[msk], mae[msk], "--s", ms=3, color=col, alpha=0.6, label="MAE")
            ax.grid(alpha=0.25, lw=0.5)
        axm.set_xlim(0.3, brk)
        axt.set_xlim(brk, float(W.max()))
        axt.set_xticks(np.round(np.linspace(brk, float(W.max()), 3)).astype(int))
        # config count (grey) on a secondary y-axis spanning BOTH panels -- flags how many structures
        # fall in each window (trust where > 100). The count ticks live on the far-right (tail) spine,
        # since the main panel's right spine is removed for the axis break.
        a2m, a2t = axm.twinx(), axt.twinx()
        a2m.plot(W[lo], ncfg[lo], color="0.5", lw=1, alpha=0.7)
        a2t.plot(W[hi], ncfg[hi], color="0.5", lw=1, alpha=0.7)
        a2t.axhline(100, color="0.7", ls="--", lw=0.7)
        for a2 in (a2m, a2t):
            a2.set_ylim(0, float(ncfg.max()) * 1.05)
        a2m.set_yticks(
            []
        )  # hide the count ticks at the break; show them on the far-right spine only
        a2t.set_ylabel("configs in window", color="0.5", fontsize=8)
        a2t.tick_params(labelsize=7, colors="0.5")
        # hide the error + count spines at the break, then draw the diagonal break marks
        for a in (axm, a2m):
            a.spines["right"].set_visible(False)
        for a in (axt, a2t):
            a.spines["left"].set_visible(False)
        axt.tick_params(left=False)
        bm = {
            "marker": [(-1, -0.5), (1, 0.5)],
            "markersize": 8,
            "linestyle": "none",
            "color": "k",
            "mec": "k",
            "mew": 1,
            "clip_on": False,
        }
        a2m.plot([1, 1], [0, 1], transform=a2m.transAxes, **bm)  # on the twins (top) so they show
        a2t.plot([0, 0], [0, 1], transform=a2t.transAxes, **bm)
        axm.set_ylabel(f"{lab} error ({unit})")  # y-label identifies the metric (Energy vs Force)
        axm.legend(fontsize=8, loc="upper right")
        sf.supxlabel("Energy window ΔE_form above min (eV/atom)", fontsize=9)
    fig.suptitle(
        f"Error vs energy window  (tail beyond {brk:g} eV/atom compressed) — knee: {hp}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


# --------------------------------------------------------------- 4) error vs kT
def error_kT(bm, out, hp):
    kT = np.array([m["kT"] for m in bm])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for k, (rk, mk, lab, unit, col) in enumerate(
        [("E_rmse", "E_mae", "Energy", "eV/atom", EC), ("F_rmse", "F_mae", "Force", "eV/Å", FC)]
    ):
        ax[k].plot(kT, [m[rk] for m in bm], "-o", ms=3, color=col, label="RMSE")
        ax[k].plot(kT, [m[mk] for m in bm], "--s", ms=3, color=col, alpha=0.6, label="MAE")
        ax[k].axvline(5.0, color="green", ls=":", lw=1, label="current eweight scale (≈5)")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("kT (eV/atom)  [low=near-ground, high=uniform]")
        ax[k].set_ylabel(f"{lab} weighted error ({unit})")
        ax[k].set_title(f"{lab}: Boltzmann-weighted error vs kT")
        ax[k].grid(alpha=0.25, lw=0.5)
        a2 = ax[k].twinx()
        a2.plot(kT, [m["n_eff"] for m in bm], color="0.5", lw=1, alpha=0.7)
        a2.set_ylabel("n_eff (configs)", color="0.5", fontsize=8)
        a2.axhline(50, color="0.7", ls="--", lw=0.7)
        ax[k].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Boltzmann-kT error (n_eff in grey; trust where n_eff>50) — knee: {hp}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out_dir", default=None)
    args = ap.parse_args()
    run_dir = args.run_dir.rstrip("/") + "/"
    out_dir = (args.out_dir or run_dir).rstrip("/") + "/"
    os.makedirs(out_dir, exist_ok=True)

    run = R.load_run(run_dir)
    batch = R.final_batch(run_dir)
    df = pd.read_csv(f"{run_dir}pareto-front/results_{batch}.csv")
    knee = R.select_knee(df, "test_e_rmse_weighted", "test_f_rmse_weighted")
    hp = (
        f"rcut={knee['rcut0']:g} nmax={int(knee['nmax1'])},{int(knee['nmax2'])} "
        f"lmax={int(knee['lmax1'])},{int(knee['lmax2'])} eweight={knee['eweight']:g}"
    )
    print(f"chosen knee: {hp}")

    convergence(run, batch, knee, out_dir + "convergence.pdf")  # stored data, no recon

    print("regenerating feature_names ...")
    fnames = R.feature_names(run)
    combo = R.combo_from_row(knee)
    print("reconstructing knee predictions ...")
    rec = R.reconstruct_cv(run, combo, batch, fnames)
    print("computing formation energy (reads per-config trajs; cached after) ...")
    fe = R.formation_energy(run)
    dE_row = R.dE_per_row(rec, fe["dE"])
    dE_cfg = dE_row[rec["is_energy"]]
    print(
        "  per-element refs (eV/atom): " + ", ".join(f"{k}={v:.2f}" for k, v in fe["refs"].items())
    )

    parity(rec, dE_row, out_dir + "parity.pdf", hp)

    # Broken x-axis: sample the informative bulk densely (0.3 -> brk, ~99% of configs) and the long
    # high-energy tail sparsely (brk -> max); error_window draws the bulk at full width and the tail
    # compressed past an axis break, so the endpoint shows without the outlier stretching the range.
    brk = 6.0
    mx = float(dE_row.max())
    windows = sorted(
        set(np.round(np.linspace(0.3, brk, 30), 3))
        | {0.5, 1.0, 2.0}
        | set(np.round(np.linspace(brk, mx, 8), 3))
    )
    wm = R.windowed_metrics(rec, dE_row, windows)
    error_window(wm, out_dir + "error_window.pdf", hp, brk=brk)

    kTs = np.logspace(np.log10(0.1), np.log10(10.0), 30)
    bm = R.boltzmann_metrics(rec, dE_row, dE_cfg, kTs)
    error_kT(bm, out_dir + "error_kT.pdf", hp)


if __name__ == "__main__":
    main()
