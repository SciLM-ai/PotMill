#!/usr/bin/env python
"""Pareto-front plots (accuracy vs accuracy vs cost) for a finished PotMill run.

Front definition matches the pipeline (`pareto.py`): a model is dominated only if
another is STRICTLY better in all three objectives (E-RMSE, F-RMSE, cost). Because
eweight variants of a descriptor combo share the same cost, this keeps tied-cost
variants on the front -- so the WEIGHTED front here reproduces the stored
`pareto_front` flag exactly. The UNWEIGHTED front is recomputed with the same
definition (the pipeline only stores the weighted one).

Outputs:
  pareto3d.pdf  - 3D scatter, weighted (pipeline metric) + unweighted physical
  pareto2d.pdf  - 2D projections (E-F, E-cost, F-cost), points coloured by the 3rd axis

Usage:
    python -m potmill.analysis.plot_pareto <run_dir> [-o out_dir] [--elev E] [--azim A]
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from potmill.analysis._recon import final_batch, select_knee  # noqa: E402


def pareto_mask(E, F, C):
    """Non-dominated set under the pipeline's definition: i is dominated iff some j is
    STRICTLY less in all three objectives (E, F, C)."""
    P = np.column_stack([np.asarray(E), np.asarray(F), np.asarray(C)])
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if np.any(np.all(P - P[i] < 0, axis=1)):
            keep[i] = False
    return keep


def _panel3d(ax, E, F, C, front, knee_idx, title, labels, elev, azim):
    ax.scatter(
        E[~front],
        F[~front],
        C[~front],
        s=8,
        c="0.7",
        alpha=0.5,
        depthshade=False,
        label="dominated",
    )
    ax.scatter(E[front], F[front], C[front], s=34, c="k", depthshade=False, label="Pareto front")
    ax.scatter(
        [E.iloc[knee_idx]],
        [F.iloc[knee_idx]],
        [C.iloc[knee_idx]],
        s=180,
        marker="*",
        c="red",
        edgecolor="k",
        linewidth=0.5,
        depthshade=False,
        zorder=10,
        label="chosen knee",
    )
    ax.set_xlabel("\n" + labels[0], fontsize=9)
    ax.set_ylabel("\n" + labels[1], fontsize=9)
    ax.set_zlabel("\n" + labels[2], fontsize=9)
    ax.set_title(f"{title}  ({int(front.sum())} on front)", fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")


def fig_3d(df, wfront, ufront, knee_idx, hp, out, elev, azim):
    fig = plt.figure(figsize=(13, 6))
    axw = fig.add_subplot(1, 2, 1, projection="3d")
    _panel3d(
        axw,
        df["test_e_rmse_weighted"],
        df["test_f_rmse_weighted"],
        df["cost"],
        wfront,
        knee_idx,
        "Weighted (pipeline metric)",
        ["E-RMSE (weighted)", "F-RMSE (weighted)", "cost (s)"],
        elev,
        azim,
    )
    axu = fig.add_subplot(1, 2, 2, projection="3d")
    _panel3d(
        axu,
        df["test_e_rmse"],
        df["test_f_rmse"],
        df["cost"],
        ufront,
        knee_idx,
        "Unweighted physical",
        ["E-RMSE (eV/atom)", "F-RMSE (eV/Å)", "cost (s)"],
        elev,
        azim,
    )
    fig.suptitle(
        f"PotMill Pareto front — {len(df)} models  |  knee: {hp}", fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


def fig_2d(df, front, knee_idx, hp, out):
    """2D projections of the UNWEIGHTED physical front; each panel colours by the 3rd axis."""
    panels = [
        ("test_e_rmse", "test_f_rmse", "cost", "E-RMSE (eV/atom)", "F-RMSE (eV/Å)", "cost (s)"),
        ("test_e_rmse", "cost", "test_f_rmse", "E-RMSE (eV/atom)", "cost (s)", "F-RMSE (eV/Å)"),
        ("test_f_rmse", "cost", "test_e_rmse", "F-RMSE (eV/Å)", "cost (s)", "E-RMSE (eV/atom)"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for k, (xc, yc, cc, xl, yl, cl) in enumerate(panels):
        sc = ax[k].scatter(df[xc], df[yc], c=df[cc], s=22, cmap="viridis", zorder=2)
        fr = front.copy()
        fr[knee_idx] = False  # don't circle the knee -- it's drawn as the star below
        ax[k].scatter(
            df[xc][fr],
            df[yc][fr],
            s=70,
            facecolors="none",
            edgecolors="k",
            linewidths=1.3,
            zorder=3,
            label="Pareto front",
        )
        kcol = sc.cmap(sc.norm(df[cc].iloc[knee_idx]))  # knee's own colorbar (3rd-axis) color
        ax[k].scatter(
            df[xc].iloc[knee_idx],
            df[yc].iloc[knee_idx],
            s=340,
            marker="*",
            color=kcol,
            edgecolor="k",
            linewidth=0.9,
            zorder=5,
            label="chosen knee",
        )
        cb = fig.colorbar(sc, ax=ax[k])
        cb.set_label(cl, fontsize=8)
        ax[k].set_xlabel(xl)
        ax[k].set_ylabel(yl)
        ax[k].grid(alpha=0.25, lw=0.5)
        if k == 0:
            ax[k].legend(fontsize=8)
    fig.suptitle(
        f"Pareto 2D projections (unweighted physical, {int(front.sum())} on front) — knee: {hp}",
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
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-52.0)
    args = ap.parse_args()
    run_dir = args.run_dir.rstrip("/") + "/"
    out_dir = (args.out_dir or run_dir).rstrip("/") + "/"
    os.makedirs(out_dir, exist_ok=True)

    batch = final_batch(run_dir)
    df = pd.read_csv(f"{run_dir}pareto-front/results_{batch}.csv")
    knee = select_knee(df, "test_e_rmse_weighted", "test_f_rmse_weighted")
    knee_idx = int(df.index.get_loc(knee.name))
    hp = (
        f"rcut={knee['rcut0']:g} nmax={int(knee['nmax1'])},{int(knee['nmax2'])} "
        f"lmax={int(knee['lmax1'])},{int(knee['lmax2'])} eweight={knee['eweight']:g}"
    )

    wfront = df["pareto_front"].values.astype(bool)  # weighted: the pipeline's own flag
    ufront = pareto_mask(df["test_e_rmse"], df["test_f_rmse"], df["cost"])  # unweighted, same defn
    # sanity: our recompute of the weighted front must reproduce the stored flag
    chk = pareto_mask(df["test_e_rmse_weighted"], df["test_f_rmse_weighted"], df["cost"])
    assert (chk == wfront).all(), "weighted front recompute != stored pareto_front (stop)"

    fig_3d(df, wfront, ufront, knee_idx, hp, out_dir + "pareto3d.pdf", args.elev, args.azim)
    fig_2d(df, ufront, knee_idx, hp, out_dir + "pareto2d.pdf")
    print(
        f"  weighted front={int(wfront.sum())} (==stored)  unweighted front={int(ufront.sum())}  knee: {hp}"
    )


if __name__ == "__main__":
    main()
