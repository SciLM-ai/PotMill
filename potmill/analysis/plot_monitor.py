#!/usr/bin/env python
"""Post-processing plot for PotMill pipeline monitoring data.

Generates a 3-panel figure with shared x-axis (elapsed time):
  Panel 1: Mean GPU utilization (%)
  Panel 2: Mean CPU utilization (per-physical-core, %; SMT siblings summed)
  Panel 3: Gantt chart of pipeline stage activity

Usage:
    python -m potmill.analysis.plot_monitor [path/to/pipeline_monitor.csv] [-o output.pdf]
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

STAGE_COLORS = {
    "entropy": "#0072B2",
    "labeling": "#D55E00",
    "b_collecting": "#CC79A7",
    "featurization": "#009E73",
    "fitting": "#F0E442",
    "cost": "#56B4E9",
    "pareto": "#E69F00",
    "pops": "#999999",
}

STAGE_LABELS = {
    "entropy": "Entropy",
    "labeling": "Labeling",
    "b_collecting": "B Collecting",
    "featurization": "Featurization",
    "fitting": "Fitting",
    "cost": "Cost",
    "pareto": "Pareto",
    "pops": "POPS/UQ",
}

STAGE_ORDER = list(STAGE_LABELS.keys())


def _find_active_spans(t, active, gap_tolerance_min=None):
    """Find contiguous spans where active is True, bridging only short sampling
    gaps (a few dropped/blank samples), NOT the genuine idle time between bursts.

    The merge tolerance is derived from the sampling cadence when not given. The
    monitor now samples every ~1-2 s, so a fixed 30 s tolerance fused a bursty
    stage's many short bursts (e.g. b_collecting, which grabs a core briefly per
    combine_b and releases it) into one misleading continuous bar. Bridging only
    ~4 samples keeps real bursts separate while still closing the one-row holes
    left by an occasional flux-exec sampling gap. Capped at the old 30 s so slow
    sampling keeps its previous behaviour."""
    if gap_tolerance_min is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        gap_tolerance_min = min(0.5, 4.0 * dt)

    raw_spans = []
    in_span = False
    start = None
    for i in range(len(active)):
        if active.iloc[i] and not in_span:
            start = t[i]
            in_span = True
        elif not active.iloc[i] and in_span:
            raw_spans.append((start, t[i]))
            in_span = False
    if in_span:
        raw_spans.append((start, t[-1]))

    # Merge spans separated by gaps smaller than tolerance
    if not raw_spans:
        return []
    merged = [raw_spans[0]]
    for start, end in raw_spans[1:]:
        if start - merged[-1][1] <= gap_tolerance_min:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Plot PotMill pipeline monitoring data")
    parser.add_argument(
        "csv", nargs="?", default="pipeline_monitor.csv", help="Path to pipeline_monitor.csv"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output figure path (default: <csv_stem>.pdf)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.csv.rsplit(".", 1)[0] + ".pdf"

    df = pd.read_csv(args.csv)
    t = df["elapsed_s"].values / 60.0  # minutes

    # Determine which stages were ever active (check both remaining and running columns)
    has_running_cols = any(f"n_{s}_running" in df.columns for s in STAGE_ORDER)
    active_stages = []
    for stage in STAGE_ORDER:
        col = f"n_{stage}"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if vals.max() > 0:
                active_stages.append(stage)

    gantt_rows = len(active_stages) if active_stages else 1
    gantt_ratio = max(1.2, 0.35 * gantt_rows)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 3.0 + gantt_ratio * 1.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, gantt_ratio], "hspace": 0.08},
    )

    # ---- Panel 1: GPU Utilization ----
    # Drop missing samples (a flux-exec timeout under load leaves an empty cell) so the trace
    # connects across the gap instead of plunging to 0, which would misread as a real drop.
    ax_gpu = axes[0]
    gpu_util = pd.to_numeric(df["mean_gpu_util_pct"], errors="coerce")
    gpu_m = gpu_util.notna()
    ax_gpu.fill_between(t[gpu_m], 0, gpu_util[gpu_m], alpha=0.25, color="#0072B2")
    ax_gpu.plot(t[gpu_m], gpu_util[gpu_m], color="#0072B2", linewidth=0.8)
    ax_gpu.set_ylabel("GPU Util. (%)", fontsize=10)
    ax_gpu.set_ylim(0, 105)
    ax_gpu.yaxis.set_major_locator(MultipleLocator(25))
    ax_gpu.grid(True, alpha=0.25, linewidth=0.5)
    ax_gpu.tick_params(labelsize=9)

    # ---- Panel 2: CPU Utilization ----
    ax_cpu = axes[1]
    cpu_util = pd.to_numeric(df["mean_cpu_util_pct"], errors="coerce")
    cpu_m = cpu_util.notna()  # see Panel 1: connect across missing samples, don't plunge to 0
    ax_cpu.fill_between(t[cpu_m], 0, cpu_util[cpu_m], alpha=0.25, color="#D55E00")
    ax_cpu.plot(t[cpu_m], cpu_util[cpu_m], color="#D55E00", linewidth=0.8)
    ax_cpu.set_ylabel("CPU Util. (physical, %)", fontsize=10)
    ax_cpu.set_ylim(0, 105)
    ax_cpu.yaxis.set_major_locator(MultipleLocator(25))
    ax_cpu.grid(True, alpha=0.25, linewidth=0.5)
    ax_cpu.tick_params(labelsize=9)

    # ---- Panel 3: Gantt Chart ----
    ax_gantt = axes[2]

    # Floor width so a single-sample (sub-cadence) burst still renders as a
    # visible tick rather than a zero-width (invisible) bar.
    min_bar_w = 1.5 * float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    for i, stage in enumerate(active_stages):
        running_col = f"n_{stage}_running"
        remaining_col = f"n_{stage}"
        if has_running_cols and running_col in df.columns:
            # Use running count: active when workers are executing tasks
            running = pd.to_numeric(df[running_col], errors="coerce").fillna(0)
            active = running > 0
        else:
            # Fallback: detect activity from remaining count decreasing
            remaining = pd.to_numeric(df[remaining_col], errors="coerce").fillna(0)
            completed = remaining.max() - remaining
            active = completed.diff().fillna(0) > 0
        spans = _find_active_spans(t, active)

        for start, end in spans:
            ax_gantt.barh(
                i,
                max(end - start, min_bar_w),
                left=start,
                height=0.65,
                color=STAGE_COLORS[stage],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

    ax_gantt.set_yticks(range(len(active_stages)))
    ax_gantt.set_yticklabels([STAGE_LABELS[s] for s in active_stages], fontsize=9)
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlabel("Elapsed Time (min)", fontsize=10)
    ax_gantt.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax_gantt.tick_params(labelsize=9)
    ax_gantt.set_xlim(t[0], t[-1])

    fig.suptitle("PotMill Pipeline Resource Monitor", fontsize=13, fontweight="bold")
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.08, top=0.94)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
