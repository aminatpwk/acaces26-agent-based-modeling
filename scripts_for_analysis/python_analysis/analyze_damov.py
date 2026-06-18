#!/usr/bin/env python3
"""DAMOV characterisation + Intel Top-Down L1/L2 analysis.

Reads the per-rep Parquet files produced by parse_perf.py and computes:

  DAMOV metrics (per rep, then aggregated across reps):
    - LLC MPKI   = MEM_LOAD_RETIRED.L2_MISS * 1000 / instructions
    - LFMR       = MEM_LOAD_RETIRED.L2_MISS / MEM_LOAD_RETIRED.L1_MISS
    - IPC        = instructions / cycles
    - Mem-bound  = TOPDOWN.MEMORY_BOUND_SLOTS / TOPDOWN.SLOTS_P

  Top-Down L1 (slot fractions, Intel Sapphire Rapids):
    - Memory Bound  = TOPDOWN.MEMORY_BOUND_SLOTS / TOPDOWN.SLOTS_P
    - (Retiring, Bad Spec, Front-End Bound are NOT directly available
       from this 5-event subset — noted as N/A)

  Top-Down L2 memory sub-breakdown:
    - L1 stall fraction  = MEMORY_ACTIVITY.STALLS_L1D_MISS / TOPDOWN.SLOTS_P
    - L2 stall fraction  = MEMORY_ACTIVITY.STALLS_L2_MISS  / TOPDOWN.SLOTS_P
    - L2 hit rate        = MEM_LOAD_RETIRED.L2_HIT / MEM_LOAD_RETIRED.L1_MISS
    - LLC miss rate      = MEM_LOAD_RETIRED.L2_MISS / MEM_LOAD_RETIRED.L1_MISS (= LFMR)

  For each metric: per-rep mean, mean-of-means, CV across reps, range %.
  Outputs:
    damov/per_rep.csv          — per-rep scalar values for every DAMOV metric
    damov/aggregate.csv        — mean, cv, range_pct, DAMOV threshold annotation
    damov/damov_report.txt     — human-readable summary with threshold decisions
    damov/rep_strip.png        — strip plot of per-rep values with ±2σ band
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVENTS = [
    "instructions", "cycles",
    "L2_RQSTS.MISS", "L2_RQSTS.REFERENCES",
    "MEM_LOAD_COMPLETED.L1_MISS_ANY",
    "MEM_LOAD_RETIRED.L1_HIT", "MEM_LOAD_RETIRED.L2_HIT",
    "MEM_LOAD_RETIRED.L1_MISS", "MEM_LOAD_RETIRED.L2_MISS",
    "TOPDOWN.SLOTS_P", "TOPDOWN.MEMORY_BOUND_SLOTS",
    "MEMORY_ACTIVITY.STALLS_L1D_MISS", "MEMORY_ACTIVITY.STALLS_L2_MISS",
]

# DAMOV thresholds (from the DAMOV paper, Table 1)
DAMOV_THRESHOLDS = {
    "llc_mpki":      {"threshold": 1.0,  "direction": ">", "label": "LLC MPKI > 1 → memory-bound candidate"},
    "lfmr":          {"threshold": 0.1,  "direction": ">", "label": "LFMR > 0.1 → significant LLC traffic"},
    "mem_bound_frac":{"threshold": 0.2,  "direction": ">", "label": "Mem-bound fraction > 0.2 → memory-bound"},
    "ipc":           {"threshold": 1.0,  "direction": "<", "label": "IPC < 1.0 → limited by memory/stalls"},
}

METRIC_DESCRIPTIONS = {
    "llc_mpki":          "LLC MPKI (L2_MISS×1000/instructions)",
    "lfmr":              "LFMR (L2_MISS/L1_MISS)",
    "ipc":               "IPC (instructions/cycles)",
    "mem_bound_frac":    "Memory-bound fraction (TDB)",
    "l1_stall_frac":     "L1D stall fraction (STALLS_L1D/SLOTS)",
    "l2_stall_frac":     "L2 stall fraction (STALLS_L2/SLOTS)",
    "l2_hit_rate":       "L2 hit rate (L2_HIT/L1_MISS)",
    "l2_miss_rate_req":  "L2 miss rate (L2_RQSTS.MISS/L2_RQSTS.REF) — includes prefetcher",
    "l1_miss_rate":      "L1 miss rate (L1_MISS/(L1_HIT+L1_MISS))",
}


def load_rep(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def compute_damov_per_tick(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-tick DAMOV and Top-Down L2 metrics."""
    out = pd.DataFrame({"ts": df["ts"]})
    instr = df["instructions"]
    cyc   = df["cycles"]
    l1m   = df["MEM_LOAD_RETIRED.L1_MISS"]
    l2m   = df["MEM_LOAD_RETIRED.L2_MISS"]
    l2h   = df["MEM_LOAD_RETIRED.L2_HIT"]
    l1h   = df["MEM_LOAD_RETIRED.L1_HIT"]
    slots = df["TOPDOWN.SLOTS_P"]
    mbs   = df["TOPDOWN.MEMORY_BOUND_SLOTS"]
    sl1d  = df["MEMORY_ACTIVITY.STALLS_L1D_MISS"]
    sl2   = df["MEMORY_ACTIVITY.STALLS_L2_MISS"]
    l2rq_m = df["L2_RQSTS.MISS"]
    l2rq_r = df["L2_RQSTS.REFERENCES"]

    out["llc_mpki"]         = l2m * 1000.0 / instr
    out["lfmr"]             = l2m / l1m
    out["ipc"]              = instr / cyc
    out["mem_bound_frac"]   = mbs / slots
    out["l1_stall_frac"]    = sl1d / slots
    out["l2_stall_frac"]    = sl2 / slots
    out["l2_hit_rate"]      = l2h / l1m
    out["l2_miss_rate_req"] = l2rq_m / l2rq_r
    out["l1_miss_rate"]     = l1m / (l1h + l1m)
    return out


def rep_scalar(df_tick: pd.DataFrame) -> dict:
    """Reduce per-tick metric df to per-rep scalars (mean of finite values)."""
    row = {}
    for col in df_tick.columns:
        if col == "ts":
            continue
        v = df_tick[col].to_numpy()
        v = v[np.isfinite(v)]
        row[col] = float(v.mean()) if v.size else float("nan")
    return row


def check_threshold(metric: str, value: float) -> str:
    if metric not in DAMOV_THRESHOLDS:
        return "—"
    t = DAMOV_THRESHOLDS[metric]
    passes = (value > t["threshold"]) if t["direction"] == ">" else (value < t["threshold"])
    return "✓ EXCEEDS" if passes else "✗ below"


def plot_strip(per_rep: pd.DataFrame, out_path: Path) -> None:
    metrics = [c for c in per_rep.columns if c != "rep"]
    n = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()
    reps = per_rep["rep"].to_numpy()

    for i, m in enumerate(metrics):
        ax = axes[i]
        y = per_rep[m].to_numpy()
        mu = np.nanmean(y)
        sd = np.nanstd(y, ddof=1)
        cv_pct = sd / mu * 100 if mu != 0 else 0

        ax.axhspan(mu - 2*sd, mu + 2*sd, color="steelblue", alpha=0.10)
        ax.axhline(mu, color="steelblue", linewidth=1, alpha=0.7)
        ax.plot(reps, y, "o-", color="darkorange", markersize=5, linewidth=0.9)

        # DAMOV threshold line
        if m in DAMOV_THRESHOLDS:
            t = DAMOV_THRESHOLDS[m]
            ax.axhline(t["threshold"], color="crimson", linewidth=0.8,
                       linestyle="--", alpha=0.6)
            ax.text(reps[-1] + 0.05, t["threshold"], f"threshold={t['threshold']}",
                    fontsize=7, color="crimson", va="center")

        ax.set_title(f"{m}\nCV={cv_pct:.2f}%", fontsize=9)
        ax.set_xticks(reps)
        ax.set_xlabel("rep", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("DAMOV + Top-Down per-rep means (±2σ band, red dashed = DAMOV threshold)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet-dir",
        default="/home/toros/abm/perf_results/epidemiology/benchmark_results/"
                "20260425_145215/analysis/parquet",
    )
    p.add_argument(
        "--out-dir",
        default="/home/toros/abm/perf_results/epidemiology/benchmark_results/"
                "20260425_145215/analysis/damov",
    )
    p.add_argument("--reps", type=int, nargs="*", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    pq_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pq_dir.glob("rep_*.parquet"),
                   key=lambda f: int(f.stem.split("_")[1]))
    if args.reps:
        files = [f for f in files if int(f.stem.split("_")[1]) in args.reps]
    if not files:
        print(f"No parquet files in {pq_dir}", file=sys.stderr)
        return 1

    print(f"Computing DAMOV metrics for {len(files)} reps...")
    per_rep_rows = []
    for f in files:
        rep = int(f.stem.split("_")[1])
        df = load_rep(f)
        tick_metrics = compute_damov_per_tick(df)
        row = {"rep": rep}
        row.update(rep_scalar(tick_metrics))
        per_rep_rows.append(row)
        print(f"  rep {rep:2d}: llc_mpki={row['llc_mpki']:.3f}  "
              f"lfmr={row['lfmr']:.3f}  ipc={row['ipc']:.3f}  "
              f"mem_bound={row['mem_bound_frac']:.3f}")

    per_rep = pd.DataFrame(per_rep_rows)
    per_rep.to_csv(out_dir / "per_rep.csv", index=False)

    # Aggregate across reps
    metrics = [c for c in per_rep.columns if c != "rep"]
    agg_rows = []
    for m in metrics:
        vals = per_rep[m].to_numpy()
        vals = vals[np.isfinite(vals)]
        mu   = float(vals.mean()) if vals.size else float("nan")
        sd   = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        cv   = sd / mu if mu != 0 else float("nan")
        rng  = float((vals.max() - vals.min()) / mu * 100) if mu != 0 else float("nan")
        agg_rows.append({
            "metric":           m,
            "description":      METRIC_DESCRIPTIONS.get(m, ""),
            "mean":             mu,
            "std":              sd,
            "cv_pct":           cv * 100,
            "range_pct":        rng,
            "damov_threshold":  DAMOV_THRESHOLDS.get(m, {}).get("threshold", "—"),
            "damov_verdict":    check_threshold(m, mu),
        })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    # Human-readable report
    lines = []
    lines.append("=" * 70)
    lines.append("DAMOV CHARACTERISATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("DAMOV headline metrics")
    lines.append("-" * 70)
    for _, row in agg[agg["metric"].isin(DAMOV_THRESHOLDS)].iterrows():
        lines.append(f"  {row['metric']:20s}  mean={row['mean']:>10.4f}  "
                     f"CV={row['cv_pct']:5.2f}%  range={row['range_pct']:5.2f}%  "
                     f"→ {row['damov_verdict']}")
        if row["metric"] in DAMOV_THRESHOLDS:
            lines.append(f"    ({DAMOV_THRESHOLDS[row['metric']]['label']})")
    lines.append("")
    lines.append("Top-Down L2 memory sub-breakdown")
    lines.append("-" * 70)
    td_metrics = ["l1_stall_frac", "l2_stall_frac", "l2_hit_rate",
                  "l2_miss_rate_req", "l1_miss_rate"]
    for _, row in agg[agg["metric"].isin(td_metrics)].iterrows():
        lines.append(f"  {row['metric']:20s}  mean={row['mean']:>10.4f}  "
                     f"CV={row['cv_pct']:5.2f}%  range={row['range_pct']:5.2f}%")
    lines.append("")
    lines.append("Top-Down L1 note")
    lines.append("-" * 70)
    lines.append("  Retiring / Bad Speculation / Front-End Bound are not directly")
    lines.append("  measurable from this 13-event perf group (would require separate")
    lines.append("  PERF_METRICS_* counters on Sapphire Rapids). Only Memory Bound")
    lines.append("  (= TOPDOWN.MEMORY_BOUND_SLOTS / TOPDOWN.SLOTS_P) is available.")
    lines.append("")

    # Threshold summary
    lines.append("DAMOV threshold summary")
    lines.append("-" * 70)
    all_pass = True
    for m, info in DAMOV_THRESHOLDS.items():
        row = agg[agg["metric"] == m]
        if row.empty:
            continue
        verdict = row.iloc[0]["damov_verdict"]
        passes = "✓" in verdict
        if not passes:
            all_pass = False
        lines.append(f"  {info['label']}")
        lines.append(f"    measured mean = {row.iloc[0]['mean']:.4f}  {verdict}")
    lines.append("")
    if all_pass:
        lines.append("  ✓ ALL DAMOV thresholds exceeded → workload qualifies as")
        lines.append("    memory-bound; PIM offloading is motivated.")
    else:
        lines.append("  ✗ Not all DAMOV thresholds exceeded — see individual metrics.")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    (out_dir / "damov_report.txt").write_text(report_text)
    print()
    print(report_text)

    if not args.no_plots:
        plot_strip(per_rep, out_dir / "rep_strip.png")
        print(f"\nPlot -> {out_dir / 'rep_strip.png'}")

    print(f"\nOutputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())