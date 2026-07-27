#!/usr/bin/env python3
"""DAMOV characterisation + cache-hierarchy analysis.

Reads the per-rep Parquet files produced by parse_perf.py and computes:

  DAMOV metrics (per rep, then aggregated across reps):
    - LLC MPKI   = MEM_LOAD_RETIRED.L2_MISS * 1000 / instructions
    - LFMR       = MEM_LOAD_RETIRED.L2_MISS / MEM_LOAD_RETIRED.L1_MISS
    - IPC        = instructions / cycles

  Cache-hierarchy breakdown (from retired-load events):
    - L1 miss rate  = MEM_LOAD_RETIRED.L1_MISS / (L1_HIT + L1_MISS)
    - L2 hit rate   = MEM_LOAD_RETIRED.L2_HIT  / MEM_LOAD_RETIRED.L1_MISS
    - L2 miss rate  = MEM_LOAD_RETIRED.L2_MISS / (L2_HIT + L2_MISS)
    - L3 miss rate  = MEM_LOAD_RETIRED.L3_MISS / (L3_HIT + L3_MISS)

  NOTE: Top-Down slot-based metrics (Memory Bound, Retiring, Bad Speculation,
  Front-End Bound, and the L1D/L2 stall fractions) require TOPDOWN.* and
  MEMORY_ACTIVITY.* counters, which are NOT part of the collected event set
  (see EVENTS below) and are therefore not computed here. Likewise
  L2_RQSTS.MISS/REFERENCES were not collected, so the L2 miss rate above is
  derived from MEM_LOAD_RETIRED.L2_HIT/L2_MISS instead (retired loads only,
  not all L2 requests/prefetches).

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
    "MEM_LOAD_COMPLETED.L1_MISS_ANY",
    "MEM_LOAD_RETIRED.L1_HIT", "MEM_LOAD_RETIRED.L2_HIT",
    "MEM_LOAD_RETIRED.L1_MISS", "MEM_LOAD_RETIRED.L2_MISS",
    "MEM_LOAD_RETIRED.L3_HIT", "MEM_LOAD_RETIRED.L3_MISS",
]

# DAMOV thresholds (from the DAMOV paper, Table 1).
# NOTE: the paper's "mem_bound_frac > 0.2" criterion is omitted here because
# TOPDOWN.MEMORY_BOUND_SLOTS / TOPDOWN.SLOTS_P was not collected. The
# DAMOV verdict below is therefore based on 3 of the paper's 4 criteria.
DAMOV_THRESHOLDS = {
    "l2_mpki": {"threshold": 1.0, "direction": ">", "label": "LLC MPKI > 1 → memory-bound candidate"},
    "lfmr":     {"threshold": 0.1, "direction": ">", "label": "LFMR > 0.1 → significant LLC traffic"},
    "ipc":      {"threshold": 1.0, "direction": "<", "label": "IPC < 1.0 → limited by memory/stalls"},
}

METRIC_DESCRIPTIONS = {
    "l2_mpki":      "LLC MPKI (L2_MISS×1000/instructions)",
    "lfmr":          "LFMR (L2_MISS/L1_MISS)",
    "ipc":           "IPC (instructions/cycles)",
    "l1_miss_rate":  "L1 miss rate (L1_MISS/(L1_HIT+L1_MISS))",
    "l2_hit_rate":   "L2 hit rate (L2_HIT/L1_MISS)",
    "l2_miss_rate":  "L2 miss rate (L2_MISS/(L2_HIT+L2_MISS))",
    "l3_miss_rate":  "L3 miss rate (L3_MISS/(L3_HIT+L3_MISS))",
}


def load_rep(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def compute_damov_per_tick(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-tick DAMOV and cache-hierarchy metrics from retired-load events."""
    out = pd.DataFrame({"ts": df["ts"]})
    instr = df["instructions"]
    cyc   = df["cycles"]
    l1m   = df["MEM_LOAD_RETIRED.L1_MISS"]
    l1h   = df["MEM_LOAD_RETIRED.L1_HIT"]
    l2m   = df["MEM_LOAD_RETIRED.L2_MISS"]
    l2h   = df["MEM_LOAD_RETIRED.L2_HIT"]
    l3m   = df["MEM_LOAD_RETIRED.L3_MISS"]
    l3h   = df["MEM_LOAD_RETIRED.L3_HIT"]

    out["l2_mpki"]     = l2m * 1000.0 / instr
    out["lfmr"]         = l2m / l1m
    out["ipc"]          = instr / cyc
    out["l1_miss_rate"] = l1m / (l1h + l1m)
    out["l2_hit_rate"]  = l2h / l1m
    out["l2_miss_rate"] = l2m / (l2h + l2m)
    out["l3_miss_rate"] = l3m / (l3h + l3m)
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

    fig.suptitle("DAMOV + cache-hierarchy per-rep means (±2σ band, red dashed = DAMOV threshold)",
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
        print(f"  rep {rep:2d}: l2_mpki={row['l2_mpki']:.3f}  "
              f"lfmr={row['lfmr']:.3f}  ipc={row['ipc']:.3f}")

    per_rep = pd.DataFrame(per_rep_rows)
    per_rep.to_csv(out_dir / "per_rep.csv", index=False)

    # Aggregate across reps
    metrics = [c for c in per_rep.columns if c != "rep"]
    agg_rows = []
    for m in metrics:
        vals = per_rep[m].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"  WARNING: metric '{m}' has no finite values in any rep "
                  f"(source event likely all-zero/uncounted) — reporting as NaN",
                  file=sys.stderr)
        mu   = float(vals.mean()) if vals.size else float("nan")
        sd   = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        cv   = sd / mu if (vals.size and mu != 0) else float("nan")
        rng  = (float((vals.max() - vals.min()) / mu * 100)
                if (vals.size and mu != 0) else float("nan"))
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
    lines.append("Cache-hierarchy sub-breakdown")
    lines.append("-" * 70)
    td_metrics = ["l1_miss_rate", "l2_hit_rate", "l2_miss_rate", "l3_miss_rate"]
    for _, row in agg[agg["metric"].isin(td_metrics)].iterrows():
        lines.append(f"  {row['metric']:20s}  mean={row['mean']:>10.4f}  "
                     f"CV={row['cv_pct']:5.2f}%  range={row['range_pct']:5.2f}%")
    lines.append("")
    lines.append("Top-Down note")
    lines.append("-" * 70)
    lines.append("  No Top-Down slot metrics (Retiring, Bad Speculation, Front-End")
    lines.append("  Bound, or Memory Bound) are measurable from this event set —")
    lines.append("  TOPDOWN.* and MEMORY_ACTIVITY.* counters were not collected.")
    lines.append("  L2_RQSTS.MISS/REFERENCES were also not collected, so the L2")
    lines.append("  miss rate above is derived from retired-load hit/miss counts")
    lines.append("  (MEM_LOAD_RETIRED.L2_HIT/L2_MISS) rather than all L2 requests.")
    lines.append("  The DAMOV mem-bound-fraction criterion (>0.2) is therefore")
    lines.append("  omitted from the verdict below; only LLC MPKI, LFMR, and IPC")
    lines.append("  are evaluated against the paper's thresholds.")
    lines.append("")

    # Threshold summary
    lines.append("DAMOV threshold summary (3 of 4 paper criteria — see note above)")
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
        lines.append("  ✓ All available DAMOV thresholds exceeded → workload is a")
        lines.append("    strong memory-bound candidate; PIM offloading is motivated.")
        lines.append("    (Mem-bound-fraction criterion not evaluated — see note above.)")
    else:
        lines.append("  ✗ Not all available DAMOV thresholds exceeded — see individual metrics.")
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