#!/usr/bin/env python3
"""Within-repetition variation of perf metrics.

For each rep × event:
  - Summary stats: n, mean, std, cv, min, p1, p5, p25, p50, p75, p95, p99, max
  - Coverage-adjusted value (count / coverage) summary
  - Lag-1 autocorrelation (how 'sticky' the series is between consecutive ticks)
  - Linear trend slope (per second)
  - Coefficient of variation of rolling means at several window sizes

Derived metrics also computed:
  - IPC               = instructions / cycles
  - L1_miss_rate      = MEM_LOAD_RETIRED.L1_MISS / (L1_HIT + L1_MISS)
  - L2_miss_rate      = L2_RQSTS.MISS / L2_RQSTS.REFERENCES
  - Mem-bound frac    = TOPDOWN.MEMORY_BOUND_SLOTS / TOPDOWN.SLOTS_P
  - StallL1D_per_cyc  = MEMORY_ACTIVITY.STALLS_L1D_MISS / cycles
  - StallL2_per_cyc   = MEMORY_ACTIVITY.STALLS_L2_MISS  / cycles

Outputs:
  within/summary_raw.csv     — per rep × event raw counts
  within/summary_adj.csv     — per rep × event coverage-adjusted (count/cov)
  within/summary_derived.csv — per rep × derived metric
  within/timeseries_plots/   — one PNG per rep with all 13 metrics over time
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
    "instructions",
    "cycles",
    "L2_RQSTS.MISS",
    "L2_RQSTS.REFERENCES",
    "MEM_LOAD_COMPLETED.L1_MISS_ANY",
    "MEM_LOAD_RETIRED.L1_HIT",
    "MEM_LOAD_RETIRED.L2_HIT",
    "MEM_LOAD_RETIRED.L1_MISS",
    "MEM_LOAD_RETIRED.L2_MISS",
    "TOPDOWN.SLOTS_P",
    "TOPDOWN.MEMORY_BOUND_SLOTS",
    "MEMORY_ACTIVITY.STALLS_L1D_MISS",
    "MEMORY_ACTIVITY.STALLS_L2_MISS",
]

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def describe(x: np.ndarray) -> dict:
    """Compact descriptive stats."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    m = x.mean()
    s = x.std(ddof=1) if x.size > 1 else 0.0
    out = {
        "n": int(x.size),
        "mean": float(m),
        "std": float(s),
        "cv": float(s / m) if m != 0 else float("nan"),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    qs = np.percentile(x, PERCENTILES)
    for p, q in zip(PERCENTILES, qs):
        out[f"p{p}"] = float(q)
    return out


def lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    a = x[:-1] - x[:-1].mean()
    b = x[1:] - x[1:].mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def slope_per_sec(ts: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of x vs ts (units: metric per second)."""
    ts = np.asarray(ts, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(ts) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan")
    t = ts[mask] - ts[mask].mean()
    y = x[mask] - x[mask].mean()
    denom = (t * t).sum()
    return float((t * y).sum() / denom) if denom > 0 else float("nan")


def rolling_cv(x: pd.Series, window: int) -> float:
    """CV of rolling means with given window. Captures slow drift."""
    if len(x) < window * 2:
        return float("nan")
    r = x.rolling(window, min_periods=window).mean().dropna()
    if r.empty or r.mean() == 0:
        return float("nan")
    return float(r.std(ddof=1) / r.mean())


def derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"ts": df["ts"]})
    out["IPC"] = df["instructions"] / df["cycles"]
    l1_total = df["MEM_LOAD_RETIRED.L1_HIT"] + df["MEM_LOAD_RETIRED.L1_MISS"]
    out["L1_miss_rate"] = df["MEM_LOAD_RETIRED.L1_MISS"] / l1_total
    out["L2_miss_rate"] = df["L2_RQSTS.MISS"] / df["L2_RQSTS.REFERENCES"]
    out["MemBound_frac"] = df["TOPDOWN.MEMORY_BOUND_SLOTS"] / df["TOPDOWN.SLOTS_P"]
    out["StallL1D_per_cyc"] = df["MEMORY_ACTIVITY.STALLS_L1D_MISS"] / df["cycles"]
    out["StallL2_per_cyc"] = df["MEMORY_ACTIVITY.STALLS_L2_MISS"] / df["cycles"]
    return out


def analyze_rep(rep: int, parquet: Path, plots_dir: Path | None) -> tuple[list[dict], list[dict], list[dict]]:
    df = pd.read_parquet(parquet)
    # Local time axis (start at 0)
    ts = df["ts"].to_numpy()
    ts_local = ts - ts[0]

    raw_rows: list[dict] = []
    adj_rows: list[dict] = []
    for ev in EVENTS:
        vals = df[ev].to_numpy()
        cov = df[f"{ev}__cov"].to_numpy()

        row = {"rep": rep, "event": ev}
        row.update(describe(vals))
        row["lag1_acf"] = lag1_autocorr(vals)
        row["slope_per_sec"] = slope_per_sec(ts_local, vals)
        row["rolling_cv_60s"] = rolling_cv(df[ev], 60)
        row["rolling_cv_600s"] = rolling_cv(df[ev], 600)
        row["cov_mean"] = float(np.nanmean(cov))
        row["cov_min"] = float(np.nanmin(cov))
        raw_rows.append(row)

        # Coverage-adjusted (extrapolate count to 100% sampling)
        with np.errstate(divide="ignore", invalid="ignore"):
            adj = vals / (cov / 100.0)
        adj_row = {"rep": rep, "event": ev}
        adj_row.update(describe(adj))
        adj_row["lag1_acf"] = lag1_autocorr(adj)
        adj_rows.append(adj_row)

    der = derived_metrics(df)
    der_rows: list[dict] = []
    for col in der.columns:
        if col == "ts":
            continue
        v = der[col].to_numpy()
        row = {"rep": rep, "metric": col}
        row.update(describe(v))
        row["lag1_acf"] = lag1_autocorr(v)
        row["slope_per_sec"] = slope_per_sec(ts_local, v)
        row["rolling_cv_60s"] = rolling_cv(der[col], 60)
        row["rolling_cv_600s"] = rolling_cv(der[col], 600)
        der_rows.append(row)

    if plots_dir is not None:
        _plot_rep(rep, ts_local, df, der, plots_dir)

    return raw_rows, adj_rows, der_rows


def _plot_rep(rep: int, ts_local: np.ndarray, df: pd.DataFrame, der: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 13 raw events + 6 derived = 19 panels => 5x4 grid
    panels = EVENTS + [c for c in der.columns if c != "ts"]
    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14), sharex=True)
    axes = axes.flatten()
    # Downsample for plotting if very large
    step = max(1, len(ts_local) // 5000)
    t = ts_local[::step] / 3600.0  # hours

    for i, name in enumerate(panels):
        ax = axes[i]
        if name in df.columns:
            y = df[name].to_numpy()[::step]
        else:
            y = der[name].to_numpy()[::step]
        ax.plot(t, y, linewidth=0.4)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("time within rep (h)", fontsize=8)
    fig.suptitle(f"Rep {rep}: time series", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"rep_{rep:02d}_timeseries.png", dpi=120)
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
        "20260425_145215/analysis/within",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--reps", type=int, nargs="*", default=None)
    args = p.parse_args()

    pq_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = None if args.no_plots else out_dir / "timeseries_plots"

    files = sorted(pq_dir.glob("rep_*.parquet"),
                   key=lambda f: int(f.stem.split("_")[1]))
    if args.reps:
        files = [f for f in files if int(f.stem.split("_")[1]) in args.reps]
    if not files:
        print(f"No parquet files in {pq_dir}", file=sys.stderr)
        return 1

    all_raw, all_adj, all_der = [], [], []
    for f in files:
        rep = int(f.stem.split("_")[1])
        print(f"  rep {rep:2d}: analyzing {f.name}")
        r, a, d = analyze_rep(rep, f, plots_dir)
        all_raw.extend(r)
        all_adj.extend(a)
        all_der.extend(d)

    pd.DataFrame(all_raw).to_csv(out_dir / "summary_raw.csv", index=False)
    pd.DataFrame(all_adj).to_csv(out_dir / "summary_adj.csv", index=False)
    pd.DataFrame(all_der).to_csv(out_dir / "summary_derived.csv", index=False)

    print(f"\nWrote within-rep summaries to {out_dir}")
    print("  - summary_raw.csv     (raw event counts)")
    print("  - summary_adj.csv     (count / coverage*100, extrapolated)")
    print("  - summary_derived.csv (IPC, L1/L2 miss rate, mem-bound, stalls/cyc)")
    if plots_dir:
        print(f"  - {plots_dir.name}/ (one PNG per rep)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
