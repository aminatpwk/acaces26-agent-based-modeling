#!/usr/bin/env python3
"""Across-repetition variation of perf metrics.

For each event (and each derived metric):
  - per-rep mean & std (table: reps × metrics)
  - aggregate: mean-of-means, CV-of-means (between-rep noise)
  - mean-of-within-rep-CV (within-rep noise)
  - between-vs-within variance ratio (eta_sq) — fraction of total
    variance explained by rep identity (one-way ANOVA decomposition)
  - Kruskal-Wallis H statistic & p value (downsampled for tractability)
  - F-stat & p value (one-way ANOVA)

Outputs:
  across/per_rep_means.csv     — wide: rep × event/metric → mean
  across/per_rep_stds.csv      — wide: rep × event/metric → std
  across/aggregate.csv         — long: per metric aggregate stats
  across/boxplots.png          — distributions across reps
  across/rep_means_strip.png   — per-rep mean with ±2σ global band

All analyses include coverage-adjusted variants where appropriate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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

DERIVED = ["IPC", "L1_miss_rate", "L2_miss_rate", "MemBound_frac",
           "StallL1D_per_cyc", "StallL2_per_cyc"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IPC"] = df["instructions"] / df["cycles"]
    l1_total = df["MEM_LOAD_RETIRED.L1_HIT"] + df["MEM_LOAD_RETIRED.L1_MISS"]
    df["L1_miss_rate"] = df["MEM_LOAD_RETIRED.L1_MISS"] / l1_total
    df["L2_miss_rate"] = df["L2_RQSTS.MISS"] / df["L2_RQSTS.REFERENCES"]
    df["MemBound_frac"] = df["TOPDOWN.MEMORY_BOUND_SLOTS"] / df["TOPDOWN.SLOTS_P"]
    df["StallL1D_per_cyc"] = df["MEMORY_ACTIVITY.STALLS_L1D_MISS"] / df["cycles"]
    df["StallL2_per_cyc"] = df["MEMORY_ACTIVITY.STALLS_L2_MISS"] / df["cycles"]
    return df


def load_rep(path: Path, cov_adjust: bool) -> pd.DataFrame:
    """Load one rep parquet. If cov_adjust, scale each event by 100/coverage."""
    df = pd.read_parquet(path)
    if cov_adjust:
        for ev in EVENTS:
            cov = df[f"{ev}__cov"].to_numpy() / 100.0
            with np.errstate(divide="ignore", invalid="ignore"):
                df[ev] = df[ev].to_numpy() / cov
    df = add_derived(df)
    keep = ["ts"] + EVENTS + DERIVED
    return df[keep]


def kw_test(samples: list[np.ndarray], max_per_rep: int = 5000) -> tuple[float, float]:
    """Kruskal-Wallis with optional downsampling per rep for tractability."""
    cleaned = []
    rng = np.random.default_rng(42)
    for s in samples:
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        if s.size > max_per_rep:
            idx = rng.choice(s.size, size=max_per_rep, replace=False)
            s = s[idx]
        cleaned.append(s)
    if len(cleaned) < 2:
        return float("nan"), float("nan")
    try:
        h, p = stats.kruskal(*cleaned)
        return float(h), float(p)
    except ValueError:
        return float("nan"), float("nan")


def anova_decompose(samples: list[np.ndarray]) -> dict:
    """One-way ANOVA on full samples.

    Returns mean of means, SS_between, SS_within, eta_sq, F, p.
    Robust to wildly unequal sample sizes (which we don't actually have here).
    """
    clean = [s[np.isfinite(s)] for s in samples]
    clean = [s for s in clean if s.size > 0]
    k = len(clean)
    if k < 2:
        return {
            "k_reps": k, "mean_of_means": float("nan"),
            "ss_between": float("nan"), "ss_within": float("nan"),
            "ss_total": float("nan"), "eta_sq": float("nan"),
            "F": float("nan"), "p": float("nan"),
        }
    sizes = np.array([s.size for s in clean])
    means = np.array([s.mean() for s in clean])
    vars_ = np.array([s.var(ddof=1) if s.size > 1 else 0.0 for s in clean])
    N = sizes.sum()
    grand = (sizes * means).sum() / N
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float((vars_ * (sizes - 1)).sum())
    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
    df_b = k - 1
    df_w = N - k
    ms_b = ss_between / df_b if df_b > 0 else float("nan")
    ms_w = ss_within / df_w if df_w > 0 else float("nan")
    F = ms_b / ms_w if ms_w and ms_w > 0 else float("nan")
    p = float(stats.f.sf(F, df_b, df_w)) if np.isfinite(F) else float("nan")
    return {
        "k_reps": k, "mean_of_means": float(grand),
        "ss_between": ss_between, "ss_within": ss_within, "ss_total": ss_total,
        "eta_sq": float(eta_sq), "F": float(F), "p": p,
    }


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
        "20260425_145215/analysis/across",
    )
    p.add_argument("--cov-adjust", action="store_true",
                   help="Scale event counts by 100/coverage before analysis.")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    pq_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pq_dir.glob("rep_*.parquet"),
                   key=lambda f: int(f.stem.split("_")[1]))
    if not files:
        print(f"No parquet files in {pq_dir}", file=sys.stderr)
        return 1

    metrics = EVENTS + DERIVED
    print(f"Loading {len(files)} reps...")
    rep_data: dict[int, pd.DataFrame] = {}
    for f in files:
        rep = int(f.stem.split("_")[1])
        rep_data[rep] = load_rep(f, args.cov_adjust)
        print(f"  rep {rep:2d}: {len(rep_data[rep])} ticks")

    reps = sorted(rep_data.keys())

    # Per-rep means / stds (wide format)
    mean_rows = []
    std_rows = []
    for rep in reps:
        df = rep_data[rep]
        row_m = {"rep": rep}
        row_s = {"rep": rep}
        for m in metrics:
            v = df[m].to_numpy()
            v = v[np.isfinite(v)]
            row_m[m] = float(v.mean()) if v.size else float("nan")
            row_s[m] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        mean_rows.append(row_m)
        std_rows.append(row_s)
    per_rep_means = pd.DataFrame(mean_rows)
    per_rep_stds = pd.DataFrame(std_rows)
    per_rep_means.to_csv(out_dir / "per_rep_means.csv", index=False)
    per_rep_stds.to_csv(out_dir / "per_rep_stds.csv", index=False)

    # Aggregate stats per metric
    agg_rows = []
    for m in metrics:
        samples = [rep_data[r][m].to_numpy() for r in reps]
        means = per_rep_means[m].to_numpy()
        stds = per_rep_stds[m].to_numpy()
        within_cv = np.where(means != 0, stds / means, np.nan)
        mom = means.mean()
        cv_means = means.std(ddof=1) / mom if mom != 0 else float("nan")
        H, p_kw = kw_test(samples)
        an = anova_decompose(samples)
        agg_rows.append({
            "metric": m,
            "mean_of_rep_means": mom,
            "std_of_rep_means": float(means.std(ddof=1)),
            "cv_of_rep_means": float(cv_means),
            "min_rep_mean": float(means.min()),
            "max_rep_mean": float(means.max()),
            "range_pct_of_mean": float((means.max() - means.min()) / mom * 100)
                                  if mom != 0 else float("nan"),
            "mean_within_rep_cv": float(np.nanmean(within_cv)),
            "ratio_between_within": (float(cv_means) /
                                     float(np.nanmean(within_cv)))
                                     if np.nanmean(within_cv) > 0
                                     else float("nan"),
            "kruskal_H": H, "kruskal_p": p_kw,
            "anova_F": an["F"], "anova_p": an["p"], "eta_sq": an["eta_sq"],
        })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    # ---- Plots ----
    if not args.no_plots:
        _plot_boxplots(rep_data, reps, metrics, out_dir / "boxplots.png")
        _plot_rep_means_strip(per_rep_means, agg, metrics,
                              out_dir / "rep_means_strip.png")

    # Console summary
    print(f"\nPer-rep means (head):")
    print(per_rep_means.to_string(index=False))
    print(f"\nAggregate across reps:")
    cols = ["metric", "mean_of_rep_means", "cv_of_rep_means",
            "mean_within_rep_cv", "ratio_between_within",
            "range_pct_of_mean", "eta_sq", "anova_p"]
    print(agg[cols].to_string(index=False,
                              formatters={c: "{:.4g}".format
                                          for c in cols if c != "metric"}))
    print(f"\nWrote across-rep results to {out_dir}")
    return 0


def _plot_boxplots(rep_data: dict, reps: list[int],
                   metrics: list[str], out_path: Path) -> None:
    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14))
    axes = axes.flatten()
    rng = np.random.default_rng(0)
    for i, m in enumerate(metrics):
        ax = axes[i]
        data = []
        for r in reps:
            v = rep_data[r][m].to_numpy()
            v = v[np.isfinite(v)]
            if v.size > 20000:
                v = rng.choice(v, 20000, replace=False)
            data.append(v)
        ax.boxplot(data, tick_labels=[str(r) for r in reps], showfliers=False,
                   widths=0.6)
        ax.set_title(m, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("rep", fontsize=8)
    fig.suptitle("Distribution per rep (outliers hidden, downsampled)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_rep_means_strip(per_rep_means: pd.DataFrame,
                          agg: pd.DataFrame,
                          metrics: list[str],
                          out_path: Path) -> None:
    nrows, ncols = 5, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14))
    axes = axes.flatten()
    reps = per_rep_means["rep"].to_numpy()
    for i, m in enumerate(metrics):
        ax = axes[i]
        y = per_rep_means[m].to_numpy()
        mu = y.mean()
        sd = y.std(ddof=1)
        ax.axhspan(mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.10)
        ax.axhline(mu, color="C0", linewidth=1, alpha=0.7)
        ax.plot(reps, y, "o-", color="C1", markersize=4, linewidth=0.8)
        ax.set_title(f"{m}\nCV={sd/mu*100:.2f}%" if mu != 0 else m,
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(reps)
        ax.grid(alpha=0.3)
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("rep", fontsize=8)
    fig.suptitle("Per-rep means (blue band = mean ± 2σ across reps)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
