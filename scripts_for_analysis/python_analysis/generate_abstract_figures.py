"""
generate_figures.py
-------------------

Usage:
  python generate_figures.py
  python generate_figures.py --root /path/to/repo --dpi 300 --out /path/to/outdir
"""

import argparse
import csv
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Agent scales and the DAMOV aggregate files that exist for each.
# Paths are relative to the repository root passed via --root.
SCALE_CONFIG = {
    "10M": [
        "10million_agents/vm_run_01/analysis/damov/aggregate.csv",
        "10million_agents/vm_run_02/analysis/damov/aggregate.csv",
        "10million_agents/vm_run_03/analysis/damov/aggregate.csv",
    ],
    "50M": [
        "50million_agents/vm_run_01/analysis/damov/aggregate.csv",
        "50million_agents/vm_run_02/analysis/damov/aggregate.csv",
        "50million_agents/vm_run_03/analysis/damov/aggregate.csv",
    ],
    "100M": [
        "100million_agents/20260519_101746/vm_run_01/analysis/damov/aggregate.csv",
        "100million_agents/20260531_183949/vm_run_02/analysis/damov/aggregate.csv",
        "100million_agents/20260614_081215/vm_run_03/analysis/damov/aggregate.csv",
    ],
    "200M": [
        "200million_agents/vm_run_01/analysis/damov/aggregate.csv",
        "200million_agents/vm_run_02/analysis/damov/aggregate.csv",
        "200million_agents/vm_run_03/analysis/damov/aggregate.csv",
    ],
}

# DAMOV thresholds for each headline metric (None = no threshold line)
THRESHOLDS = {
    "llc_mpki":      10,
    "mem_bound_frac": 0.30,
    "ipc":           None,
    "lfmr":          0.56,
}

PANEL_META = [
    ("llc_mpki",       "LLC MPKI",               "Misses per kilo-instruction"),
    ("mem_bound_frac", "Memory-bound fraction",   "Fraction of pipeline slots"),
    ("ipc",            "IPC",                     "Instructions per cycle"),
    ("lfmr",           "LFMR",                    "L2_MISS / L1_MISS"),
]

BAR_COLOR  = "#1A5EA8"
ERR_COLOR  = "#0B3D6E"
THRESH_CLR = "#CC2222"

# Color palette for agent scales
SCALE_COLORS = {
    "10M": "#1A5EA8",
    "50M": "#F39C12",
    "100M": "#27AE60",
    "200M": "#C0392B",
}

# Desired scale ordering
SCALE_ORDER = ["10M", "50M", "100M", "200M"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aggregate(csv_path: Path) -> dict:
    """Return {metric: mean_value} from one aggregate.csv file."""
    result = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            result[row["metric"]] = float(row["mean"])
    return result


def collect_scale_data(root: Path) -> dict:
    """
    Returns a dict keyed by scale label, each value being a dict of
    {metric: [mean_run1, mean_run2, ...]} pooled across VM runs.
    """
    def discover_scale_paths(root: Path) -> dict:
        """Discover DAMOV aggregate.csv files under `results/` and group by scale.

        Returns mapping label -> list of relative paths (strings) to aggregate.csv
        """
        results_dir = root / "results"
        mapping_unordered = {}
        if not results_dir.exists():
            return {}

        for scale_dir in results_dir.iterdir():
            if not scale_dir.is_dir():
                continue
            agg_paths = []
            # search for vm_run_* directories recursively
            for vm in scale_dir.rglob("vm_run_*"):
                agg = vm / "analysis" / "damov" / "aggregate.csv"
                if agg.exists():
                    agg_paths.append(str(agg.relative_to(root)))
            # also handle older layouts where vm_run_* may be nested under timestamp dirs
            # rglob above will cover nested cases as well
            if agg_paths:
                name = scale_dir.name
                m = re.search(r"(\d+)", name)
                if m:
                    label = f"{int(m.group(1))}M"
                else:
                    label = name
                mapping_unordered[label] = agg_paths

        # Reorder according to SCALE_ORDER
        mapping = {}
        for label in SCALE_ORDER:
            if label in mapping_unordered:
                mapping[label] = mapping_unordered[label]
        return mapping

    data = {}
    discovered = discover_scale_paths(root)
    if not discovered:
        print("  WARNING: no DAMOV aggregate files discovered under results/")
        return data

    print("Discovered DAMOV aggregate files for scales:")
    for label, agg_paths in discovered.items():
        print(f"  {label}: {len(agg_paths)} files")

    for label, agg_paths in discovered.items():
        per_metric = {}
        for rel_path in agg_paths:
            agg_path = root / rel_path
            if not agg_path.exists():
                print(f"  WARNING: not found – {agg_path}")
                continue
            agg = load_aggregate(agg_path)
            for metric, val in agg.items():
                per_metric.setdefault(metric, []).append(val)
        if per_metric:
            data[label] = per_metric
        else:
            print(f"  WARNING: no DAMOV aggregate data loaded for {label}")
    return data


# ---------------------------------------------------------------------------
# Figure 1 – 4 separate figures (one per metric)
# ---------------------------------------------------------------------------

def plot_fig1(data: dict, out_dir: Path, dpi: int = 300):
    """Generate 4 separate figures, one per metric, with all agent scales colored differently."""
    scales = list(data.keys())
    x = np.arange(len(scales))

    for metric, title, ylabel in PANEL_META:
        # Smaller figsize to reduce whitespace
        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("white")

        # Plot individual VM-run points for each agent scale
        for i, s in enumerate(scales):
            vals = data[s].get(metric, [])
            if not vals:
                continue
            color = SCALE_COLORS.get(s, "#888888")
            # deterministic offsets so points don't overlap
            n = len(vals)
            if n > 1:
                offsets = np.linspace(-0.12, 0.12, n)
            else:
                offsets = [0.0]
            for off, v in zip(offsets, vals):
                ax.scatter(i + off, v, color=color, edgecolor="black",
                           zorder=4, s=48, alpha=0.85, label=s if off == offsets[0] else "")

        # Compute per-scale means
        means = []
        valid_x = []
        valid_means = []
        for i, s in enumerate(scales):
            vals = data[s].get(metric, [])
            if vals:
                m = statistics.mean(vals)
                means.append(m)
                valid_x.append(i)
                valid_means.append(m)
            else:
                means.append(np.nan)

        # Plot mean line connecting all scales
        means_arr = np.array(means, dtype=float)
        ax.plot(x, means_arr, "-o", color="#333333", linewidth=2.0, markersize=7,
                zorder=5, alpha=0.7)

        # Fit and plot linear trend line
        if len(valid_x) >= 2:
            coeff = np.polyfit(valid_x, valid_means, 1)
            fit_y = np.polyval(coeff, x)
            ax.plot(x, fit_y, linestyle="--", color="#666666", linewidth=1.5,
                    zorder=3, alpha=0.6, label=f"trend (slope={coeff[0]:.3g})")

        # Tighten y-axis around actual data range
        all_vals = [v for s in scales for v in data[s].get(metric, [])]
        if all_vals:
            ymin, ymax = min(all_vals), max(all_vals)
            margin = (ymax - ymin) * 0.15
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xticks(x)
        ax.set_xticklabels(scales, fontsize=9)
        ax.set_xlabel("Agent count", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.yaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

        out_path = out_dir / f"fig1_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 – dual-axis MBF / IPC trend
# ---------------------------------------------------------------------------

def plot_fig2(data: dict, out_path: Path, dpi: int = 300):
    scales = list(data.keys())

    mbf_means = [statistics.mean(data[s]["mem_bound_frac"]) for s in scales]
    mbf_errs  = [statistics.stdev(data[s]["mem_bound_frac"]) if len(data[s]["mem_bound_frac"]) > 1 else 0.0
                 for s in scales]
    ipc_means = [statistics.mean(data[s]["ipc"]) for s in scales]
    ipc_errs  = [statistics.stdev(data[s]["ipc"]) if len(data[s]["ipc"]) > 1 else 0.0
                 for s in scales]

    xi = np.arange(len(scales))

    fig, ax1 = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor("white")

    # MBF (left axis)
    ax1.plot(xi, mbf_means, "o-", color="#1A5EA8", linewidth=2, markersize=6,
             label="Memory-bound fraction (left)")
    ax1.fill_between(
        xi,
        [m - e for m, e in zip(mbf_means, mbf_errs)],
        [m + e for m, e in zip(mbf_means, mbf_errs)],
        alpha=0.15, color="#1A5EA8",
    )
    ax1.set_ylabel("Memory-bound fraction", fontsize=10, color="#1A5EA8")
    ax1.tick_params(axis="y", labelcolor="#1A5EA8")
    ax1.set_ylim(0.55, 0.68)
    ax1.set_xticks(xi)
    ax1.set_xticklabels(scales, fontsize=10)
    ax1.set_xlabel("Agent count", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.35)
    ax1.set_axisbelow(True)

    # IPC (right axis)
    ax2 = ax1.twinx()
    ax2.plot(xi, ipc_means, "s--", color="#B85A0D", linewidth=2, markersize=6,
             label="IPC (right)")
    ax2.fill_between(
        xi,
        [m - e for m, e in zip(ipc_means, ipc_errs)],
        [m + e for m, e in zip(ipc_means, ipc_errs)],
        alpha=0.12, color="#B85A0D",
    )
    ax2.set_ylabel("IPC", fontsize=10, color="#B85A0D")
    ax2.tick_params(axis="y", labelcolor="#B85A0D")
    ax2.set_ylim(0.36, 0.50)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    ax1.set_title(
        "Figure 2. Memory-bound fraction and IPC across agent scales\n"
        "(shaded band = \u00b11 std dev across VM runs)",
        fontsize=9, pad=6,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate CARTopiaX ACACES figures.")
    parser.add_argument(
        "--root", default=".",
        help="Path to repository root (directory that contains 'results/'). Default: current directory.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for figures. Default: same as --root.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure DPI. Default: 300.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root : {root}")
    print(f"Output directory: {out_dir}")
    print(f"DPI             : {args.dpi}")
    print()

    print("Loading data...")
    data = collect_scale_data(root)

    for label, metrics in data.items():
        loaded = [m for m, vals in metrics.items() if vals]
        run_count = max((len(vals) for vals in metrics.values()), default=0)
        print(f"  {label}: {len(loaded)} metrics loaded across {run_count} VM runs")

    print()
    print("Generating Figure 1 (4 separate figures)...")
    plot_fig1(data, out_dir, dpi=args.dpi)

    print("Generating Figure 2...")
    plot_fig2(data, out_dir / "fig2_mbf_ipc.png", dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()