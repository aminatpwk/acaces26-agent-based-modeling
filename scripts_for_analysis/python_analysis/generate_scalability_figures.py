"""
generate_scalability_figures.py
--------------------------------

Parses BioDynaMo `metadata` files under `results/*/vm_run_*/output/epidemiology*/`
and produces two standalone (i.e. NOT side-by-side) matplotlib figures
comparing scalability across agent-based simulation runs:

  1. fig_runtime_per_iteration_vs_agents.png
     Runtime per iteration [s]  vs.  Number of agents   (log-log)

  2. fig_memory_vs_agents.png
     Peak memory usage [GB]     vs.  Number of agents   (log-x, linear-y)

Data source
-----------
Each `metadata` file contains lines such as:

    Total simulation runtime	: 1955946 ms
    Peak memory usage (MB)		: 4724.36
    Number of iterations executed	: 3000
    Number of agents		: 10000000

A `metadata` file living directly in `output/epidemiology/` (no trailing
digit) is a wrapper/aggregate file BioDynaMo writes for the whole batch of
repetitions -- it always reports `Number of agents: 0` and
`Number of iterations executed: 0`, so it is skipped automatically.

Repetitions (`epidemiology1`, `epidemiology2`, ...) inside the same
`vm_run_NN` folder are averaged together, giving one data point per
(agent count, VM run) pair. Three VM runs (vm_run_01/02/03) are available
for 10M/50M/100M/200M agents.

225M and 1B agents are included on the x-axis for context, but BOTH of
those sweeps were OOM-killed by the kernel before finishing (see
`results/225_million_agents/sim_logs_terminal` and
`results/1_billion_agents/sim_logs_terminal`) -- no metadata file was ever
written, so there is no runtime/memory data point for them. They are
marked on the plots with a vertical dashed line instead of a data point.

Usage
-----
    python generate_scalability_figures.py
    python generate_scalability_figures.py --root /path/to/repo --out /path/to/outdir --dpi 300
"""

import argparse
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory name (under results/) -> number of agents it represents.
# Order matters: this is also the x-axis category order.
AGENT_SCALE_DIRS = {
    "10_million_agents":  10_000_000,
    "50_million_agents":  50_000_000,
    "100_million_agents": 100_000_000,
    "200_million_agents": 200_000_000,
    "225_million_agents": 225_000_000,
    "1_billion_agents":   1_000_000_000,
}

# Scales that were OOM-killed and never produced a metadata file.
OOM_SCALES = {225_000_000, 1_000_000_000}

# Styling: up to 5 distinct, consistent color/linestyle/marker combos,
# shared between both figures so the same series always looks the same.
SERIES_STYLE = {
    "vm_run_01": dict(color="#1A5EA8", linestyle="-",  marker="o", label="VM run 1"),
    "vm_run_02": dict(color="#F39C12", linestyle="--", marker="s", label="VM run 2"),
    "vm_run_03": dict(color="#27AE60", linestyle="-.", marker="^", label="VM run 3"),
    "mean":      dict(color="#C0392B", linestyle="-",  marker="D", label="Mean across VM runs"),
    "oom":       dict(color="#7F1D1D", linestyle=":",  marker="x", label="OOM-killed (no data)"),
}

METADATA_FIELD_RE = {
    "agents":   re.compile(r"Number of agents\s*:\s*(\d+)"),
    "runtime_ms": re.compile(r"Total simulation runtime\s*:\s*(\d+)\s*ms"),
    "peak_mem_mb": re.compile(r"Peak memory usage \(MB\)\s*:\s*([\d.]+)"),
    "iterations": re.compile(r"Number of iterations executed\s*:\s*(\d+)"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_metadata_file(path: Path) -> dict:
    """Extract the fields we need from one metadata file. Returns {} if the
    file is missing a field or is one of the zero-agent aggregate wrappers."""
    text = path.read_text(errors="ignore")

    values = {}
    for key, pattern in METADATA_FIELD_RE.items():
        m = pattern.search(text)
        if not m:
            return {}
        values[key] = m.group(1)

    agents = int(values["agents"])
    iterations = int(values["iterations"])
    if agents == 0 or iterations == 0:
        # This is the "epidemiology" (no trailing digit) aggregate wrapper
        # file, not an actual repetition run -- skip it.
        return {}

    return {
        "agents": agents,
        "runtime_ms": int(values["runtime_ms"]),
        "iterations": iterations,
        "peak_mem_mb": float(values["peak_mem_mb"]),
    }


def collect_data(root: Path) -> dict:
    """
    Returns:
        {
          agent_count (int): {
              vm_run_name (str): {
                  "runtime_per_iter_s": [float, ...]   # one value per repetition
                  "peak_mem_gb":        [float, ...]
              },
              ...
          },
          ...
        }
    """
    results_dir = root / "results"
    data = {}

    for scale_dir_name, agent_count in AGENT_SCALE_DIRS.items():
        scale_dir = results_dir / scale_dir_name
        if not scale_dir.exists():
            continue

        per_vm_run = {}
        for vm_run_dir in sorted(scale_dir.glob("vm_run_*")):
            metadata_files = sorted(vm_run_dir.glob("output/epidemiology*/metadata"))
            runtimes_per_iter, mems_gb = [], []

            for meta_path in metadata_files:
                parsed = parse_metadata_file(meta_path)
                if not parsed:
                    continue
                runtimes_per_iter.append(parsed["runtime_ms"] / parsed["iterations"] / 1000.0)
                mems_gb.append(parsed["peak_mem_mb"] / 1024.0)

            if runtimes_per_iter:
                per_vm_run[vm_run_dir.name] = {
                    "runtime_per_iter_s": runtimes_per_iter,
                    "peak_mem_gb": mems_gb,
                }

        if per_vm_run:
            data[agent_count] = per_vm_run

    return data


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def agent_count_formatter(x, _pos=None):
    if x >= 1_000_000_000:
        return f"{x / 1_000_000_000:g}B"
    if x >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    return f"{x:g}"


def _setup_axes(ax, y_log: bool):
    ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    all_ticks = sorted(AGENT_SCALE_DIRS.values())
    ax.set_xticks(all_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(agent_count_formatter))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Number of agents", fontsize=11)
    # 200M and 225M sit very close together on a log axis -- rotate labels
    # so they don't overlap.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def _mark_oom_scales(ax, label_y_frac=0.92):
    """Draw a vertical dashed line + 'OOM' annotation for scales that never
    produced a data point (225M, 1B agents)."""
    style = SERIES_STYLE["oom"]
    ymin, ymax = ax.get_ylim()
    y_text = ymin * (ymax / ymin) ** label_y_frac if ax.get_yscale() == "log" \
        else ymin + (ymax - ymin) * label_y_frac

    first = True
    for agent_count in sorted(OOM_SCALES):
        ax.axvline(agent_count, color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.4, alpha=0.8, zorder=2,
                    label=style["label"] if first else None)
        ax.text(agent_count, y_text, "OOM", color=style["color"], fontsize=8,
                ha="center", va="top", rotation=90, alpha=0.9)
        first = False


def _plot_series(ax, data: dict, metric_key: str):
    """Plot one line per VM run plus a bold mean line for `metric_key`
    ('runtime_per_iter_s' or 'peak_mem_gb')."""
    vm_run_names = sorted({vm for per_vm in data.values() for vm in per_vm})

    # Per-VM-run lines
    for vm_run in vm_run_names:
        style = SERIES_STYLE.get(vm_run, SERIES_STYLE["vm_run_01"])
        xs, ys = [], []
        for agent_count in sorted(data.keys()):
            per_vm = data[agent_count].get(vm_run)
            if per_vm and per_vm[metric_key]:
                xs.append(agent_count)
                ys.append(statistics.mean(per_vm[metric_key]))
        if xs:
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=6, linewidth=1.6,
                    alpha=0.85, label=style["label"], zorder=3)

    # Bold mean-across-VM-runs line
    style = SERIES_STYLE["mean"]
    xs, ys = [], []
    for agent_count in sorted(data.keys()):
        vals = [statistics.mean(per_vm[metric_key])
                for per_vm in data[agent_count].values() if per_vm[metric_key]]
        if vals:
            xs.append(agent_count)
            ys.append(statistics.mean(vals))
    if xs:
        ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=7, linewidth=2.4,
                alpha=0.95, label=style["label"], zorder=4)


# ---------------------------------------------------------------------------
# Figure 1: Runtime per iteration vs Number of agents (log-log)
# ---------------------------------------------------------------------------

def plot_runtime_figure(data: dict, out_path: Path, dpi: int = 300):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _plot_series(ax, data, "runtime_per_iter_s")
    _setup_axes(ax, y_log=True)
    _mark_oom_scales(ax)

    ax.set_ylabel("Runtime per iteration [s]", fontsize=11)
    ax.set_title("Runtime per Iteration vs Number of Agents", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Memory consumption vs Number of agents (log-x, linear-y)
# ---------------------------------------------------------------------------

def plot_memory_figure(data: dict, out_path: Path, dpi: int = 300):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _plot_series(ax, data, "peak_mem_gb")
    _setup_axes(ax, y_log=False)
    _mark_oom_scales(ax)

    ax.set_ylabel("Memory consumption [GB]", fontsize=11)
    ax.set_title("Memory Consumption vs Number of Agents", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate scalability figures (runtime/iteration and memory vs agent count)."
    )
    parser.add_argument("--root", default=".",
                         help="Path to repository root (directory containing 'results/'). Default: current directory.")
    parser.add_argument("--out", default=None,
                         help="Output directory for figures. Default: <root>/poster/figures.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default: 300.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root / "poster" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository root : {root}")
    print(f"Output directory: {out_dir}")
    print(f"DPI             : {args.dpi}")
    print()

    print("Parsing metadata files under results/ ...")
    data = collect_data(root)

    for agent_count in sorted(data.keys()):
        n_runs = len(data[agent_count])
        n_reps = sum(len(v["runtime_per_iter_s"]) for v in data[agent_count].values())
        print(f"  {agent_count_formatter(agent_count)}: {n_runs} VM run(s), {n_reps} repetition(s) total")

    missing = sorted(a for a in AGENT_SCALE_DIRS.values() if a not in data)
    if missing:
        print(f"  (no metadata found for: {', '.join(agent_count_formatter(a) for a in missing)} "
              f"-- OOM-killed before completion, marked as dashed lines on the plots)")

    if not data:
        print("ERROR: no usable metadata files found under results/. Nothing to plot.")
        return

    print()
    print("Generating Figure 1: Runtime per Iteration vs Number of Agents ...")
    plot_runtime_figure(data, out_dir / "fig_runtime_per_iteration_vs_agents.png", dpi=args.dpi)

    print("Generating Figure 2: Memory Consumption vs Number of Agents ...")
    plot_memory_figure(data, out_dir / "fig_memory_vs_agents.png", dpi=args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
