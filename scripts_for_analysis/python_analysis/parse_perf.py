#!/usr/bin/env python3
"""Stream-parse epidemiology_*.perf files into per-rep Parquet.

Input format (one row per (timestamp, event)):
    <ts_seconds>  <count>  <EVENT_NAME>  [# <derived metric>]  (<coverage>%)

Per timestamp tick there are exactly 13 events in a fixed order.
Output: one Parquet per rep with columns
    ts, <event> (value), <event>__cov (coverage %)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
N = len(EVENTS)

COV_RE = re.compile(r"\(([\d.]+)%\)")


def parse_file(path: Path) -> pd.DataFrame:
    """Parse one .perf file into a wide DataFrame."""
    # Pre-allocate arrays
    # First pass: count complete ticks (lines / 13)
    with path.open("rb") as fh:
        nlines = sum(1 for _ in fh)
    nticks = nlines // N
    if nlines % N != 0:
        print(
            f"  WARN: {path.name} has {nlines} lines, not divisible by {N}. "
            f"Truncating to {nticks * N}.",
            file=sys.stderr,
        )

    ts_arr = np.empty(nticks, dtype=np.float64)
    val_arrs = [np.empty(nticks, dtype=np.float64) for _ in range(N)]
    cov_arrs = [np.empty(nticks, dtype=np.float32) for _ in range(N)]

    i = 0  # tick index
    j = 0  # within-tick line index (0..12)
    with path.open("r") as fh:
        for raw in fh:
            parts = raw.split()
            if len(parts) < 4:
                continue
            ts = float(parts[0])
            try:
                value = float(parts[1])
            except ValueError:
                # <not counted> or similar — treat as NaN
                value = np.nan
            event = parts[2]
            # coverage is always the last (XX.XX%) token
            m = COV_RE.match(parts[-1])
            cov = float(m.group(1)) if m else np.nan

            # Validate event order: should equal EVENTS[j]
            if event != EVENTS[j]:
                # Recover: search for event in EVENTS to find correct index.
                if event in EVENTS:
                    j = EVENTS.index(event)
                else:
                    # Unknown event: skip the line
                    continue

            if j == 0:
                if i >= nticks:
                    break
                ts_arr[i] = ts
            val_arrs[j][i] = value
            cov_arrs[j][i] = cov

            j += 1
            if j == N:
                j = 0
                i += 1
                if i >= nticks:
                    break

    # If file ended mid-tick, truncate
    nfinal = i
    cols = {"ts": ts_arr[:nfinal]}
    for k, ev in enumerate(EVENTS):
        cols[ev] = val_arrs[k][:nfinal]
        cols[f"{ev}__cov"] = cov_arrs[k][:nfinal]
    return pd.DataFrame(cols)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-dir",
        default="/home/toros/abm/perf_results/epidemiology/benchmark_results/"
        "20260425_145215/perf_splits",
        help="Directory containing epidemiology_*.perf",
    )
    p.add_argument(
        "--out-dir",
        default="/home/toros/abm/perf_results/epidemiology/benchmark_results/"
        "20260425_145215/analysis/parquet",
        help="Output directory for parquet files",
    )
    p.add_argument(
        "--reps", type=int, nargs="*", default=None,
        help="Optional subset of rep numbers (e.g. --reps 1 2). Default: all.",
    )
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("epidemiology_*.perf"),
                   key=lambda f: int(f.stem.split("_")[1]))
    if not files:
        print(f"No epidemiology_*.perf in {in_dir}", file=sys.stderr)
        return 1

    if args.reps:
        files = [f for f in files if int(f.stem.split("_")[1]) in args.reps]

    summary_rows = []
    for f in files:
        rep = int(f.stem.split("_")[1])
        t0 = time.time()
        df = parse_file(f)
        elapsed = time.time() - t0
        out = out_dir / f"rep_{rep:02d}.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        print(
            f"  rep {rep:2d}: {len(df):>7d} ticks, "
            f"ts {df['ts'].iloc[0]:.1f}..{df['ts'].iloc[-1]:.1f}s, "
            f"{elapsed:5.1f}s -> {out.name}"
        )
        summary_rows.append({
            "rep": rep,
            "file": f.name,
            "ticks": len(df),
            "ts_start": float(df["ts"].iloc[0]),
            "ts_end": float(df["ts"].iloc[-1]),
            "duration_s": float(df["ts"].iloc[-1] - df["ts"].iloc[0]),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "rep_summary.csv", index=False)
    print(f"\nWrote {len(summary_rows)} reps to {out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
