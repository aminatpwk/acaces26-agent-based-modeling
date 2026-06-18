#!/usr/bin/env python3
"""
split_perf.py — Stream-split a perf stat file into one file per repetition.

Usage:
    python3 split_perf.py <perf_file> [--output-dir <dir>]

The script never loads the full file into memory: it reads one line at a time
and writes directly to the current repetition's output file.
"""

import argparse
import math
import os
import sys

# ---------------------------------------------------------------------------
# Repetition durations (milliseconds) — edit if your timings change
# ---------------------------------------------------------------------------
DURATIONS_MS = [
    32637559,   # epidemiology_1
    31934606,   # epidemiology_2
    32139954,   # epidemiology_3
]

EVENTS_PER_SAMPLE = 13   # fixed number of perf event lines per 1-second sample
HEADER_LINES      = 3    # "# started on ..." + blank + "# time counts ..."


def durations_to_line_counts(durations_ms, events_per_sample):
    """Convert each repetition's duration in ms to a line count."""
    counts = []
    for ms in durations_ms:
        samples = math.floor(ms / 1000)
        counts.append(samples * events_per_sample)
    return counts


def split(perf_path: str, output_dir: str):
    line_counts = durations_to_line_counts(DURATIONS_MS, EVENTS_PER_SAMPLE)

    total_reps = len(DURATIONS_MS)
    os.makedirs(output_dir, exist_ok=True)

    # Print plan
    print(f"Input file : {perf_path}")
    print(f"Output dir : {output_dir}")
    print(f"Header lines to skip: {HEADER_LINES}")
    print()
    for i, (ms, lc) in enumerate(zip(DURATIONS_MS, line_counts), 1):
        samples = math.floor(ms / 1000)
        print(f"  Rep {i:>2} (epidemiology_{i}): {ms:>12,} ms  →  "
              f"{samples:>6,} samples  →  {lc:>8,} lines")
    print(f"\n  Total data lines expected: {sum(line_counts):,}")
    print()

    rep_idx        = 0          # which repetition we are writing (0-based)
    data_line_no   = 0          # lines written for the current repetition
    total_line_no  = 0          # absolute line counter (across whole file)
    out_fh         = None
    skipped_header = 0

    def open_rep(idx):
        path = os.path.join(output_dir, f"epidemiology_{idx + 1}.perf")
        print(f"  → Opening {path}")
        return open(path, "w", buffering=1 << 20)   # 1 MB write buffer

    with open(perf_path, "r", buffering=1 << 20) as fh:
        for raw_line in fh:
            total_line_no += 1

            # --- skip file-level header lines ---
            if skipped_header < HEADER_LINES:
                skipped_header += 1
                continue

            # --- skip inline comment lines (e.g. repeated column headers) ---
            if raw_line.lstrip().startswith("#"):
                continue

            # --- open first output file ---
            if out_fh is None:
                out_fh = open_rep(rep_idx)

            # --- write line to current rep file ---
            out_fh.write(raw_line)
            data_line_no += 1

            # --- check if this rep is complete ---
            if data_line_no >= line_counts[rep_idx]:
                out_fh.close()
                print(f"  ✓ epidemiology_{rep_idx + 1} done "
                      f"({data_line_no:,} lines written)")
                rep_idx      += 1
                data_line_no  = 0

                if rep_idx < total_reps:
                    out_fh = open_rep(rep_idx)
                else:
                    out_fh = None
                    break   # all repetitions written; stop reading

    # Close any still-open handle (e.g. last rep or early EOF)
    if out_fh is not None:
        out_fh.close()
        print(f"  ✓ epidemiology_{rep_idx + 1} done "
              f"({data_line_no:,} lines written)  [file ended]")

    print(f"\nDone. Total file lines read: {total_line_no:,}")

    # Warn if the file had fewer lines than expected
    expected_total = sum(line_counts) + HEADER_LINES
    if total_line_no < expected_total:
        shortfall = expected_total - total_line_no
        print(f"  ⚠  File was {shortfall:,} lines shorter than expected. "
              f"Last repetition may be incomplete.")


def main():
    parser = argparse.ArgumentParser(description="Split a perf stat file by repetition.")
    parser.add_argument("perf_file", help="Path to the perf stat output file")
    parser.add_argument("--output-dir", default="perf_splits",
                        help="Directory for output files (default: ./perf_splits)")
    args = parser.parse_args()

    if not os.path.isfile(args.perf_file):
        print(f"Error: file not found: {args.perf_file}", file=sys.stderr)
        sys.exit(1)

    split(args.perf_file, args.output_dir)


if __name__ == "__main__":
    main()