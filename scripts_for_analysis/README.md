# Perf-split analysis: 13 metrics × 10 repetitions

Scripts that parse the 10 `epidemiology_*.perf` files in `../perf_splits/` and
quantify variation **within** each rep (over time) and **across** reps.

## Run

```bash
./run_analysis.sh
```

This calls, in order:

1. `parse_perf.py` — streams each `.perf` file (~1.7M lines, 200 MB) and
   writes a wide Parquet to `parquet/rep_NN.parquet`. One row per timestamp,
   columns: `ts`, then for each of 13 events `<event>` (raw count) and
   `<event>__cov` (perf time-multiplexing coverage %).
2. `analyze_within.py` — for each rep × metric: descriptive stats
   (mean/median/CV/percentiles), lag-1 autocorrelation, linear trend
   per second, CV of 60 s and 600 s rolling means. Also produces 6 derived
   metrics (IPC, L1/L2 miss rate, MemBound fraction, stalls/cycle) and a
   one-page time-series figure per rep.
3. `analyze_across.py` — wide table of per-rep means and stds, plus an
   aggregate with CV-of-rep-means, mean-within-rep-CV, the ratio of those
   two (between-vs-within), one-way ANOVA (η², F, p) and Kruskal–Wallis
   (downsampled to 5 k samples / rep). Plots: boxplots per rep and a strip
   plot of per-rep means with ±2σ band.

## Metrics

Raw events (in file order):

| # | Event |
|---|-------|
| 1 | `instructions` |
| 2 | `cycles` |
| 3 | `L2_RQSTS.MISS` |
| 4 | `L2_RQSTS.REFERENCES` |
| 5 | `MEM_LOAD_COMPLETED.L1_MISS_ANY` |
| 6 | `MEM_LOAD_RETIRED.L1_HIT` |
| 7 | `MEM_LOAD_RETIRED.L2_HIT` |
| 8 | `MEM_LOAD_RETIRED.L1_MISS` |
| 9 | `MEM_LOAD_RETIRED.L2_MISS` |
| 10 | `TOPDOWN.SLOTS_P` |
| 11 | `TOPDOWN.MEMORY_BOUND_SLOTS` |
| 12 | `MEMORY_ACTIVITY.STALLS_L1D_MISS` |
| 13 | `MEMORY_ACTIVITY.STALLS_L2_MISS` |

Derived (computed in both within / across):

* `IPC = instructions / cycles`
* `L1_miss_rate = L1_MISS / (L1_HIT + L1_MISS)`
* `L2_miss_rate = L2_RQSTS.MISS / L2_RQSTS.REFERENCES`
* `MemBound_frac = MEMORY_BOUND_SLOTS / SLOTS_P`
* `StallL1D_per_cyc`, `StallL2_per_cyc`

## Notes

* Perf time-multiplexes events; each row's last column is the coverage %.
  `--cov-adjust` on `analyze_across.py` scales counts by `100/coverage`
  before any stats (the raw run does not, since ratios already cancel).
* Each .perf file (one rep) is ~36 h of 1 Hz samples — ≈ 132 k ticks × 13
  events. ANOVA uses full data; Kruskal–Wallis is downsampled to keep
  the rank computation tractable.
* `eta_sq` is the ANOVA fraction of variance explained by *rep identity*.
  With ~1.3 M points per metric, ANOVA / Kruskal p-values are essentially
  always ~0 — use `eta_sq`, `ratio_between_within`, and `range_pct_of_mean`
  for effect-size judgment, not the p-values.

## Output tree

```
analysis/
  parquet/
    rep_NN.parquet            # one per rep
    rep_summary.csv           # rep, file, ticks, ts_start, ts_end, duration_s
  within/
    summary_raw.csv           # rep x event raw stats
    summary_adj.csv           # coverage-adjusted
    summary_derived.csv       # derived metrics
    timeseries_plots/
      rep_NN_timeseries.png   # 19-panel time series, x = hours since rep start
  across/
    per_rep_means.csv         # wide: rep x metric -> mean
    per_rep_stds.csv          # wide: rep x metric -> std
    aggregate.csv             # per metric: between/within CV, eta_sq, F, p
    boxplots.png              # 19 panels, 10 boxes per panel
    rep_means_strip.png       # per-rep means with ±2σ band
```
