#!/usr/bin/env python3
"""
plot_memory.py -- two figures for the memory-footprint finding.
Fit is only plotted/extrapolated to 225M (modest 1.125x beyond the
last real data point at 200M); the 1B OOM result is noted in text,
not plotted, since extrapolating the linear fit 5x past the data
would overstate precision.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = "#0072B2"
ACCENT = "#D55E00"
INK    = "#2C3E50"
GRID   = "#E7E6E1"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
COLW = 3.4

N = np.array([10_000_000, 50_000_000, 100_000_000, 200_000_000], dtype=float)
mem_mb = np.array([4751.1, 14983.4, 27751.2, 55701.8])
mem_bytes = mem_mb * 1024 * 1024
bytes_per_agent = mem_bytes / N

A = np.vstack([np.ones_like(N), N]).T
(C, k), *_ = np.linalg.lstsq(A, mem_bytes, rcond=None)
pred = A @ [C, k]
r2 = 1 - np.sum((mem_bytes - pred)**2) / np.sum((mem_bytes - mem_bytes.mean())**2)

# ============================================================ Fig 1: bytes/agent bar
fig, ax = plt.subplots(figsize=(COLW, 2.5))
labels = ["10M", "50M", "100M", "200M"]
x = np.arange(len(N))
ax.bar(x, bytes_per_agent, color=BASE, edgecolor="none")
ax.axhline(k, color=ACCENT, linestyle="--", linewidth=1.1, zorder=3)
ax.annotate(f"asymptotic $k$ = {k:.0f} B/agent", xy=(3, k),
            xytext=(1.15, k - 65), color=ACCENT, fontsize=7.5, ha="left")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Agent count")
ax.set_ylabel("Peak memory / agent (bytes)")
ax.set_ylim(0, max(bytes_per_agent) * 1.2)
ax.yaxis.grid(True, linewidth=0.6, color=GRID); ax.set_axisbelow(True)
for xi, v in zip(x, bytes_per_agent):
    ax.annotate(f"{v:.0f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=7.5)
fig.tight_layout()
fig.savefig("bytes_per_agent.pdf", dpi=300)
fig.savefig("bytes_per_agent.png", dpi=300)
plt.close(fig)

# ============================================================ Fig 2: memory vs N, linear x, modest extrapolation to 225M
fig, ax = plt.subplots(figsize=(COLW, 2.6))

Nx = np.linspace(0, 240_000_000, 300)
fit_gb = (C + k * Nx) / 1024**3
ax.plot(Nx / 1e6, fit_gb, color=INK, linewidth=1.2, zorder=2, label="linear fit")

ax.scatter(N / 1e6, mem_mb / 1024, s=34, color=BASE, edgecolor="white",
           linewidth=0.6, zorder=4, label="measured")

ax.axhline(64, color=ACCENT, linestyle=":", linewidth=1.0, zorder=1)
ax.annotate("64 GB VM limit", xy=(150, 64), xytext=(0, 4),
            textcoords="offset points", color=ACCENT, fontsize=7.3, ha="left")

oom_225_gb = (C + k * 225_000_000) / 1024**3
ax.scatter([225], [oom_225_gb], marker="x", s=55, color=ACCENT,
           linewidth=1.7, zorder=5, label="OOM-killed (225M)")
ax.annotate("225M:\nOOM-killed", xy=(225, oom_225_gb), xytext=(-58, -6),
            textcoords="offset points", color=ACCENT, fontsize=7.3, ha="left")

ax.set_xlim(0, 240)
ax.set_ylim(0, 70)
ax.set_xlabel("Agent count (millions)")
ax.set_ylabel("Peak memory (GB)")
ax.yaxis.grid(True, linewidth=0.6, color=GRID); ax.set_axisbelow(True)

ax.text(0.03, 0.78, f"$C$={C/1024**3:.2f} GB\n$k$={k:.0f} B/agent\n$R^2$={r2:.4f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.3, color=INK,
        linespacing=1.4)

ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("memory_fit.pdf", dpi=300)
fig.savefig("memory_fit.png", dpi=300)
plt.close(fig)

print(f"C={C/1024**3:.3f} GB, k={k:.2f} B/agent, R2={r2:.5f}")
print(f"Fitted memory at 225M: {oom_225_gb:.1f} GB")
print("done")