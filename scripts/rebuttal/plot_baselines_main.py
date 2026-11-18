"""Main-text baselines figure (cross-entropy only).

(a) the 6-gram Markov predictor at every input resolution (new tasks), one
    colour per resolution, same viridis scale as taskdiv_gap -- compare with
    Fig. 2a, the transformer at every resolution.
(b, c) the cross-entropy row of baselines_taskdiv at 64 and 256 input bins,
    unchanged: transformer new tasks (solid, seed range shaded), seen tasks
    (dashed), and the k-gram predictors.

Medians over seeds 0-3; same data and evaluation contexts as
plot_baselines_taskdiv.py. Torch-free:
  python scripts/rebuttal/plot_baselines_main.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_baselines_taskdiv import KS, K_COLORS, K_MARKERS, MS, collect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ns_all", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--ns_detail", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--bl_dir", default="runs_rebuttal_baselines")
    ap.add_argument("--out", default="figures_rebuttal/baselines_main.pdf")
    a = ap.parse_args()

    data = {n: collect(n, a.seeds, a.bl_dir) for n in set(a.ns_all) | set(a.ns_detail)}
    ncol = 1 + len(a.ns_detail)
    fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol + 1, 5.6))

    # (a) the k-gram predictor at every resolution
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(a.ns_all)))
    for n, c in zip(a.ns_all, colors):
        ax.plot(MS, np.median(data[n][f"k{a.k}_ce"], 0), "-D", color=c, lw=2.2,
                ms=5, label=str(n))
    ax.set_title(f"{a.k}-gram Markov predictor")
    ax.legend(title="input bins", ncol=2, fontsize=11.5, title_fontsize=12,
              frameon=False, loc="upper right")

    # (b, c) unchanged cross-entropy panels of baselines_taskdiv
    for ax, n in zip(axes[1:], a.ns_detail):
        d = data[n]
        tf = d["tf_ce_new"]
        ax.fill_between(MS, tf.min(0), tf.max(0), color="0.15", alpha=0.15, lw=0)
        ax.plot(MS, np.median(tf, 0), "-o", color="0.1", lw=2.6, ms=6,
                label="transformer, new tasks", zorder=5)
        ax.plot(MS, np.median(d["tf_ce_seen"], 0), "--", color="0.1", lw=1.6,
                label="transformer, seen tasks", zorder=4)
        for k in KS:
            ax.plot(MS, np.median(d[f"k{k}_ce"], 0), linestyle="-",
                    marker=K_MARKERS[k], color=K_COLORS[k], lw=2.0, ms=5,
                    label=f"{k}-gram (last {k} token{'s' if k > 1 else ''})")
        ax.set_title(f"{n} input bins")

    for i, ax in enumerate(axes):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(alpha=0.18, lw=0.5, which="both")
        ax.set_xlabel(r"training tasks $m$")
        ax.text(-0.13, 1.06, f"({'abcdefgh'[i]})", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")
    axes[0].set_ylabel("new-task cross-entropy (nats)")

    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.6, -0.16),
               framealpha=0.95)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}, {png}")


if __name__ == "__main__":
    main()
