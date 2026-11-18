"""Where does the crossover happen for the transformer vs the 6-gram counter?

One panel per objective (cross-entropy left, square loss right). Each input
resolution is one colour (same viridis scale as taskdiv_gap): transformer
new-task error solid, 6-gram counter new-task error dotted. The summary
file lists the crossover m* of each curve (log-scale midpoint between E(100)
and the plateau mean(E(4000), E(8000)), as defined in the paper); m* is left
undefined when E(100)/plateau < 2.

Medians over seeds 0-3, same data and evaluation contexts as
plot_baselines_taskdiv.py. Torch-free:
  python scripts/rebuttal/plot_baselines_crossover.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_baselines_taskdiv import MS, collect  # noqa: E402

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})
LM = np.log10(MS)


def mstar(c, min_drop=2.0):
    """(m*, error at m*) or (None, None)."""
    c = np.asarray(c, float)
    plateau = c[-2:].mean()
    if c[0] / plateau < min_drop:
        return None, None
    h = np.sqrt(c[0] * plateau)
    for i in range(len(c) - 1):
        if c[i] >= h >= c[i + 1]:
            f = (np.log(c[i]) - np.log(h)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LM[i] + f * (LM[i + 1] - LM[i])), h
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--bl_dir", default="runs_rebuttal_baselines")
    ap.add_argument("--out", default="figures_rebuttal/baselines_6gram_crossover.pdf")
    a = ap.parse_args()

    colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(a.ns)))
    data = {n: collect(n, a.seeds, a.bl_dir) for n in a.ns}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    lines = [f"m* (median curves over seeds {a.seeds}); '-' = undefined (drop < 2x)",
             f"{'obj':4s} {'N':>4s}  {'transformer':>11s}  {f'{a.k}-gram':>9s}"]
    for ax, (obj, title, ylab) in zip(axes, [
            ("ce", "Cross-entropy", "new-task cross-entropy (nats)"),
            ("mse", "Square loss", "new-task mean squared error")]):
        for n, c in zip(a.ns, colors):
            d = data[n]
            tf = np.median(d[f"tf_{obj}_new"], 0)
            kg = np.median(d[f"k{a.k}_{obj}"], 0)
            ax.plot(MS, tf, "-o", color=c, lw=2.4, ms=5.5, zorder=4)
            ax.plot(MS, kg, ":", marker="D", mfc="white", color=c, lw=2.0, ms=5,
                    zorder=3)
            mt, yt = mstar(tf)
            mk, yk = mstar(kg)
            fmt = lambda v: f"{v:.0f}" if v else "-"
            lines.append(f"{obj:4s} {n:>4d}  {fmt(mt):>11s}  {fmt(mk):>9s}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(alpha=0.18, lw=0.5, which="both")
        ax.set_title(title)
        ax.set_xlabel(r"training tasks $m$")
        ax.set_ylabel(ylab)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    res = [plt.Line2D([], [], color=c, lw=2.4) for c in colors]
    sty = [plt.Line2D([], [], color="0.25", ls="-", marker="o", lw=2.4, ms=5.5),
           plt.Line2D([], [], color="0.25", ls=":", marker="D", mfc="white", lw=2.0, ms=5)]
    fig.legend(res + sty,
               [f"{n} bins" for n in a.ns]
               + ["transformer", f"{a.k}-gram counter (last {a.k} tokens)"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13),
               frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    summary = os.path.splitext(a.out)[0] + "_summary.txt"
    with open(summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {a.out}, {png}, {summary}")


if __name__ == "__main__":
    main()
