"""Which resolution generalizes "first" depends on how generalization is measured.

Reviewer pqmj: finer tokenizations do need more training maps to approach
their OWN error floor, but at a COMMON absolute error they often get there with
fewer maps than coarse tokenizations, whose floor is higher. Both readings come
from the same curves (the runs behind taskdiv_gap); this figure shows them side
by side, plus the summary that reconciles them:

  (a) new-task error against input resolution, one line per task count m.
      The curves are U-shaped and the best resolution moves to finer
      partitions as m grows.
  (b) tasks needed (x) to reach a common target error eps (y), per resolution.
  (c) tasks needed (x) to come within a factor k (y) of the resolution's own
      large-m plateau (the paper's m* is a version of this).
Training tasks are on the x axis throughout, as in the paper's Figure 2.

Rows: cross-entropy (64 output bins) and square loss (scalar output).
Lines are medians over seeds; shading in (b, c) is the seed range of the task
count. Task counts are interpolated linearly in (log m, log error) between the
seven trained m values, taking the first crossing.

Torch-free. Run from the repo root:
  python scripts/rebuttal/taskdiv_resolution_tradeoff.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 16.5, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13,
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "xtick.major.size": 6, "ytick.major.size": 6,
})

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
ARMS = [(8,   "runs_divbins/in8_{o}_m{m}_seed{s}"),
        (16,  "runs_divbins/in16_{o}_m{m}_seed{s}"),
        (32,  "runs_divbins/in32_{o}_m{m}_seed{s}"),
        (64,  "runs_divbins_320k/in64_{o}_m{m}_seed{s}"),
        (128, "runs_divbins_320k/in128_{o}_m{m}_seed{s}"),
        (256, "runs_divbins_320k/in256_{o}_m{m}_seed{s}")]
OBJECTIVES = [("bins", "ce", lambda v: np.nanmean(v), "cross-entropy (nats)",
               "Cross-entropy (64 output bins)"),
              ("mse", "rms", lambda v: np.nanmean(v ** 2), "mean squared error",
               "Square loss (scalar output)")]


def load_curves(otag, stem, agg, seeds):
    """{N: array (n_seeds, len(MS))} of new-task error, complete seeds only."""
    out = {}
    for n, tmpl in ARMS:
        rows = []
        for s in seeds:
            paths = [tmpl.format(o=otag, m=m, s=s) + "/eval_per_r.npz" for m in MS]
            if all(os.path.exists(p) for p in paths):
                rows.append([agg(np.load(p)[f"{stem}_per_r"]) for p in paths])
        if rows:
            out[n] = np.array(rows)
    return out


def tasks_to_reach(curve, eps):
    """Smallest m with error <= eps (log-log interpolation, first crossing).
    Returns MS[0] if already reached at the first point, nan if never."""
    below = np.nonzero(curve <= eps)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(MS[0])
    x0, x1 = np.log(MS[i - 1]), np.log(MS[i])
    y0, y1 = np.log(curve[i - 1]), np.log(curve[i])
    return float(np.exp(x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0)))


def plateau(curve):
    return np.mean(curve[-2:])          # m in {4000, 8000}, as in the paper


def paper_mstar(curve):
    """Half the log-scale decrease from m=100 to the plateau (paper's m*)."""
    target = np.exp(0.5 * (np.log(curve[0]) + np.log(plateau(curve))))
    return tasks_to_reach(curve, target)


def draw_threshold_panel(ax, curves, colors, grid, to_target, ylabel):
    """Training tasks needed (x) to reach each target on the grid (y)."""
    for (n, c), col in zip(curves.items(), colors):
        med = np.median(c, 0)
        x_med = np.array([to_target(med, g) for g in grid])
        x_seeds = np.array([[to_target(cs, g) for g in grid] for cs in c])
        with np.errstate(all="ignore"):
            lo, hi = np.nanmin(x_seeds, 0), np.nanmax(x_seeds, 0)
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.fill_betweenx(grid[ok], lo[ok], hi[ok], color=col, alpha=0.16, lw=0)
        clipped = x_med <= MS[0]
        ax.plot(np.where(clipped, np.nan, x_med), grid, color=col, lw=2.4,
                label=f"{n}")
        ax.plot(np.where(clipped, MS[0], np.nan), grid, color=col, lw=1.4,
                ls=":", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(MS[0] * 0.85, MS[-1] * 1.25)
    ax.set_xlabel("training tasks needed $m$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18, lw=0.5, which="both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_rebuttal/taskdiv_resolution_tradeoff.pdf")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20.5, 12.0))
    res_colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(ARMS)))
    m_colors = plt.cm.Oranges(np.linspace(0.35, 0.95, len(MS)))
    lines = []

    for row, (otag, stem, agg, ylab, title) in enumerate(OBJECTIVES):
        curves = load_curves(otag, stem, agg, a.seeds)
        ns = np.array(list(curves))
        med = np.array([np.median(curves[n], 0) for n in ns])     # (N, m)
        lines.append(f"== {title}  (seeds per arm: "
                     + ", ".join(f"{n}:{len(curves[n])}" for n in ns) + ")")

        # (a) error vs resolution, one line per m; best resolution circled
        ax = axes[row, 0]
        for j, (m, col) in enumerate(zip(MS, m_colors)):
            ax.plot(ns, med[:, j], "-o", color=col, lw=2.0, ms=6, label=f"{m}")
            b = np.argmin(med[:, j])
            ax.plot(ns[b], med[b, j], "o", ms=13, mfc="none", mec="k", mew=1.8)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("input bins $N$")
        ax.set_ylabel(f"new-task {ylab}")
        ax.set_title(f"{title}\nerror vs resolution (circle = best $N$)", fontsize=15)
        ax.grid(alpha=0.18, lw=0.5, which="both")
        best = [int(ns[np.argmin(med[:, j])]) for j in range(len(MS))]
        lines.append("best N by m:  " + "  ".join(f"m={m}:{b}" for m, b in zip(MS, best)))

        # (b) tasks to reach a common absolute error
        floor_min = med[:, -1].min()
        grid = np.geomspace(floor_min * 1.02, med[:, 0].max(), 300)
        draw_threshold_panel(axes[row, 1], curves, res_colors, grid,
                             tasks_to_reach, "common target error $\\epsilon$ (stricter $\\downarrow$)")
        axes[row, 1].set_title("tasks to reach a common error", fontsize=15)

        # (c) tasks to come within factor k of own plateau
        kgrid = np.geomspace(1.05, 20, 300)
        draw_threshold_panel(axes[row, 2], curves, res_colors, kgrid,
                             lambda c, k: tasks_to_reach(c, k * plateau(c)),
                             "within factor $k$ of own plateau (stricter $\\downarrow$)")
        axes[row, 2].set_title("tasks to approach its own floor", fontsize=15)
        ks = [1.1, 1.5, 2, 3, 5, 10, 20]
        axes[row, 2].set_yticks(ks, [f"{k:g}" for k in ks])
        axes[row, 2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

        for n in ns:
            ms_ = [paper_mstar(c) for c in curves[n]]
            lines.append(f"N={n:3d}  plateau {np.median([plateau(c) for c in curves[n]]):.3g}"
                         f"  paper m* median {np.nanmedian(ms_):6.0f}"
                         f"  (seeds {', '.join(f'{v:.0f}' for v in ms_)})")
        for eps in ([1.6, 1.1, 0.5, 0.2, 0.1] if otag == "bins"
                    else [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]):
            vals = [tasks_to_reach(np.median(curves[n], 0), eps) for n in ns]
            lines.append(f"eps={eps:<7g} m needed: " + "  ".join(
                f"{n}:{'never' if np.isnan(v) else ('<=100' if v <= 100 else f'{v:.0f}')}"
                for n, v in zip(ns, vals)))

    axes[0, 0].legend(title="tasks $m$", ncol=2, fontsize=12, title_fontsize=13,
                      loc="lower left", framealpha=0.92)
    axes[0, 1].legend(title="input bins $N$", ncol=2, fontsize=12, title_fontsize=13,
                      loc="upper right", framealpha=0.92)
    for label, ax in zip("abcdef", axes.flat):
        ax.text(-0.13, 1.13, f"({label})", transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="top")
    fig.text(0.5, -0.01, "Dotted: already reached at the smallest trained $m=100$. "
             "Shading: range over seeds.", ha="center", fontsize=13, color="0.35")
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
