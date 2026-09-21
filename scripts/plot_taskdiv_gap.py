"""Task-diversity overlay with seen and new tasks on shared axes.

One panel per training objective instead of a 2x2 grid. Within a panel each
input resolution gets one colour, with

    solid  = new tasks   (300-point grid spanning the full parameter range)
    dashed = seen tasks  (fresh trajectories at training parameter values)

The seen-task curves are nearly flat, so they read as the floor that the
new-task curve descends toward; the vertical gap between a matched pair is the
seen--new gap, and it closes as task diversity grows. That gap is the quantity
of interest, and putting the rows on top of each other makes it directly
legible rather than something to be inferred across panels.

Aggregation note: the stored arrays are per-r RMS values. Mean squared error is
therefore mean(rms**2), NOT mean(rms)**2 -- aggregating the RMS first and then
squaring is a different (and smaller) quantity.

Arms are read at a converged checkpoint: 160k steps for 8-32 bins, 320k for
64-256 (see plot_taskdiv_converged.py for the convergence analysis).

Torch-free. Run from the repo root:
  python scripts/plot_taskdiv_gap.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16.5,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
TOTAL_TRAJ = 32_000

ARMS = [(8,   "runs_divbins/in8_{o}_m{m}_seed{s}",       160),
        (16,  "runs_divbins/in16_{o}_m{m}_seed{s}",      160),
        (32,  "runs_divbins/in32_{o}_m{m}_seed{s}",      160),
        (64,  "runs_divbins_320k/in64_{o}_m{m}_seed{s}", 320),
        (128, "runs_divbins_320k/in128_{o}_m{m}_seed{s}", 320),
        (256, "runs_divbins_320k/in256_{o}_m{m}_seed{s}", 320)]

# (output tag, panel title, y label, per-r key stem, aggregator)
PANELS = [
    ("bins", "Cross-entropy loss (64 output bins)",
     "mean cross-entropy (nats)", "ce", lambda v: np.nanmean(v)),
    ("mse",  "Square loss (scalar output)",
     "mean squared error", "rms", lambda v: np.nanmean(v ** 2)),
]


def load(tmpl, otag, m, key, seed=0):
    p = tmpl.format(o=otag, m=m, s=seed) + "/eval_per_r.npz"
    if not os.path.exists(p):
        return None
    return np.load(p)[key]


# Seeds are drawn individually rather than as a band. Under the square loss the
# run-to-run spread is about 1x away from each arm's transition and 52x to 88x
# at it, because near the threshold a run either generalises or does not; a
# filled range there would merge two populations into one region and the median
# would report whichever branch had the majority. Under cross-entropy the spread
# never exceeds 2x and a band would have been safe, but one convention across
# both panels is easier to read than two.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_taskdiv/taskdiv_gap.pdf")
    ap.add_argument("--traj_axis", action="store_true",
                    help="secondary axis: ALLOCATED trajectories per task. A "
                         "validation split is drawn from the same budget "
                         "(val_frac 0.15, capped at max_val_traj=600), so "
                         "31,400 of the 32,000 reach the optimizer.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="seeds to overlay as thin lines; the median is heavy")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(ARMS)))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.7))

    for ax, (otag, title, ylab, stem, agg) in zip(axes, PANELS):
        for (nbin, tmpl, _steps), c in zip(ARMS, colors):
            per_seed_new, per_seed_seen = [], []
            for sd in a.seeds:
                new = [load(tmpl, otag, m, f"{stem}_per_r", sd) for m in MS]
                seen = [load(tmpl, otag, m, f"{stem}_at_train_r", sd) for m in MS]
                if any(v is None for v in new + seen):
                    continue
                per_seed_new.append([agg(v) for v in new])
                per_seed_seen.append([agg(v) for v in seen])
            if not per_seed_new:
                print(f"  [skip] {nbin} bins / {otag}: missing runs")
                continue
            yn = np.array(per_seed_new); ys = np.array(per_seed_seen)
            for row in yn:
                ax.plot(MS, row, "-", lw=0.8, color=c, alpha=0.45, zorder=2)
            ax.plot(MS, np.median(yn, 0), "-o", ms=5.8, lw=2.4, color=c,
                    zorder=3, label=f"{nbin}")
            ax.plot(MS, np.median(ys, 0), "--", lw=1.7, color=c, alpha=0.65,
                    zorder=2)
            print(f"  {nbin:>4} bins / {otag}: {len(yn)} seeds, "
                  f"max spread {np.max(yn.max(0) / yn.min(0)):.0f}x")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"number of training tasks $m$ (distinct $r$ values)")
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=46)
        ax.grid(alpha=0.18, lw=0.5, which="both")

        if otag == "bins":   # uniform-prediction reference
            ax.axhline(np.log(64), color="0.35", lw=1.0, ls=":", zorder=1)
            ax.text(MS[0] * 1.05, np.log(64) * 1.14,
                    r"uniform prediction, $\log 64$", fontsize=14, color="0.35")

        if a.traj_axis:
            sec = ax.secondary_xaxis("top")
            sec.set_xticks(MS)
            sec.set_xticklabels([str(TOTAL_TRAJ // int(m)) for m in MS], fontsize=13.5)
            sec.set_xlabel("allocated trajectories per task", fontsize=15)

    # one shared legend: colour = resolution, style = task set
    h, l = axes[0].get_legend_handles_labels()
    leg1 = axes[0].legend(h, l, title="input bins", title_fontsize=14.5,
                          loc="lower left", ncol=2, framealpha=0.92)
    axes[0].add_artist(leg1)
    style = [plt.Line2D([], [], color="0.25", ls="-", lw=2.4, marker="o", ms=5.8),
             plt.Line2D([], [], color="0.25", ls="--", lw=1.7)]
    axes[1].legend(style, ["new tasks", "seen tasks"],
                   loc="lower left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
