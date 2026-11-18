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
  python scripts/plot_taskdiv_gap.py --regime_panel \
      --out figures_taskdiv/taskdiv_gap_regime.pdf     # adds panel (c)

--regime_panel adds (c): the cross-entropy crossover m* against input
resolution, separately for fixed-point, periodic and chaotic-band maps (new-task
average restricted to the evaluation-grid points in each regime) and for the
whole family (black). m* of the median curve over seeds; bars span the per-seed
m*. Same definition and numbers as scripts/rebuttal/plot_taskdiv_by_regime.py.
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


LOG_MS = np.log10(MS)
REGIMES = [("fixed point", r"$0.5\leq r<3$", 0.5, 3.0, "#0072B2"),
           ("periodic", r"$3\leq r<3.57$", 3.0, 3.5699, "#E69F00"),
           ("chaotic band", r"$r\geq3.57$", 3.5699, 4.01, "#CC79A7")]


def mstar(c, min_drop=2.0):
    """Log-scale midpoint between E(100) and mean(E(4000), E(8000))."""
    c = np.asarray(c, float)
    plateau = c[-2:].mean()
    if c[0] / plateau < min_drop:
        return None
    h = np.sqrt(c[0] * plateau)
    for i in range(len(c) - 1):
        if c[i] >= h >= c[i + 1]:
            f = (np.log(c[i]) - np.log(h)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LOG_MS[i] + f * (LOG_MS[i + 1] - LOG_MS[i]))
    return None


def regime_mstar(seeds, otag, stem, agg):
    """{name: (bins, m*, lower err, upper err)} per regime and for all maps."""
    series = {name: ([], [], [], []) for name, *_ in REGIMES + [("all r", "", 0, 0, "k")]}
    for nbin, tmpl, _steps in ARMS:
        curves = {name: [] for name in series}
        for sd in seeds:
            rows = {name: [] for name in series}
            for m in MS:
                v = load(tmpl, otag, m, f"{stem}_per_r", sd)
                r = load(tmpl, otag, m, "r_grid", sd)
                if v is None:
                    break
                rows["all r"].append(agg(v))
                for name, _, lo, hi, _ in REGIMES:
                    rows[name].append(agg(v[(r >= lo) & (r < hi)]))
            if len(rows["all r"]) == len(MS):
                for name in series:
                    curves[name].append(rows[name])
        for name, (xs, ys, lo, hi) in series.items():
            if not curves[name]:
                continue
            med = mstar(np.median(curves[name], 0))
            per = [v for v in (mstar(c) for c in curves[name]) if v]
            if med:
                xs.append(nbin); ys.append(med)
                lo.append(max(med - min(per), 0) if per else 0)
                hi.append(max(max(per) - med, 0) if per else 0)
    return series


def regime_panel(ax, seeds, objectives, title, legend=True):
    """m* against input bins, per regime and overall, for one or both losses
    (cross-entropy solid, square loss dashed when both are drawn)."""
    both = len(objectives) == 2
    for otag, stem, agg in objectives:
        series = regime_mstar(seeds, otag, stem, agg)
        ls = "--" if (both and otag == "mse") else "-"
        for name, rng, _, _, col in REGIMES + [("all r", "", 0, 0, "k")]:
            xs, ys, lo, hi = series[name]
            overall = name == "all r"
            eb = ax.errorbar(xs, ys, yerr=[lo, hi], color=col, lw=2.6 if overall else 2.0,
                             marker="o" if overall else "s", ms=7, capsize=3,
                             elinewidth=1.1, zorder=5 if overall else 3,
                             mfc="white" if ls == "--" else col,
                             label=(("all maps" if overall else f"{name}, {rng}")
                                    if ls == "-" else None))
            eb[-1][0].set_linestyle(ls) if eb[-1] else None
            eb[0].set_linestyle(ls)
            print(f"  m* {otag:>4s} {name:>12s}: " + ", ".join(f"{x}:{y:.0f}" for x, y in zip(xs, ys)))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([a for a, *_ in ARMS]); ax.set_xticklabels([str(a) for a, *_ in ARMS])
    ax.set_yticks([200, 400, 800, 1600]); ax.set_yticklabels(["200", "400", "800", "1600"])
    ax.set_ylim(130, 2400)              # shared across regime panels
    ax.minorticks_off()
    ax.set_xlabel("input bins $N$")
    ax.set_ylabel(r"crossover task count $m^*$")
    ax.set_title(title, pad=12 if not both else 46)
    ax.grid(alpha=0.18, lw=0.5)
    h, l = ax.get_legend_handles_labels()
    if both:
        h += [plt.Line2D([], [], color="0.25", lw=2, marker="s", ms=7),
              plt.Line2D([], [], color="0.25", lw=2, ls="--", marker="s", ms=7, mfc="white")]
        l += ["cross-entropy", "square loss"]
    if legend:
        ax.legend(h, l, loc="upper left", fontsize=12.5, framealpha=0.92)


def load(tmpl, otag, m, key, seed=0):
    p = tmpl.format(o=otag, m=m, s=seed) + "/eval_per_r.npz"
    if not os.path.exists(p):
        return None
    return np.load(p)[key]


# The seed range is shaded, with the median drawn over it. Read the square-loss
# band with care at each arm's transition: the spread there is 52x to 88x against
# about 1x either side, because a run near the threshold either generalises or
# does not, so the fill spans two branches rather than describing scatter about
# the median. Under cross-entropy the spread never exceeds 2x and the band is an
# ordinary range.


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
    ap.add_argument("--regime_panel", choices=["ce", "both", "grid"], default=None,
                    help="add m* by dynamical regime: 'ce' = panel (c), cross-entropy "
                         "only; 'both' = panel (c), cross-entropy solid and square loss "
                         "dashed; 'grid' = 2x2, a second row with each loss's m* under "
                         "its own panel")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(ARMS)))
    if a.regime_panel == "grid":
        fig, grid = plt.subplots(2, 2, figsize=(14.0, 12.4),
                                 gridspec_kw={"height_ratios": [1, 0.8]})
        axes = list(grid[0]) + list(grid[1])
    elif a.regime_panel:
        fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.7),
                                 gridspec_kw={"width_ratios": [1, 1, 0.85]})
    else:
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
            ax.fill_between(MS, yn.min(0), yn.max(0), color=c, alpha=0.22,
                            lw=0, zorder=1)
            ax.fill_between(MS, ys.min(0), ys.max(0), color=c, alpha=0.12,
                            lw=0, zorder=1)
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

    ce = ("bins", "ce", lambda v: np.nanmean(v))
    sq = ("mse", "rms", lambda v: np.nanmean(v ** 2))
    if a.regime_panel == "ce":
        regime_panel(axes[2], a.seeds, [ce], "Crossover by regime (cross-entropy)")
    elif a.regime_panel == "both":
        regime_panel(axes[2], a.seeds, [ce, sq], "Crossover by regime")
    elif a.regime_panel == "grid":
        regime_panel(axes[2], a.seeds, [ce], "Crossover by regime, cross-entropy")
        regime_panel(axes[3], a.seeds, [sq], "Crossover by regime, square loss",
                     legend=False)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes):
        ax.text(-0.14, 1.17, label, transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
