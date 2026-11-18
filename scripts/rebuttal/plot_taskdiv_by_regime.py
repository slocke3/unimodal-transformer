"""Task-diversity curves and crossovers split by dynamical regime.

Does the whole family cross over at one task count, or does each regime have
its own transition? Uses the transformer runs behind taskdiv_gap (seeds 0-3),
restricting the new-task average to the evaluation-grid points in each regime
(and the seen-task average to the training r in it):

  fixed point   0.5 <= r < 3        (includes the decay to 0 for r < 1)
  periodic      3 <= r < 3.5699     (period-doubling cascade)
  chaotic band  r >= 3.5699         (includes the periodic windows inside it)

Outputs
  taskdiv_by_regime        rows CE / MSE, one column per regime; one curve
                           per input resolution (viridis, as taskdiv_gap);
                           solid new tasks, dashed seen tasks; medians over
                           seeds.
  taskdiv_mstar_by_regime  m* against input resolution, one line per regime
                           plus the family average (black); m* of the median
                           curve, bars = range of per-seed m*.
  ..._summary.txt          the m* table.
  --bands                  also shade the full seed range of each new-task
                           curve (taskdiv_by_regime_bands).

m* is the paper's definition: log-scale midpoint between E(100) and the
plateau mean(E(4000), E(8000)), interpolated in (log m, log E); undefined when
E(100)/plateau < 2. Torch-free:
  python scripts/rebuttal/plot_taskdiv_by_regime.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15.5, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

MS = [100, 250, 500, 1000, 2000, 4000, 8000]
LM = np.log10(MS)
SEEDS = [0, 1, 2, 3]
NS = [8, 16, 32, 64, 128, 256]
TMPL = {8: "runs_divbins/in8_{o}_m{m}_seed{s}", 16: "runs_divbins/in16_{o}_m{m}_seed{s}",
        32: "runs_divbins/in32_{o}_m{m}_seed{s}", 64: "runs_divbins_320k/in64_{o}_m{m}_seed{s}",
        128: "runs_divbins_320k/in128_{o}_m{m}_seed{s}",
        256: "runs_divbins_320k/in256_{o}_m{m}_seed{s}"}
REGIMES = [("fixed point", r"$0.5\leq r<3$", 0.5, 3.0, "#0072B2"),
           ("periodic", r"$3\leq r<3.57$", 3.0, 3.5699, "#E69F00"),
           ("chaotic band", r"$r\geq3.57$", 3.5699, 4.01, "#CC79A7")]
OBJS = [("ce", "bins", "ce_per_r", "ce_at_train_r", False, "new-task cross-entropy (nats)"),
        ("mse", "mse", "rms_per_r", "rms_at_train_r", True, "new-task mean squared error")]
OUT = "figures_rebuttal"


def mstar(c, min_drop=2.0):
    c = np.asarray(c, float)
    plateau = c[-2:].mean()
    if not np.isfinite(plateau) or c[0] / plateau < min_drop:
        return None
    h = np.sqrt(c[0] * plateau)
    for i in range(len(c) - 1):
        if c[i] >= h >= c[i + 1]:
            f = (np.log(c[i]) - np.log(h)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LM[i] + f * (LM[i + 1] - LM[i]))
    return None


def load(n, o, newkey, seenkey, sq):
    """{regime: (seeds x m new, seeds x m seen)}, plus the all-r curve."""
    out = {name: ([], []) for name, *_ in REGIMES}
    allc = []
    for s in SEEDS:
        rows = {name: ([], []) for name, *_ in REGIMES}
        ra = []
        for m in MS:
            p = TMPL[n].format(o=o, m=m, s=s) + "/eval_per_r.npz"
            if not os.path.exists(p):
                break
            d = np.load(p)
            r, sr = d["r_grid"], d["seen_r"]
            v = d[newkey] ** 2 if sq else d[newkey]
            w = d[seenkey] ** 2 if sq else d[seenkey]
            ra.append(np.nanmean(v))
            for name, _, lo, hi, _ in REGIMES:
                rows[name][0].append(np.nanmean(v[(r >= lo) & (r < hi)]))
                k = (sr >= lo) & (sr < hi)
                rows[name][1].append(np.nanmean(w[k]) if k.any() else np.nan)
        if len(ra) == len(MS):
            allc.append(ra)
            for name in rows:
                out[name][0].append(rows[name][0])
                out[name][1].append(rows[name][1])
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}, np.array(allc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", action="store_true",
                    help="shade the full seed range of the new-task curves, as in taskdiv_gap; "
                         "writes taskdiv_by_regime_bands.* instead of taskdiv_by_regime.*")
    a = ap.parse_args()
    data = {(obj, n): load(n, o, nk, sk, sq)
            for obj, o, nk, sk, sq, _ in OBJS for n in NS}
    lines = ["m* by regime: median curve over seeds 0-3 [per-seed range]; '-' = undefined"]

    # ---- curves: rows = objective, columns = regime --------------------------
    colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(NS)))
    fig, axes = plt.subplots(2, len(REGIMES), figsize=(5.6 * len(REGIMES) + 1, 10.5))
    for i, (obj, *_, ylab) in enumerate(OBJS):
        for j, (name, rng, *_rest) in enumerate(REGIMES):
            ax = axes[i, j]
            for n, c in zip(NS, colors):
                new, seen = data[(obj, n)][0][name]
                if a.bands:
                    ax.fill_between(MS, new.min(0), new.max(0), color=c, alpha=0.18, lw=0)
                ax.plot(MS, np.median(new, 0), "-o", color=c, lw=2.2, ms=5, label=str(n))
                ax.plot(MS, np.nanmedian(seen, 0), "--", color=c, lw=1.4, alpha=0.65)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(alpha=0.18, lw=0.5, which="both")
            if i == 0:
                ax.set_title(f"{name}, {rng}")
            else:
                ax.set_xlabel(r"training tasks $m$")
            if j == 0:
                ax.set_ylabel(ylab)
    res = [plt.Line2D([], [], color=c, lw=2.2, marker="o", ms=5) for c in colors]
    sty = [plt.Line2D([], [], color="0.25", lw=2.2, marker="o", ms=5),
           plt.Line2D([], [], color="0.25", lw=1.4, ls="--")]
    fig.legend(res + sty, [f"{n} bins" for n in NS] + ["new tasks", "seen tasks"],
               loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.tight_layout()
    name = "taskdiv_by_regime_bands" if a.bands else "taskdiv_by_regime"
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{e}", dpi=200, bbox_inches="tight")

    # ---- m* against resolution, one line per regime --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (obj, *_) in zip(axes, OBJS):
        lines.append(f"== {obj}")
        lines.append(f"{'bins':>5s}  {'all':>14s}  " + "  ".join(f"{nm:>16s}" for nm, *_ in REGIMES))
        rows = {n: [] for n in NS}
        for name, rng, _, _, col in REGIMES + [("all r", "", 0, 0, "k")]:
            xs, ys, lo, hi = [], [], [], []
            for n in NS:
                regs, allc = data[(obj, n)]
                curves = allc if name == "all r" else regs[name][0]
                med = mstar(np.median(curves, 0))
                per = [v for v in (mstar(c) for c in curves) if v]
                rows[n].append((med, per))
                if med:
                    xs.append(n); ys.append(med)
                    lo.append(med - min(per) if per else 0)
                    hi.append(max(per) - med if per else 0)
            kw = dict(color=col, lw=2.4 if name == "all r" else 2.0, ms=7, capsize=3,
                      marker="o" if name == "all r" else "s", zorder=5 if name == "all r" else 3)
            ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)],
                        label=name if name == "all r" else f"{name} ({rng})", **kw)
        for n in NS:
            cells = rows[n][-1:] + rows[n][:-1]          # all first, then regimes
            fmt = lambda t: ("-" if not t[0] else f"{t[0]:.0f}") + (
                f" [{min(t[1]):.0f}-{max(t[1]):.0f}]" if t[1] else "")
            lines.append(f"{n:>5d}  {fmt(cells[0]):>14s}  " + "  ".join(f"{fmt(c):>16s}" for c in cells[1:]))
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("input bins $N$")
        ax.set_ylabel(r"crossover task count $m^*$")
        ax.set_title("Cross-entropy" if obj == "ce" else "Square loss")
        ax.grid(alpha=0.18, lw=0.5, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/taskdiv_mstar_by_regime.{e}", dpi=200, bbox_inches="tight")

    with open(f"{OUT}/taskdiv_mstar_by_regime_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}/{name}.*, {OUT}/taskdiv_mstar_by_regime.*")


if __name__ == "__main__":
    main()
