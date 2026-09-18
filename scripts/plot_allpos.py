"""The square-loss binning figure, trained on every position instead of the last.

train_subset.py scored only the final position of each window, so a model with a
50-step context had only ever predicted with 50 steps visible. --all_positions
scores every position; the target becomes the window shifted by one, whose last
entry is the old target, and evaluation still reads the final position. So the
two columns differ in the training loss and in nothing else: same data, same
schedule, same 160k steps, same metric.

Left is the original sweep (one seed), right the new one (five seeds, drawn
individually with the median as a line). Rows share a y scale, so the columns are
directly comparable.

Two things change and they point opposite ways, which is why this is worth
plotting rather than summarising. The transition stops depending on resolution:
m* runs 187, 225, 349, 616 for 32 to 256 bins on the left and 195, 168, 208, 220
on the right, flat within the seed spread. But the achievable error gets worse at
exactly the fine end that previously looked best -- the plateau rises 4.4x at 64
bins, 5.7x at 128 and 31x at 256, while 8 and 16 bins barely move.

That is not a contradiction. Averaging over positions weights early ones that see
one or two steps and cannot identify the map at all, so the objective is no longer
the quantity being plotted. Every-position training buys signal per step, and the
fine arms do now converge at 160k where they did not before, but it buys it by
optimising something other than the last-position error.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
BINS = [8, 16, 32, 64, 128, 256]
SEEDS = range(5)


def mse(tmpl, key):
    return np.array([float(np.nanmean(
        np.load(tmpl.format(m=m) + "/eval_per_r.npz")[key] ** 2)) for m in MS])


plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
cmap = plt.get_cmap("viridis")
cols = {b: cmap(v) for b, v in zip(BINS, np.linspace(0.12, 0.88, len(BINS)))}
fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)

for i, (key, row) in enumerate(((  "rms_at_train_r", "Seen tasks\n(evaluated at training $r$)"),
                                ("rms_per_r", "New tasks\n(full-range grid)"))):
    lo, hi = np.inf, 0
    for j, (title, kind) in enumerate(((  "Trained on the last position", "last"),
                                       ("Trained on every position", "all"))):
        ax = axes[i, j]
        for b in BINS:
            if kind == "last":
                y = mse(f"runs_divbins/in{b}_mse_m{{m}}_seed0", key)
                ax.plot(MS, y, "o-", color=cols[b], ms=5.5, lw=1.7, label=str(b))
            else:
                ys = np.array([mse(f"runs_divbins_allpos/in{b}_mse_m{{m}}_seed{s}", key)
                               for s in SEEDS])
                for s in SEEDS:
                    ax.plot(MS, ys[s], "o", color=cols[b], ms=3.5, alpha=0.5, lw=0)
                y = np.median(ys, 0)
                ax.plot(MS, y, "-", color=cols[b], lw=1.9, label=str(b))
            lo, hi = min(lo, y.min()), max(hi, y.max())
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(title, fontsize=13)
        if i == 1:
            ax.set_xlabel("number of training tasks (distinct $r$ values)")
        if j == 0:
            ax.set_ylabel(row + "\nMean squared error", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for j in (0, 1):
        axes[i, j].set_ylim(lo * 0.55, hi * 1.8)

axes[1, 0].legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
                  title="input bins", title_fontsize=9.5)
fig.text(0.5, -0.035,
         "Square loss against the exact next state, 160k steps, 32000 trajectories, context 50 — identical either side; only the training\n"
         "loss differs. Evaluation reads the final position in both. Right-hand points are the five seeds, line is their median.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/allpos.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/allpos.png")
