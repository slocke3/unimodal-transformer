"""The binning sweep normalised by each arm's own floor, with seeds shown.

Three things made the raw version hard to read, and only one was cosmetic.

Floors span 140x across the ladder, so curves end at different heights and cross
each other even when the transition order is clean. Dividing each arm by its own
large-m error removes that: every curve then runs from far above 1 down to 1, and
the only thing distinguishing them is where it happens.

Seeds are drawn individually rather than as a band. At the threshold the outcome
is bimodal -- a run generalises or it does not -- so a band smears two distinct
populations into one grey region, and a median over three runs lands wherever the
majority happened to fall. As points, 64 bins at m=500 reads as one run at the
floor and two a hundred times above it, which is what actually happened.

The m grid is too coarse to locate the threshold, and this figure does not hide
it. Counting points that fall strictly inside the transition, between 90% and 10%
of the drop: none at 32 bins, two at 64, two at 128, four at 256. At 32 bins the
whole transition happens between m=100 and m=250, so m* there is interpolated
across a single factor-of-two gap.

256 bins is marked open because it never plateaus: its last two task counts still
fall threefold, so its normalisation is against a floor it has not reached and its
m* is an extrapolation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
BINS = [32, 64, 128, 256]
SEEDS = [0, 1, 2]


def mse(nb, s):
    return np.array([float(np.nanmean(np.load(
        f"runs_divbins_640k/in{nb}_mse_m{m}_seed{s}/eval_per_r.npz")["rms_per_r"] ** 2))
        for m in MS])


Y = {nb: np.array([mse(nb, s) for s in SEEDS]) for nb in BINS}
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
cmap = plt.get_cmap("viridis")
cols = {nb: cmap(v) for nb, v in zip(BINS, np.linspace(0.12, 0.85, len(BINS)))}
fig, ax = plt.subplots(figsize=(7.8, 5.8))

for nb in BINS:
    y = Y[nb]
    floor = y[:, -1].mean()                 # last point, not a fitted plateau
    n = y / floor
    settled = nb != 256                     # 256 is still falling at m=8000
    ax.plot(MS, np.median(n, 0), "-" if settled else "--", color=cols[nb],
            lw=1.9, label=f"{nb}" + ("" if settled else "  (not plateaued)"))
    for s in SEEDS:
        ax.plot(MS, n[s], "o" if settled else "o", color=cols[nb], ms=5.5,
                mfc=cols[nb] if settled else "white", alpha=0.95, lw=0)

ax.axhline(1, color="0.55", lw=1, ls=":")
ax.text(105, 1.25, "each arm's own large-$m$ error", fontsize=9.5, color="0.42")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5)
ax.set_xlabel("number of training tasks (distinct $r$ values)")
ax.set_ylabel("new-task MSE, in units of that arm's floor")
ax.legend(frameon=False, fontsize=9.5, loc="upper right", title="input bins",
          title_fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, -0.055,
         "640k steps, square loss; every seed plotted, line is the median. Normalising by each arm's floor removes the 140x\n"
         "spread in where the curves end, so only the location of the drop distinguishes them. Points inside the transition:\n"
         "0 of 7 at 32 bins, 2 at 64, 2 at 128, 4 at 256 — the grid is too coarse to place the threshold.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/taskdiv_640k_norm.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/taskdiv_640k_norm.png")
