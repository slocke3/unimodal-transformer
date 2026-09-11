"""Where the task-diversity transition sits, as a function of input resolution.

The controlled diversity sweep holds total trajectories at 32000 and varies only
the number of distinct training tasks m, so the new-task error falls as m grows
and then plateaus. m* is where it has fallen halfway, in log error, from its
m=100 value to that plateau. Output resolution is pinned at 64 bins for every
arm, so anything that moves is the input representation, not the target.

Error is implied-map RMS rather than cross-entropy: CE is bounded by
log(n_bins_out) and undefined for the square-loss arms, while RMS is in map units
and so is the one quantity comparable across both losses and every input.

Two arms have no transition to locate, for opposite reasons, and the figure says
so rather than extrapolating a number:

  8 and 16 bins  the input-quantisation floor dominates. Knowing x only to +-1/16
                 leaves f(x) uncertain by roughly |f'|/8, and against that floor
                 memorising and solving cost nearly the same, so there is nothing
                 for a transition to separate -- the curve is flat to 1.1x.
  continuous     already converged at m=100, the smallest m swept. Its transition
                 is below the grid, not absent.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = [100, 250, 500, 1000, 2000, 4000, 8000]
INPUTS = [("in8", "8"), ("in16", "16"), ("in32", "32"), ("in64", "64"),
          ("in128", "128"), ("in256", "256"), ("cont", "cont")]
LM = np.log10(MS)


def curve(itag, otag):
    """Grid RMS as sqrt(mean of the squared per-r error). The plain mean of per-r
    RMS is a different quantity, smaller by Jensen and by an arm-dependent
    amount, so it distorts where the half-drop point lands."""
    out = []
    for m in MS:
        p = (f"runs_div_controlled/div_m{m}_seed0/eval_per_r.npz"
             if itag == "in64" and otag == "bins" else
             f"runs_divbins/{itag}_{otag}_m{m}_seed0/eval_per_r.npz")
        out.append(float(np.sqrt(np.nanmean(np.load(p)["rms_per_r"] ** 2))))
    return np.array(out)


def mstar(c, min_drop=2.0):
    plateau = c[-2:].mean()
    drop = c[0] / plateau
    if drop < min_drop:
        return None, drop
    half = np.sqrt(c[0] * plateau)
    for i in range(len(c) - 1):
        if c[i] >= half >= c[i + 1]:
            f = (np.log(c[i]) - np.log(half)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LM[i] + f * (LM[i + 1] - LM[i])), drop
    return None, drop


plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(figsize=(7.4, 5.2))
X = {t: i for i, (t, _) in enumerate(INPUTS)}
X["cont"] = 6.5                                   # set apart: not a binning

for otag, mfc, label in (("bins", "k", "64-bin cross-entropy"),
                         ("mse", "white", "square loss")):
    xs, ys = [], []
    for itag, _ in INPUTS:
        m, drop = mstar(curve(itag, otag))
        print(f"{itag:6s} {otag:5s} m*={'none' if m is None else f'{m:.0f}':>6s}"
              f"  drop {drop:.1f}x")
        if m is not None:
            xs.append(X[itag]); ys.append(m)
    ax.plot(xs, ys, "o-", color="k", mfc=mfc, ms=8, lw=1.5, label=label)

ax.axvspan(-0.45, 1.45, color="0.88", zorder=0)
ax.text(0.5, 128, "no transition\ninput floor\ndominates", ha="center",
        va="center", fontsize=9.5, color="0.35")
ax.axvline(5.75, color="0.75", lw=1, ls=":")
ax.annotate("", xy=(6.5, 88), xytext=(6.5, 150),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.6))
ax.text(6.5, 158, "below the\nswept range", ha="center", va="bottom",
        fontsize=9.5, color="0.35")
ax.axhline(100, color="0.75", lw=1, ls="--")
ax.text(6.35, 103, "smallest $m$ swept", ha="right", va="bottom",
        fontsize=9, color="0.5")

ax.set_yscale("log"); ax.set_ylim(80, 900); ax.set_xlim(-0.45, 7.0)
ax.set_xticks(list(X.values()))
ax.set_xticklabels([lab for _, lab in INPUTS])
ax.set_yticks([100, 200, 400, 800])
ax.set_yticklabels(["100", "200", "400", "800"])
ax.set_xlabel("input resolution  (bins;  'cont' = raw $x$)")
ax.set_ylabel("transition task count  $m^*$")
ax.legend(frameon=False, fontsize=11, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.5, -0.05,
         "Output pinned at 64 bins throughout, total trajectories at 32000, so only the input changes.\n"
         "$m^*$ is the half-drop point of new-task implied-map RMS. Continuous input is not the fine-binning limit:\n"
         "a bin embedding is a lookup table with no notion that neighbouring bins are near, so finer bins need MORE tasks.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/transition_vs_bins.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/transition_vs_bins.png")
