"""The task-diversity overlay with every arm read at a converged checkpoint.

The 160k-step sweep had not converged at the fine end: taking validation loss at
160k as a fraction of its value at 40k gives 0.99, 0.98 and 0.95 for 8, 16 and 32
bins, but 0.82, 0.75 and 0.69 for 64, 128 and 256. Those three were rerun at 320k,
where the last eighth of training moves the loss by under half a percent.

So each arm is taken at the budget where it has converged -- 160k for 8, 16 and 32
bins, 320k for 64, 128 and 256. The budgets differ, which would matter if the
coarse arms were still improving; they are not, having levelled off by 40k, so
giving them the extra steps would change nothing. Reading every arm at 320k
instead would have cost 21 more jobs to reach the same numbers.

This matters for the earlier reading. Doubling the budget lowers the fine arms
much more at high task count than at low, which widens their drop and pushes the
transition later still. The direction of the result survives; the magnitudes at
the fine end were understated.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
# (tag, label, run directory template, steps) -- converged source for each arm
ARMS = [("in8", 8, "runs_divbins/in8_{o}_m{m}_seed0", 160),
        ("in16", 16, "runs_divbins/in16_{o}_m{m}_seed0", 160),
        ("in32", 32, "runs_divbins/in32_{o}_m{m}_seed0", 160),
        ("in64", 64, "runs_divbins_320k/in64_{o}_m{m}_seed0", 320),
        ("in128", 128, "runs_divbins_320k/in128_{o}_m{m}_seed0", 320),
        ("in256", 256, "runs_divbins_320k/in256_{o}_m{m}_seed0", 320)]
LM = np.log10(MS)


def curve(tmpl, otag, key, power=1.0):
    """Aggregate a per-r array across r.

    The stored scalar-arm arrays are per-r RMS values, so mean squared error is
    mean(rms**2). Raising the AVERAGED rms to a power instead gives
    (mean rms)**2, which is smaller by Var(rms) across r -- and since that
    variance shrinks as the model becomes uniform across the family, the error
    is not a constant factor but a distortion of the curve's shape.
    """
    return np.array([float(np.nanmean(np.load(
        tmpl.format(o=otag, m=m) + "/eval_per_r.npz")[key] ** power))
        for m in MS])


def mstar(c, min_drop=2.0):
    plateau = c[-2:].mean(); drop = c[0] / plateau
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
cmap = plt.get_cmap("viridis")
cols = [cmap(v) for v in np.linspace(0.12, 0.88, len(ARMS))]
fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True)

for j, (otag, ctitle, ylab, power) in enumerate(
        (("bins", "Cross-entropy loss", "Mean cross-entropy (nats)", 1.0),
         ("mse", "Square loss", "Mean squared error", 2.0))):
    print(f"--- {ctitle} ---")
    for i, (row_key, row_lab) in enumerate(
            (("at_train_r", "Seen tasks\n(evaluated at training $r$)"),
             ("per_r", "New tasks\n(full-range grid)"))):
        ax = axes[i, j]
        for (tag, nb, tmpl, steps), c in zip(ARMS, cols):
            y = curve(tmpl, otag, ("ce_" if otag == "bins" else "rms_") + row_key,
                      power)
            ax.plot(MS, y, "o-", color=c, ms=5, lw=1.6, label=str(nb))
            if i == 1:
                m, d = mstar(y)
                print(f"  {nb:>4} bins ({steps}k)  m* = "
                      f"{'none' if m is None else f'{m:6.0f}'}   drop {d:5.1f}x")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(ctitle, fontsize=13)
        if i == 1:
            ax.set_xlabel("number of training tasks (distinct $r$ values)")
        ax.set_ylabel((row_lab + "\n" + ylab) if j == 0 else ylab, fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    lo = min(axes[i, j].get_ylim()[0] for i in (0, 1))
    hi = max(axes[i, j].get_ylim()[1] for i in (0, 1))
    for i in (0, 1):
        axes[i, j].set_ylim(lo, hi)

axes[1, 0].legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
                  title="input bins", title_fontsize=9.5)
fig.text(0.5, -0.035,
         "Every arm at a converged checkpoint: 160k steps for 8-32 bins, which had levelled off by 40k, "
         "320k for 64-256 bins, which had not.\nOutput pinned at 64 bins and total trajectories at 32000 "
         "throughout, so only input resolution changes.  Continuous input excluded.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/taskdiv_overlay_converged.{e}", dpi=170,
                bbox_inches="tight")
print("wrote figures_taskdiv/taskdiv_overlay_converged.png")
