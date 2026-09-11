"""How much of its context a trained model actually uses.

The 320k square-loss models at 64 input bins, trained with a 50-step context,
re-evaluated with attention masked to the last L positions. Truncating the input
instead would move the prediction to position L-1, which these models never
predict at -- they are trained on the final position only, with learned
positional embeddings -- so the mask keeps the query where it was trained and
removes only the older keys. Verified to block at every layer: scrambling the
hidden positions leaves the output bit-identical, and L=50 reproduces the
unrestricted forward exactly.

One panel, new tasks -- the quantity the diversity transition is defined on. The
seen-task curves have the same shape, the same plateau and the same late collapse,
but they are not interchangeable: they differ from these by about 10% typically
and by as much as a factor of two in places, so this panel should be read as the
new-task result rather than as standing in for both.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = [100, 250, 500, 1000, 2000, 4000, 8000]
z = np.load("figures_taskdiv/eval_context.npz")
L = z["contexts"]

# how different is the seen-task row, really?
d = [abs(np.nanmean(z[f"in64_mse_m{m}_seed0_L{l}_rms_seen"]) ** 2
         / np.nanmean(z[f"in64_mse_m{m}_seed0_L{l}_rms_new"]) ** 2 - 1)
     for m in MS for l in L]
print(f"seen vs new: median |ratio-1| = {100*np.median(d):.1f}%, "
      f"max = {100*max(d):.1f}%")

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(7.4, 5.4))

for m, c in zip(MS, [cmap(v) for v in np.linspace(0.12, 0.88, len(MS))]):
    y = [np.nanmean(z[f"in64_mse_m{m}_seed0_L{l}_rms_new"]) ** 2 for l in L]
    ax.plot(L, y, "o-", color=c, ms=5.5, lw=1.7, label=str(m))
    print(f"  m={m:5d}  " + "  ".join(f"{v:.2e}" for v in y))

full = np.mean([np.nanmean(z[f"in64_mse_m{m}_seed0_L50_rms_new"]) ** 2 for m in MS])
ax.axhline(full, color="0.55", lw=1, ls=":")
ax.text(0.7, full * 1.35, "full 50-step context", fontsize=9.5, color="0.4")

# Linear in L, not log. The structure worth seeing is the collapse between
# L=30 and L=50; on a log axis that is the last 13% of the width while the
# featureless plateau from 2 to 20 takes 59% of it.
ax.set_yscale("log")
ax.set_xlim(0, 52)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel("context positions the model is allowed to attend to")
ax.set_ylabel("mean squared error, new tasks")
ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
          title="training tasks $m$", title_fontsize=9.5)
ax.axvspan(30, 50, color="0.92", zorder=0)
ax.grid(alpha=0.25, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.5, -0.05,
         "Models trained at context 50 (square loss, 64 input bins, 320k steps), evaluated with attention\n"
         "masked to the last $L$ positions. Error barely moves from $L=2$ to $L=20$, then falls two orders\n"
         "of magnitude over the shaded final twenty — the model needs nearly its whole trained window to engage.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/context_eval.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/context_eval.png")
