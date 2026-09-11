"""Error against context length, for the square-loss arm at 64 input bins.

Two ways of varying context, on one axis:

  trained at that length (solid)   -- runs_ctx, one model per context, plus the
      existing 320k in64_mse arm as the L=50 point. traj_len = context_len + 100
      throughout, so every arm has 99 windows per trajectory and the same 3.168M
      window pool; otherwise a shorter context would quietly draw a larger pool
      from the same trajectories.

  trained at 50, restricted (dashed) -- the same in64_mse checkpoints evaluated
      with attention masked to the last L positions. Truncating the input would
      move the prediction to a position these models never predict at, so the
      mask keeps the query at its trained position and removes only older keys.

They meet by construction at L=50, where the restriction is vacuous. Everywhere
else the gap is the difference between the context a model USES and the context
it was TRAINED with -- the dashed curve can only ever show what a 50-step model
falls back on when deprived, while the solid curve shows what is achievable when
the shorter length is the training condition.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = [100, 250, 500, 1000, 2000, 4000, 8000]
CTX = [10, 25, 35, 50, 75]


def trained(L, m, key):
    p = (f"runs_divbins_320k/in64_mse_m{m}_seed0" if L == 50
         else f"runs_ctx/L{L}_m{m}_seed0")
    f = p + "/eval_per_r.npz"
    if not glob.glob(f):
        return np.nan
    # mean of the SQUARED per-r error: the mean squared error over the grid.
    # Averaging per-r RMS then squaring is smaller by Jensen and arm-dependent.
    return float(np.nanmean(np.load(f)[key] ** 2))


z = np.load("figures_taskdiv/eval_context.npz")
RL = z["contexts"]

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
cmap = plt.get_cmap("viridis")
cols = [cmap(v) for v in np.linspace(0.12, 0.88, len(MS))]
fig, axes = plt.subplots(2, 1, figsize=(7.6, 8.4), sharex=True)

missing = []
for i, (key, rkey, lab) in enumerate(
        (("rms_at_train_r", "rms_seen", "Seen tasks\n(evaluated at training $r$)"),
         ("rms_per_r", "rms_new", "New tasks\n(full-range grid)"))):
    ax = axes[i]
    for m, c in zip(MS, cols):
        y = [trained(L, m, key) for L in CTX]
        if i == 1:
            missing.extend(L for L, v in zip(CTX, y) if not np.isfinite(v))
        ax.plot(CTX, y, "o-", color=c, ms=6, lw=1.7, label=str(m))
        yr = [np.nanmean(z[f"in64_mse_m{m}_seed0_L{L}_{rkey}"] ** 2) for L in RL]
        ax.plot(RL, yr, "--", color=c, lw=1.2, alpha=0.75)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 25, 50, 75])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "75"])
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylabel(lab + "\nMean squared error", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[1].set_xlabel("context length")
axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
               title="training tasks $m$", title_fontsize=9)
axes[0].plot([], [], "k-", lw=1.7, label="_")
h = [plt.Line2D([], [], color="k", lw=1.7, marker="o", ms=6),
     plt.Line2D([], [], color="k", lw=1.2, ls="--")]
axes[1].legend(h, ["trained at this length", "trained at 50, attention restricted"],
               frameon=False, fontsize=9.5, loc="lower left")

note = ("Square loss against the exact next state, 64 input bins, 320k steps, 32000 trajectories.\n"
        "The two meet at $L=50$ by construction, where the restriction is vacuous.")
if missing:
    note += f"\n$L=75$ incomplete: {7 - sum(1 for L in missing if L == 75)}/7 task counts finished."
fig.text(0.5, -0.035, note, ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/context.{e}", dpi=170, bbox_inches="tight")

# ---- trained-at-length only, in the form of the task-diversity panel --------
# Same two rows and same square-loss axis as taskdiv_overlay_converged, with
# context length on x in place of task count. Linear in L: the ladder is
# 10-75 and a log axis would compress the top, where the curves separate.
fig2, ax2 = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True)
for i, (key, lab) in enumerate(
        (("rms_at_train_r", "Seen tasks\n(evaluated at training $r$)"),
         ("rms_per_r", "New tasks\n(full-range grid)"))):
    a = ax2[i]
    for m, c in zip(MS, cols):
        a.plot(CTX, [trained(L, m, key) for L in CTX], "o-", color=c, ms=6,
               lw=1.7, label=str(m))
    a.set_yscale("log"); a.grid(alpha=0.25, lw=0.5)
    a.set_xlim(5, 80); a.set_xticks(CTX)
    a.set_ylabel(lab + "\nMean squared error", fontsize=10)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax2[1].set_xlabel("context length")
ax2[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
              title="training tasks $m$", title_fontsize=9)
fig2.text(0.5, -0.035,
          "Square loss against the exact next state, 64 input bins, 320k steps, 32000 trajectories, one model per context.\n"
          "traj_len = context_len + 100 throughout, so every arm keeps 99 windows per trajectory and the same pool.",
          ha="center", fontsize=9.5, color="0.25")
fig2.tight_layout()
for e in ("png", "pdf"):
    fig2.savefig(f"figures_taskdiv/context_trained.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/context_trained.png")

# ---- the task-diversity panel proper, coloured BY context length -----------
# Exactly the square-loss column of taskdiv_overlay_converged -- task count on x,
# two rows -- with context length taking the place of input bins as the series.
LM = np.log10(MS)


def mstar(c, min_drop=2.0):
    c = np.asarray(c); pl = c[-2:].mean(); drop = c[0] / pl
    if drop < min_drop:
        return None, drop
    h = np.sqrt(c[0] * pl)
    for i in range(len(c) - 1):
        if c[i] >= h >= c[i + 1]:
            f = (np.log(c[i]) - np.log(h)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LM[i] + f * (LM[i + 1] - LM[i])), drop
    return None, drop


ccols = [plt.get_cmap("viridis")(v) for v in np.linspace(0.12, 0.88, len(CTX))]
fig3, ax3 = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True)
for i, (key, lab) in enumerate(
        (("rms_at_train_r", "Seen tasks\n(evaluated at training $r$)"),
         ("rms_per_r", "New tasks\n(full-range grid)"))):
    a = ax3[i]
    for L, c in zip(CTX, ccols):
        y = [trained(L, m, key) for m in MS]
        a.plot(MS, y, "o-", color=c, ms=6, lw=1.7, label=str(L))
        if i == 1:
            mm, dd = mstar(y)
            print("  L=%-3d m* = %-8s drop %6.1fx" %
                  (L, "none" if mm is None else "%.0f" % mm, dd))
    a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.25, lw=0.5)
    a.set_ylabel(lab + "\nMean squared error", fontsize=10)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax3[1].set_xlabel("number of training tasks (distinct $r$ values)")
ax3[0].legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
              title="context length", title_fontsize=9.5)
fig3.text(0.5, -0.035,
          "Square loss against the exact next state, 64 input bins, 320k steps, 32000 trajectories, one model per context length.\n"
          "traj_len = context_len + 100 throughout, so every arm keeps 99 windows per trajectory and the same pool.",
          ha="center", fontsize=9.5, color="0.25")
fig3.tight_layout()
for e in ("png", "pdf"):
    fig3.savefig(f"figures_taskdiv/taskdiv_by_context.{e}", dpi=170,
                 bbox_inches="tight")
print("wrote figures_taskdiv/taskdiv_by_context.png")
print("\nseen-task MSE, trained at that length:")
print("%7s %s" % ("m", " ".join("L=%-9d" % L for L in CTX)))
for m in MS:
    print("%7d %s" % (m, " ".join("%-11.2e" % trained(L, m, "rms_at_train_r")
                                  for L in CTX)))
print("wrote figures_taskdiv/context.png")
print("\nnew-task MSE, trained at that length:")
print("%7s %s" % ("m", " ".join("L=%-9d" % L for L in CTX)))
for m in MS:
    print("%7d %s" % (m, " ".join("%-11.2e" % trained(L, m, "rms_per_r") for L in CTX)))
