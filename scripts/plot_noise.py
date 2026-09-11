"""Continuous input corrupted by noise matched to a bin width.

--noise_bins n adds uniform noise of half-width 1/(2n) to the raw input, which is
exactly the error distribution of quantising to n bins. So each noise arm has a
binning arm it can be read against, with one difference that is the whole point:
quantisation is deterministic, the same x always landing in the same bin, while
noise is independent at every position and can in principle be averaged away
across the context.

Errors are the mean of the SQUARED per-r error. Averaging the per-r RMS and
squaring afterwards, which earlier versions of these figures did, is smaller by
Jensen and by an arm-dependent factor between 1.5x and 9x, so it distorted shapes
and not just levels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
NB = [0, 8, 16, 32, 64, 128, 256]
BIN_SRC = {8: "runs_divbins/in8_mse_m{m}_seed0", 16: "runs_divbins/in16_mse_m{m}_seed0",
           32: "runs_divbins/in32_mse_m{m}_seed0",
           64: "runs_divbins_320k/in64_mse_m{m}_seed0",
           128: "runs_divbins_320k/in128_mse_m{m}_seed0",
           256: "runs_divbins_320k/in256_mse_m{m}_seed0"}


def mse(tmpl, key="rms_per_r"):
    return np.array([float(np.nanmean(
        np.load(tmpl.format(m=m) + "/eval_per_r.npz")[key] ** 2)) for m in MS])


plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
cmap = plt.get_cmap("plasma")
cols = {n: cmap(v) for n, v in zip(NB[1:], np.linspace(0.05, 0.78, len(NB) - 1))}
cols[0] = "0.15"

# ---- figure 1: the sweep, in the form of the square-loss column -------------
fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.2), sharex=True)
for i, (key, lab) in enumerate((("rms_at_train_r", "Seen tasks\n(evaluated at training $r$)"),
                                ("rms_per_r", "New tasks\n(full-range grid)"))):
    ax = axes[i]
    for n in NB:
        y = mse(f"runs_noise/nb{n}_m{{m}}_seed0", key)
        ax.plot(MS, y, "o-", color=cols[n], ms=5.5, lw=1.7,
                label="none" if n == 0 else f"1/{2*n}  ($n$={n})")
        if i == 1:
            print(f"  noise n={n:<4} " + " ".join(f"{v:.2e}" for v in y))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylabel(lab + "\nMean squared error", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[1].set_xlabel("number of training tasks (distinct $r$ values)")
axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
               title="noise half-width", title_fontsize=9)
fig.text(0.5, -0.035,
         "Continuous input, square loss, 320k steps, 32000 trajectories. Uniform noise of half-width 1/(2n)\n"
         "is exactly the error distribution of quantising to $n$ bins. Noise is baked into the dataset, not\n"
         "resampled, so a given occurrence is always corrupted the same way.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/noise.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/noise.png")

# ---- figure 2: noise against binning, matched scale -------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
scales = NB[1:]
nz = [mse(f"runs_noise/nb{n}_m{{m}}_seed0")[-2:].mean() for n in scales]
bn = [mse(BIN_SRC[n])[-2:].mean() for n in scales]
ax.plot(scales, bn, "o-", color="#b03a2e", ms=8, lw=1.8,
        label="quantised to $n$ bins  (deterministic)")
ax.plot(scales, nz, "s--", color="#1f3352", mfc="white", ms=8, lw=1.8,
        label="uniform noise of half-width 1/(2$n$)  (independent)")
ax.axhline(mse("runs_noise/nb0_m{m}_seed0")[-2:].mean(), color="0.55", lw=1, ls=":")
ax.text(8.6, mse("runs_noise/nb0_m{m}_seed0")[-2:].mean() * 1.3,
        "no corruption", fontsize=9.5, color="0.4")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(scales); ax.set_xticklabels([str(s) for s in scales])
ax.set_xlabel("corruption scale, as an equivalent bin count $n$")
ax.set_ylabel("mean squared error at large $m$, new tasks")
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(alpha=0.25, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, -0.06,
         "Same amount of corruption, delivered two ways. Averaged over the two largest task counts.\n"
         "Noise is independent at each position and can be averaged across the context; quantisation cannot.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/noise_vs_bins.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/noise_vs_bins.png")
print("\nplateau MSE at large m:  n   quantised     noise      ratio")
for n, b, z in zip(scales, bn, nz):
    print(f"                       {n:4d}  {b:.3e}  {z:.3e}  {b/z:6.2f}x")
