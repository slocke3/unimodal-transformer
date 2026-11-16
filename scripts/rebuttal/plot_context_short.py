"""Context length down to L = 1 (reviewer FWzs: "even shorter contexts, L in {1,2,5}?").

Square loss, 64 input bins, 32k trajectories, trajectory length L+100, 320k
steps -- the context ladder of the paper (runs_ctx, plus the binning sweep's
in64_mse arm at L=50) extended with L = 1, 2, 5 (runs_ctx_short, trained on
RunPod with --device_data).

  (a) new-task MSE against L, one line per task count m (median over seeds,
      range shaded). Grey marks at m = 8000: the k-gram baselines with k = 1, 2, 4
      (scripts/rebuttal/baselines_taskdiv.py), i.e. counting what follows the
      last k tokens. A transformer with L = 1 cannot beat the 1-gram's
      bin-conditional mean, so the two should meet there.
  (b) new-task MSE against m for L = 1, 2, 5, 10, 50.
Hollow diamonds in (b): L = 10 runs repeated with the RunPod loader
(runs_ctx_short/L10check_*), to be read against the original L = 10 curve.

Torch-free:  python scripts/rebuttal/plot_context_short.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15.5, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
    "axes.linewidth": 1.3,
})

MS = [100, 150, 250, 350, 500, 700, 1000, 1400, 2000, 4000, 8000]
CTX = [1, 2, 5, 10, 20, 25, 30, 35, 50, 60, 75]
SEEDS = [0, 1, 2]
OLD_M = [100, 250, 500, 1000, 2000, 4000, 8000]


def path(L, m, s):
    if L <= 5:
        return f"runs_ctx_short/L{L}_m{m}_seed{s}"
    if L == 50 and m in OLD_M:
        return f"runs_divbins_320k/in64_mse_m{m}_seed{s}"
    return f"runs_ctx/L{L}_m{m}_seed{s}"


def mse(p, key="rms_per_r"):
    f = p + "/eval_per_r.npz"
    return float(np.nanmean(np.load(f)[key] ** 2)) if os.path.exists(f) else np.nan


def grid(key="rms_per_r"):
    return np.array([[[mse(path(L, m, s), key) for m in MS] for L in CTX] for s in SEEDS])


def main():
    out = "figures_rebuttal/context_short.pdf"
    os.makedirs("figures_rebuttal", exist_ok=True)
    y = grid()                                            # (seed, L, m)
    with np.errstate(all="ignore"):
        med, lo, hi = np.nanmedian(y, 0), np.nanmin(y, 0), np.nanmax(y, 0)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))

    ax = axes[0]
    show_m = [100, 250, 500, 1000, 2000, 8000]
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_m)))
    for m, c in zip(show_m, cols):
        j = MS.index(m)
        ok = np.isfinite(med[:, j])
        ax.fill_between(np.array(CTX)[ok], lo[ok, j], hi[ok, j], color=c, alpha=0.15, lw=0)
        ax.plot(np.array(CTX)[ok], med[ok, j], "-o", color=c, lw=2.2, ms=5, label=f"m = {m}")
    bl = sorted(glob.glob("runs_rebuttal_baselines/m8000_seed*.npz"))
    if bl:
        for k, mk in ((1, "v"), (2, "s"), (4, "D")):
            v = np.median([np.load(f)[f"N64_k{k}_new_mse"].mean() for f in bl])
            ax.plot([k], [v], mk, color="0.45", ms=10, mfc="none", mew=2,
                    label=f"{k}-gram, m = 8000" if k == 1 else None)
            ax.annotate(f"{k}-gram", (k, v), textcoords="offset points", xytext=(8, 6),
                        color="0.4", fontsize=12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(CTX); ax.set_xticklabels([str(c) for c in CTX], fontsize=12)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("context length $L$ (trained at that length)")
    ax.set_ylabel("new-task mean squared error")
    ax.set_title("(a) error against context length, 64 input bins")
    ax.grid(alpha=0.2, which="both")
    ax.legend(ncol=2, fontsize=12)

    ax = axes[1]
    show_L = [1, 2, 5, 10, 50]
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(show_L)))
    for L, c in zip(show_L, cols):
        i = CTX.index(L)
        ok = np.isfinite(med[i])
        ax.fill_between(np.array(MS)[ok], lo[i, ok], hi[i, ok], color=c, alpha=0.15, lw=0)
        ax.plot(np.array(MS)[ok], med[i, ok], "-o", color=c, lw=2.2, ms=5, label=f"L = {L}")
    chk = [(m, mse(f"runs_ctx_short/L10check_m{m}_seed0")) for m in (250, 1000, 4000)]
    chk = [(m, v) for m, v in chk if np.isfinite(v)]
    if chk:
        ax.plot(*zip(*chk), "D", ms=11, mfc="none", mec=cols[show_L.index(10)], mew=2,
                label="L = 10, RunPod loader (seed 0)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training tasks $m$")
    ax.set_ylabel("new-task mean squared error")
    ax.set_title("(b) task-diversity curves by context length")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    lines = ["new-task MSE, median over seeds (n seeds)"]
    for i, L in enumerate(CTX):
        n = np.sum(np.isfinite(y[:, i, :]), 0)
        lines.append(f"L={L:2d} " + " ".join(f"{m}:{med[i, j]:.2e}({n[j]})"
                                             for j, m in enumerate(MS)))
    for m, v in chk:
        lines.append(f"L10check m={m}: {v:.2e} vs original seed0 {mse(f'runs_ctx/L10_m{m}_seed0'):.2e}")
    with open(out.replace(".pdf", "_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
