"""Hénon pilot: does the task-diversity crossover appear in a second family?

Hénon map x' = 1 - a x^2 + y, y' = 0.3 x, a in [0, 1.4], only x observed
(rescaled to [0, 1], src/maps.py). Same recipe as the logistic sweep: 32k
trajectories, context 50, 64 output bins, cross-entropy, seed 0; 160k steps at
32 input bins, 320k at 128 (runs_henon, trained on RunPod).

  (a) new-task and seen-task CE against m, Hénon at 32 and 128 input bins, with
      the logistic runs at the same resolutions (seed 0) in grey.
  (b) new-task CE by regime of a at 128 input bins.
  (c) CE across a at 128 input bins, m = 100 and m = 8000.

Torch-free:  python scripts/rebuttal/plot_henon_pilot.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15.5, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "axes.linewidth": 1.3,
})

MS = [100, 500, 2000, 8000]
NS = [32, 128]
LOGISTIC = {32: "runs_divbins/in32_bins_m{m}_seed0", 128: "runs_divbins_320k/in128_bins_m{m}_seed0"}
REGIMES = [("fixed point, a < 0.37", 0.0, 0.3675), ("period doubling, 0.37-1.06", 0.3675, 1.06),
           ("chaotic band, a > 1.06", 1.06, 1.41)]


def load(p):
    f = p + "/eval_per_r.npz"
    return np.load(f) if os.path.exists(f) else None


def main():
    out = "figures_rebuttal/henon_pilot.pdf"
    os.makedirs("figures_rebuttal", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    cols = {32: "#1f77b4", 128: "#d62728"}
    lines = ["Henon pilot (seed 0), CE in nats"]

    ax = axes[0]
    for n in NS:
        new, seen, lnew = [], [], []
        for m in MS:
            d = load(f"runs_henon/in{n}_bins_m{m}_seed0")
            new.append(np.nanmean(d["ce_per_r"]))
            seen.append(np.nanmean(d["ce_at_train_r"]))
            dl = load(LOGISTIC[n].format(m=m))
            lnew.append(np.nanmean(dl["ce_per_r"]) if dl is not None else np.nan)
        ax.plot(MS, new, "-o", color=cols[n], lw=2.2, ms=6, label=f"Hénon {n} bins, new tasks")
        ax.plot(MS, seen, "--o", color=cols[n], lw=1.6, ms=5, mfc="white",
                label=f"Hénon {n} bins, training tasks")
        ax.plot(MS, lnew, ":", marker="s", color=cols[n], alpha=0.45, lw=1.6, ms=5,
                label=f"logistic {n} bins, new tasks")
        lines.append(f"N={n}: " + "  ".join(f"m={m}: new {a:.3f} seen {b:.3f} (logistic new {c:.3f})"
                                           for m, a, b, c in zip(MS, new, seen, lnew)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training tasks $m$"); ax.set_ylabel("cross-entropy (nats)")
    ax.set_title("(a) task-diversity curves, Hénon vs logistic")
    ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=10.5)

    ax = axes[1]
    for (nm, lo, hi), ls in zip(REGIMES, ["-", "--", ":"]):
        v = []
        for m in MS:
            d = load(f"runs_henon/in128_bins_m{m}_seed0")
            a = d["r_grid"]
            v.append(np.nanmean(d["ce_per_r"][(a >= lo) & (a < hi)]))
        ax.plot(MS, v, ls, marker="o", color="0.15", lw=2.2, label=nm)
        lines.append(f"N=128 {nm}: " + "  ".join(f"m={m}: {x:.3f}" for m, x in zip(MS, v)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training tasks $m$"); ax.set_ylabel("new-task cross-entropy (nats)")
    ax.set_title("(b) by regime, 128 input bins")
    ax.grid(alpha=0.2, which="both"); ax.legend()

    ax = axes[2]
    for m, c in ((100, "#9467bd"), (8000, "#2ca02c")):
        d = load(f"runs_henon/in128_bins_m{m}_seed0")
        ax.plot(d["r_grid"], d["ce_per_r"], color=c, lw=1.6, label=f"m = {m}")
        if m == 100:
            ax.plot(d["train_r"], np.full(len(d["train_r"]), 1.6e-4), "|", color=c, ms=10,
                    alpha=0.6, label="training a, m = 100")
    for _, lo, _ in REGIMES[1:]:
        ax.axvline(lo, color="0.6", lw=1, ls="--")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 30)
    ax.set_xlabel("$a$"); ax.set_ylabel("cross-entropy (nats, clipped at 1e-4)")
    ax.set_title("(c) new-task CE across $a$, 128 input bins")
    ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    with open(out.replace(".pdf", "_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
