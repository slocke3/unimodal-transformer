"""The controlled task-diversity panel, one column per input resolution.

Reproduces the right-hand column of figures_taskdiv/task_diversity_controlled.png
-- seen tasks on top, new tasks below, against the number of distinct training
tasks, with total trajectories held at 32000 so only task count changes -- for
every input representation in the tokenisation sweep. Output is pinned at 64
bins throughout, so the columns differ only in what the model sees, and their
cross-entropies share one support and can be read against each other.

Two figures, because the sweep has two halves:
  _ce   cross-entropy in nats for the CE-trained arms, fixed-step against
        early-stopped, exactly the axes of the original.
  _rms  implied-map RMS for both losses, which is the only quantity defined for
        the square-loss arms and comparable across the two.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.maps import iterate_map

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
COLS = [("in8", "8 bins"), ("in16", "16 bins"), ("in32", "32 bins"),
        ("in64", "64 bins"), ("in128", "128 bins"), ("in256", "256 bins"),
        ("cont", "continuous")]
ORANGE, NAVY = "#d1502e", "#1f3352"


def load(itag, otag, key):
    out = []
    for m in MS:
        p = (f"runs_div_controlled/div_m{m}_seed0/eval_per_r.npz"
             if itag == "in64" and otag == "bins" else
             f"runs_divbins/{itag}_{otag}_m{m}_seed0/eval_per_r.npz")
        out.append(float(np.nanmean(np.load(p)[key])))
    return np.array(out)


def predict_the_mean_rms(n_r=60, n_traj=8, traj_len=150, burn=50, seed=0):
    """No-model reference for the RMS panels: the per-r spread of x_{n+1}, which
    is what you get by always predicting that r's mean state. The analogue of
    the uniform-over-64-bins line on the cross-entropy panels."""
    rng = np.random.default_rng(seed)
    out = []
    for r in np.linspace(0.5, 4.0, n_r):
        xs = np.concatenate([iterate_map(rng.uniform(.05, .95), r, traj_len)[burn:]
                             for _ in range(n_traj)])
        out.append(xs.std())
    return float(np.mean(out))


def grid(kind, fname):
    fig, axes = plt.subplots(2, len(COLS), figsize=(18.5, 6.6),
                             sharex=True, sharey="row")
    ref = np.log(64) if kind == "ce" else predict_the_mean_rms()
    for j, (itag, title) in enumerate(COLS):
        for i, (row_key, row_lab) in enumerate(
                (("at_train_r", "Seen tasks\n(evaluated at training $r$)"),
                 ("per_r", "New tasks\n(full-range grid)"))):
            ax = axes[i, j]
            ax.axhline(ref, color="0.6", lw=1, ls=":")
            if kind == "ce":
                series = ((f"ce_{row_key}", "bins", ORANGE, "o", ORANGE,
                           "Fixed steps (final)"),
                          (f"ce_{row_key}_bestval", "bins", NAVY, "s", "white",
                           "Early stopping (best val)"))
            else:
                series = ((f"rms_{row_key}", "bins", ORANGE, "o", ORANGE,
                           "Cross-entropy loss"),
                          (f"rms_{row_key}", "mse", NAVY, "s", "white",
                           "Square loss"))
            for key, otag, col, mk, mfc, lab in series:
                ax.plot(MS, load(itag, otag, key), mk + "-", color=col, mfc=mfc,
                        ms=5, lw=1.4, label=lab)
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(title, fontsize=12)
            if j == 0:
                ax.set_ylabel(row_lab + ("\nMean cross-entropy (nats)" if kind == "ce"
                                         else "\nImplied-map RMS"), fontsize=10)
            if i == 1:
                ax.set_xlabel("training tasks", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left")
    lab = ("uniform prediction over 64 bins" if kind == "ce"
           else "predicting each $r$'s mean state")
    fig.suptitle("Controlled task diversity by input resolution   |   total trajectories "
                 f"fixed at 32000, output fixed at 64 bins   |   dotted: {lab}",
                 fontsize=12.5, y=1.005)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"figures_taskdiv/{fname}.{e}", dpi=150, bbox_inches="tight")
    print(f"wrote figures_taskdiv/{fname}.png")


grid("ce", "taskdiv_by_bins_ce")
grid("rms", "taskdiv_by_bins_rms")
