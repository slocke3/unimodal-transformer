"""The controlled task-diversity panel, one column per input resolution.

Reproduces the right-hand column of figures_taskdiv/task_diversity_controlled.png
-- seen tasks on top, new tasks below, against the number of distinct training
tasks, with total trajectories held at 32000 so only task count changes -- for
every input representation in the tokenisation sweep. Output is pinned at 64
bins throughout, so the columns differ only in what the model sees, and their
cross-entropies share one support and can be read against each other.

For the square-loss arms there is no output binning at all: the head is a single
scalar and the target is the exact next state, the same real number for every
arm. --n_bins_out is passed to those runs but is inert -- parameter counts and
targets are identical whatever it is set to -- so they are output-matched in a
stronger sense than the cross-entropy arms, not a weaker one.

Two figures, because the sweep has two halves:
  _ce   cross-entropy in nats for the CE-trained arms, fixed-step against
        early-stopped, exactly the axes of the original.
  _loss each arm's OWN training objective at the fixed-step checkpoint --
        cross-entropy in nats for the CE-trained arms, mean squared error for
        the square-loss ones. The two are different objectives, so only the
        SHAPE and the location of the drop transfer between the colours; the
        levels do not, and each carries its own no-model reference line.
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


def load(itag, otag, key, square=False):
    """square=True averages the SQUARED per-r error, which is the mean squared
    error over the grid. Averaging the per-r RMS and squaring afterwards is a
    different and smaller quantity by Jensen -- between 1.5x and 9x smaller here,
    and by an amount that varies with the arm, so it distorts curve shapes."""
    out = []
    for m in MS:
        p = (f"runs_div_controlled/div_m{m}_seed0/eval_per_r.npz"
             if itag == "in64" and otag == "bins" else
             f"runs_divbins/{itag}_{otag}_m{m}_seed0/eval_per_r.npz")
        out.append(float(np.nanmean(np.load(p)[key])))
    return np.array(out)


def uniform_guess_mse(n_r=60, n_traj=8, traj_len=150, burn=50, seed=0):
    """The square-loss analogue of the uniform-over-64-bins cross-entropy line.

    A flat distribution on [0,1] has mean 1/2, so the point prediction it implies
    is the constant 1/2, and its MSE is E[(x - 1/2)^2] over the evaluation data.
    Both baselines then answer the same question -- what a model that has learnt
    nothing at all achieves -- which is the convention of the original figure.

    Not the per-r variance: predicting each r's own mean state already knows
    which task it is in, which is a far stronger reference and not the analogue
    of a uniform distribution.
    """
    rng = np.random.default_rng(seed)
    xs = np.concatenate([
        np.concatenate([iterate_map(rng.uniform(.05, .95), r, traj_len)[burn:]
                        for _ in range(n_traj)])
        for r in np.linspace(0.5, 4.0, n_r)])
    return float(((xs - 0.5) ** 2).mean())


def grid(kind, fname):
    fig, axes = plt.subplots(2, len(COLS), figsize=(18.5, 6.6),
                             sharex=True, sharey="row")
    ref_ce, ref_mse = np.log(64), uniform_guess_mse()
    for j, (itag, title) in enumerate(COLS):
        for i, (row_key, row_lab) in enumerate(
                (("at_train_r", "Seen tasks\n(evaluated at training $r$)"),
                 ("per_r", "New tasks\n(full-range grid)"))):
            ax = axes[i, j]
            if kind == "ce":
                ax.axhline(ref_ce, color="0.6", lw=1, ls=":")
                series = ((f"ce_{row_key}", "bins", ORANGE, "o", ORANGE, 1.0,
                           "Fixed steps (final)"),
                          (f"ce_{row_key}_bestval", "bins", NAVY, "s", "white", 1.0,
                           "Early stopping (best val)"))
            else:
                # different objectives, so each gets its own no-model baseline;
                # both are "has learnt nothing", not "knows which r it is in"
                ax.axhline(ref_ce, color=ORANGE, lw=1, ls=":", alpha=0.55)
                ax.axhline(ref_mse, color=NAVY, lw=1, ls=":", alpha=0.55)
                series = ((f"ce_{row_key}", "bins", ORANGE, "o", ORANGE, 1.0,
                           "Cross-entropy loss (nats)"),
                          (f"rms_{row_key}", "mse", NAVY, "s", "white", 2.0,
                           "Square loss (MSE)"))
            for key, otag, col, mk, mfc, power, lab in series:
                ax.plot(MS, load(itag, otag, key, square=(power == 2.0)), mk + "-", color=col,
                        mfc=mfc, ms=5, lw=1.4, label=lab)
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(title, fontsize=12)
            if j == 0:
                ax.set_ylabel(row_lab + ("\nMean cross-entropy (nats)" if kind == "ce"
                                         else "\nTraining objective"), fontsize=10)
            if i == 1:
                ax.set_xlabel("training tasks", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left")
    lab = ("dotted: uniform prediction over 64 bins" if kind == "ce" else
           "dotted: what a model that learnt nothing gets -- uniform over 64 bins, "
           "or the constant $x=1/2$")
    fig.suptitle("Controlled task diversity by input resolution   |   total trajectories "
                 f"fixed at 32000, output held identical across arms   |   {lab}",
                 fontsize=12.5, y=1.005)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"figures_taskdiv/{fname}.{e}", dpi=150, bbox_inches="tight")
    print(f"wrote figures_taskdiv/{fname}.png")


def overlay(fname="taskdiv_overlay"):
    """All binnings on shared axes, one column per loss, coloured by resolution.

    The per-resolution grid puts each binning in its own panel, which makes the
    shape of any one curve clear but leaves the ordering between them to be read
    across a page. Overlaying them shows the ordering directly: where the curves
    sit relative to each other, and where their drops fall.

    Continuous input is left out. It is not a point on this ladder -- a bin
    embedding is a lookup table with no notion that neighbouring bins are near,
    while Linear(1, d) has that metric built in -- so placing it on a resolution
    axis would imply a limit it does not take.
    """
    bins = [("in8", 8), ("in16", 16), ("in32", 32), ("in64", 64),
            ("in128", 128), ("in256", 256)]
    cmap = plt.get_cmap("viridis")
    cols = [cmap(v) for v in np.linspace(0.12, 0.88, len(bins))]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True)
    ref = {"bins": np.log(64), "mse": uniform_guess_mse()}
    for j, (otag, ctitle, ylab, power) in enumerate(
            (("bins", "Cross-entropy loss", "Mean cross-entropy (nats)", 1.0),
             ("mse", "Square loss", "Mean squared error", 2.0))):
        for i, (row_key, row_lab) in enumerate(
                (("at_train_r", "Seen tasks\n(evaluated at training $r$)"),
                 ("per_r", "New tasks\n(full-range grid)"))):
            ax = axes[i, j]
            ax.axhline(ref[otag], color="0.55", lw=1, ls=":")
            key = ("ce_" if otag == "bins" else "rms_") + row_key
            for (itag, nb), c in zip(bins, cols):
                ax.plot(MS, load(itag, otag, key, square=(power == 2.0)), "o-", color=c,
                        ms=5, lw=1.6, label=str(nb))
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(alpha=0.25, lw=0.5)
            if i == 0:
                ax.set_title(ctitle, fontsize=13)
            if i == 1:
                ax.set_xlabel("number of training tasks (distinct $r$ values)")
            ax.set_ylabel((row_lab + "\n" + ylab) if j == 0 else ylab, fontsize=10)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        # share y down each column so the two rows are directly comparable
        # (the two columns are different objectives and must NOT share)
        lo = min(axes[i, j].get_ylim()[0] for i in (0, 1))
        hi = max(axes[i, j].get_ylim()[1] for i in (0, 1))
        for i in (0, 1):
            axes[i, j].set_ylim(lo, hi)
    axes[1, 0].legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
                      title="input bins", title_fontsize=9.5)
    fig.text(0.5, -0.035,
             "Output held identical across arms and total trajectories at 32000 throughout, so only the input resolution "
             "changes.  Fixed-step checkpoint.\nDotted: what a model that learnt nothing achieves.  "
             "Continuous input is excluded -- it is a different inductive bias, not the fine end of this ladder.",
             ha="center", fontsize=9.5, color="0.25")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"figures_taskdiv/{fname}.{e}", dpi=170, bbox_inches="tight")
    print(f"wrote figures_taskdiv/{fname}.png")


grid("ce", "taskdiv_by_bins_ce")
grid("loss", "taskdiv_by_bins_loss")
overlay()
