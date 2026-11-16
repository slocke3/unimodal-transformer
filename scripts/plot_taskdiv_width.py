"""The square-loss task-diversity panel of taskdiv_gap, repeated across model widths.

One panel per d_model, each laid out like panel (b) of taskdiv_gap: task count on
x, mean squared error on y, one colour per input resolution, solid = new tasks
(300-point grid over the full range), dashed = seen tasks (training r values).

Only the width changes between panels: 4 layers, 4 heads (head size d_model/4),
feed-forward 4 x d_model, lr 3e-4, and each bin arm at the step count it has in
taskdiv_gap (160k for 8-32 bins, 320k for 64-256). Width 128 is the existing
binning sweep; the others are runs_width. One seed (seed 0) everywhere, including
width 128, so every panel carries the same single-run noise. Near each
transition single runs are bimodal -- seeds at width 128 differ 6-88x there --
so a one-grid-step difference between panels is within that noise.

Mean squared error is mean(rms**2) over the grid, not mean(rms)**2.

Torch-free. Run from the repo root:
  python scripts/plot_taskdiv_width.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.2,
})

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
TOTAL_TRAJ = 32_000
BINS = [8, 16, 32, 64, 128, 256]
WIDTHS = [32, 64, 128, 256]
PARAMS = {32: "0.05M", 64: "0.21M", 128: "0.81M", 256: "3.2M"}   # at 64 input bins


def run_dir(width, nbin, m, seed=0):
    if width == 128:   # the existing binning sweep behind taskdiv_gap
        base = "runs_divbins" if nbin <= 32 else "runs_divbins_320k"
        return f"{base}/in{nbin}_mse_m{m}_seed{seed}"
    return f"runs_width/d{width}_in{nbin}_mse_m{m}_seed{seed}"


def mse(width, nbin, key):
    out = []
    for m in MS:
        path = run_dir(width, nbin, m) + "/eval_per_r.npz"
        out.append(np.nanmean(np.load(path)[key] ** 2) if os.path.exists(path) else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_taskdiv/taskdiv_width.pdf")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.06, 0.92, len(BINS)))   # as in taskdiv_gap
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(20, 5.9), sharey=True)
    for ax, width in zip(axes, WIDTHS):
        for nbin, c in zip(BINS, colors):
            ax.plot(MS, mse(width, nbin, "rms_per_r"), "-o", ms=5, lw=2, color=c,
                    zorder=3, label=f"{nbin}")
            ax.plot(MS, mse(width, nbin, "rms_at_train_r"), "--", lw=1.5, color=c,
                    alpha=0.7, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"number of training tasks $m$")
        ax.set_title(f"width {width}  ·  {PARAMS[width]} parameters", pad=40)
        ax.grid(alpha=0.18, lw=0.5, which="both")
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(MS)
        sec.set_xticklabels([str(TOTAL_TRAJ // int(m)) for m in MS], fontsize=11)
        sec.set_xlabel("allocated trajectories per task", fontsize=12)
    axes[0].set_ylabel("mean squared error")

    # Legends go under the panels: inside, the lower-left corner is where the
    # seen-task floors of the finest binnings sit.
    h, l = axes[0].get_legend_handles_labels()
    style = [plt.Line2D([], [], color="0.25", ls="-", lw=2, marker="o", ms=5),
             plt.Line2D([], [], color="0.25", ls="--", lw=1.5)]
    fig.legend(h, l, title="input bins", title_fontsize=12.5, ncol=len(BINS),
               loc="upper center", bbox_to_anchor=(0.38, 0.0), frameon=False)
    fig.legend(style, ["new tasks", "seen tasks"], ncol=2, loc="upper center",
               bbox_to_anchor=(0.76, 0.0), frameon=False)

    fig.text(0.5, -0.15,
             "Square loss, scalar output, one seed per point. Only the model width changes: 4 layers, 4 heads, "
             "feed-forward 4× width; 8–32 bins trained 160k steps, 64–256 bins 320k, as in taskdiv_gap.",
             ha="center", fontsize=12, color="0.25")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
