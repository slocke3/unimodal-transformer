"""Plot all completed sliding-window sweep evaluations in one figure."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


def load_runs(runs_dir):
    """Load window-sweep evaluation files, grouped by training-window width."""
    grouped = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("win_w*_s*_seed*/eval_per_r.npz")):
        with np.load(path) as result:
            r_grid = result["r_grid"]
            ce = result["ce_per_r"]
            in_window = result["in_window"]
        if not (len(r_grid) == len(ce) == len(in_window)):
            raise ValueError(f"inconsistent arrays in {path}")
        width = float(path.parent.name.split("_")[1][1:])
        start = float(r_grid[in_window].min())
        grouped[width].append(
            {"path": path, "start": start, "r_grid": r_grid, "ce": ce,
             "in_window": in_window}
        )

    if not grouped:
        raise FileNotFoundError(f"no window-sweep eval files found in {runs_dir}")
    return grouped


def plot(grouped, output_path, y_scale):
    widths = sorted(grouped, reverse=True)
    ncols = 2
    nrows = int(np.ceil(len(widths) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()

    starts = [run["start"] for runs in grouped.values() for run in runs]
    norm = Normalize(vmin=min(starts), vmax=max(starts))
    cmap = plt.get_cmap("viridis")

    for ax, width in zip(axes, widths):
        runs = sorted(grouped[width], key=lambda run: run["start"])
        for run in runs:
            color = cmap(norm(run["start"]))
            # Full evaluation curve is faint; the bold section is the region
            # represented in that model's training distribution.
            ax.plot(run["r_grid"], run["ce"], color=color, alpha=0.38, lw=0.8)
            ax.plot(
                run["r_grid"][run["in_window"]],
                run["ce"][run["in_window"]],
                color=color,
                lw=2.1,
            )
        ax.set_title(f"Training-window width = {width:g} ({len(runs)} models)")
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_yscale(y_scale)
        ax.set_xlim(0.5, 4.0)

    for ax in axes[len(widths):]:
        ax.remove()
    for ax in axes[:len(widths)]:
        ax.set_xlabel(r"Logistic-map parameter $r$")
        ax.set_ylabel(f"Next-token cross-entropy (nats; {y_scale} scale)")

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axes[:len(widths)], pad=0.02
    )
    colorbar.set_label("Training-window start $r$")
    fig.suptitle(
        "Sliding-window generalization across the logistic family\n"
        "8,000 training trajectories per model; bold segments are in-window",
        fontsize=14,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--output", default="figures/window_sweep_unified.png")
    parser.add_argument("--y_scale", choices=["linear", "log"], default="log")
    args = parser.parse_args()

    grouped = load_runs(args.runs_dir)
    print(
        "loaded "
        f"{sum(len(runs) for runs in grouped.values())} runs: "
        + ", ".join(f"w={width:g}: {len(grouped[width])}" for width in sorted(grouped))
    )
    plot(grouped, args.output, args.y_scale)
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
