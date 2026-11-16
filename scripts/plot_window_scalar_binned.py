"""Plot correct-bin error after quantizing scalar-model predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from plot_window_heatmaps import predicted_fixed_point_overlap_edge


def load_runs(root):
    grouped = defaultdict(list)
    for path in sorted(Path(root).glob("win_w*_s*_seed*/eval_per_r.npz")):
        with (path.parent / "params.json").open() as handle:
            params = json.load(handle)
        with np.load(path) as result:
            if "scalar_bin_acc_per_r" not in result.files:
                continue
            grouped[float(params["width"])].append({
                "r": result["r_grid"].astype(float),
                "error": 1.0 - result["scalar_bin_acc_per_r"].astype(float),
                "start": float(params["start"]),
            })
    if not grouped:
        raise FileNotFoundError(
            "no scalar_bin_acc_per_r arrays found; run "
            "backfill_scalar_bin_accuracy.py first"
        )
    return grouped


def plot(grouped, output):
    widths = sorted(grouped, reverse=True)
    fig = plt.figure(figsize=(13.8, 8.6), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    images = []
    for ax, width in zip(axes, widths):
        runs = sorted(grouped[width], key=lambda run: run["start"])
        matrix = np.stack([run["error"] for run in runs])
        extent = [runs[0]["r"][0], runs[0]["r"][-1], len(runs), 0]
        image = ax.imshow(
            matrix, aspect="auto", interpolation="nearest", extent=extent,
            cmap="magma", vmin=0.0, vmax=1.0,
        )
        images.append(image)
        for row, run in enumerate(runs):
            ax.add_patch(Rectangle(
                (run["start"], row), width, 1, fill=False,
                edgecolor="white", linewidth=0.65, alpha=0.9,
            ))
            if np.isclose(run["start"], 3.0):
                edge = predicted_fixed_point_overlap_edge(width)
                ax.add_patch(Rectangle(
                    (edge, row), 3.0 - edge, 1, fill=False,
                    edgecolor="#00D5FF", linewidth=1.7,
                    linestyle=(0, (3, 2)), alpha=1.0,
                ))
        ax.set_title(f"Width {width:g} · {len(runs)} training windows")
        ax.set_xlabel(r"Evaluation parameter $r$")
        ax.set_ylabel("Training window (ordered by start)")
        ax.set_yticks([0.5, len(runs) - 0.5])
        ax.set_yticklabels([
            f"{runs[0]['start']:.3g}", f"{runs[-1]['start']:.3g}",
        ])
    colorbar = fig.colorbar(images[-1], ax=axes, pad=0.015, shrink=0.9)
    colorbar.set_label(
        "Incorrect-bin fraction after quantizing scalar predictions; "
        "white boxes are training ranges"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_win_fix_mse")
    parser.add_argument(
        "--output",
        default="figures_window/window_fixed_mse_binned_error.png",
    )
    args = parser.parse_args()
    plot(load_runs(args.runs), args.output)
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
