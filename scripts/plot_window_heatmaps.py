"""Heatmap and distance-summary view of the sliding-window sweep."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def load_runs(runs_dir):
    grouped = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("win_w*_s*_seed*/eval_per_r.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        with np.load(path) as result:
            run = {
                "r": result["r_grid"],
                "ce": result["ce_per_r"],
                "start": float(params["start"]),
                "end": float(params["start"] + params["width"]),
            }
        grouped[float(params["width"])].append(run)
    if not grouped:
        raise FileNotFoundError(f"no window-sweep results found under {runs_dir}")
    return grouped


def transformed_ce(ce):
    return np.log10(1.0 + np.maximum(ce, 0.0))


def distance_summary(runs, n_bins=24):
    """Excess over an in-window model at the same r, binned by OOD distance."""
    r = runs[0]["r"]
    in_window_values = []
    for j, value in enumerate(r):
        values = [
            run["ce"][j] for run in runs
            if run["start"] - 1e-9 <= value <= run["end"] + 1e-9
        ]
        in_window_values.append(np.median(values) if values else np.nan)
    reference = np.asarray(in_window_values)

    distances, excesses = [], []
    for run in runs:
        distance = np.maximum.reduce((run["start"] - r, r - run["end"],
                                      np.zeros_like(r)))
        outside = (distance > 1e-9) & np.isfinite(reference)
        distances.extend(distance[outside])
        excesses.extend(np.maximum(run["ce"][outside] - reference[outside], 0.0))

    distances = np.asarray(distances)
    excesses = transformed_ce(np.asarray(excesses))
    edges = np.linspace(0.0, distances.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median = np.full(n_bins, np.nan)
    lower = np.full(n_bins, np.nan)
    upper = np.full(n_bins, np.nan)
    for i in range(n_bins):
        selected = (distances >= edges[i]) & (distances < edges[i + 1])
        if selected.any():
            lower[i], median[i], upper[i] = np.percentile(
                excesses[selected], [25, 50, 75]
            )
    return centers, median, lower, upper


def plot(grouped, output, color_scale="log", separate_scales=False):
    widths = sorted(grouped, reverse=True)
    transform = transformed_ce if color_scale == "log" else lambda ce: ce
    all_values = np.concatenate(
        [transform(run["ce"]) for runs in grouped.values() for run in runs]
    )
    shared_vmax = np.percentile(all_values, 99.5)

    fig = plt.figure(figsize=(13, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85])
    heat_axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]

    images = []
    for ax, width in zip(heat_axes, widths):
        runs = sorted(grouped[width], key=lambda run: run["start"])
        matrix = np.stack([transform(run["ce"]) for run in runs])
        vmax = np.percentile(matrix, 99.5) if separate_scales else shared_vmax
        extent = [runs[0]["r"][0], runs[0]["r"][-1], len(runs), 0]
        image = ax.imshow(
            matrix, aspect="auto", interpolation="nearest", extent=extent,
            cmap="magma", vmin=0.0, vmax=vmax,
        )
        images.append(image)
        for row, run in enumerate(runs):
            ax.add_patch(Rectangle(
                (run["start"], row), width, 1, fill=False,
                edgecolor="white", linewidth=0.65, alpha=0.9,
            ))
        ax.set_title(f"Width {width:g} · {len(runs)} training windows")
        ax.set_xlabel(r"Evaluation parameter $r$")
        ax.set_ylabel("Training window (ordered by start)")
        ax.set_yticks([0.5, len(runs) - 0.5])
        ax.set_yticklabels([
            f"{runs[0]['start']:.3g}",
            f"{runs[-1]['start']:.3g}",
        ])

    color_label = (
        r"$\log_{10}(1 + \mathrm{CE})$"
        if color_scale == "log" else "Cross-entropy (nats)"
    )
    if separate_scales:
        for ax, image in zip(heat_axes, images):
            colorbar = fig.colorbar(image, ax=ax, pad=0.015)
            colorbar.set_label(color_label)
    else:
        colorbar = fig.colorbar(images[-1], ax=heat_axes, pad=0.015, shrink=0.9)
        colorbar.set_label(f"{color_label}; white boxes are training ranges")

    summary_ax = fig.add_subplot(grid[2, :])
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(widths)))
    for width, color in zip(widths, colors):
        x, median, lower, upper = distance_summary(grouped[width])
        summary_ax.plot(x, median, color=color, lw=2, label=f"width {width:g}")
        summary_ax.fill_between(x, lower, upper, color=color, alpha=0.16)
    summary_ax.set_xlabel(r"Distance outside training interval in $r$")
    summary_ax.set_ylabel(r"Excess CE over in-window model at same $r$")
    summary_ax.set_yticks(np.log10(1 + np.array([0, 1, 10, 100])))
    summary_ax.set_yticklabels(["0", "1", "10", "100"])
    summary_ax.grid(alpha=0.25)
    summary_ax.legend(title="Training interval", ncols=4)
    summary_ax.set_title("Typical extrapolation penalty (median and interquartile range)")

    fig.suptitle(
        "Where does each window-trained transformer generalize?\n"
        f"8,000 trajectories per model; {color_scale} CE color scale"
        f"{' fitted separately per panel' if separate_scales else ''}; "
        "white boxes are training ranges",
        fontsize=14,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--output", default="figures/window_sweep_heatmaps.png")
    parser.add_argument("--color_scale", choices=["linear", "log"], default="log")
    parser.add_argument("--separate_scales", action="store_true")
    args = parser.parse_args()
    grouped = load_runs(args.runs_dir)
    plot(grouped, args.output, args.color_scale, args.separate_scales)
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
