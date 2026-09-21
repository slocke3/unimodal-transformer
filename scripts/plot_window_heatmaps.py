"""Heatmap and distance-summary view of the sliding-window sweep.

--metric ce reads the cross-entropy of the binned-output runs; --metric mse reads
the square-loss runs, whose stored arrays are per-r RMS, so mean squared error is
rms**2.

The two need different handling and not just a different label. log10(1 + x)
suits cross-entropy, which is order 1 and additive in nats, but is useless for a
mean squared error of order 1e-6, where it returns nearly zero everywhere; MSE is
shown as log10 of itself with limits from percentiles. In the summary panel the
excess over an in-window model is a difference in nats for cross-entropy and a
ratio for MSE, since a difference of 1e-6 means nothing without knowing the floor,
which varies over orders of magnitude across the sweep.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# key in eval_per_r.npz, and the power that turns it into the plotted quantity:
# the square-loss runs store per-r RMS, so the mean squared error is its square
METRICS = {
    "ce":  dict(key="ce_per_r",  power=1),
    "mse": dict(key="rms_per_r", power=2),
}


def heat_transform(values, metric):
    if metric == "ce":
        return np.log10(1.0 + np.maximum(values, 0.0))
    return np.log10(np.maximum(values, 1e-12))


def heat_vmin(all_values, metric):
    return 0.0 if metric == "ce" else float(np.percentile(all_values, 0.5))


def color_label_for(metric, color_scale):
    if metric == "ce":
        return (r"$\log_{10}(1 + \mathrm{CE})$" if color_scale == "log"
                else "Cross-entropy (nats)")
    return (r"$\log_{10}(\mathrm{MSE})$" if color_scale == "log"
            else "Mean squared error")


def load_runs(runs_dir, metric="ce"):
    grouped = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("win_w*_s*_seed*/eval_per_r.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        spec = METRICS[metric]
        with np.load(path) as result:
            if spec["key"] not in result.files:
                continue
            run = {
                "r": result["r_grid"],
                "ce": result[spec["key"]] ** spec["power"],
                "start": float(params["start"]),
                "end": float(params["start"] + params["width"]),
            }
        grouped[float(params["width"])].append(run)
    if not grouped:
        raise FileNotFoundError(f"no window-sweep results found under {runs_dir}")
    return grouped


def distance_summary(runs, metric="ce", n_bins=24):
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
        if metric == "ce":                      # additive, in nats
            excesses.extend(np.maximum(run["ce"][outside] - reference[outside], 0.0))
        else:                                   # a ratio: the floor varies hugely
            excesses.extend(np.maximum(run["ce"][outside] / reference[outside], 1.0))

    distances = np.asarray(distances)
    excesses = (np.log10(1.0 + np.asarray(excesses)) if metric == "ce"
                else np.log10(np.asarray(excesses)))
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


def plot(grouped, output, color_scale="log", separate_scales=False, metric="ce"):
    widths = sorted(grouped, reverse=True)
    transform = ((lambda v: heat_transform(v, metric)) if color_scale == "log"
                 else (lambda v: v))
    all_values = np.concatenate(
        [transform(run["ce"]) for runs in grouped.values() for run in runs]
    )
    shared_vmax = np.percentile(all_values, 99.5)
    shared_vmin = heat_vmin(all_values, metric) if color_scale == "log" else 0.0

    fig = plt.figure(figsize=(13, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85])
    heat_axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]

    images = []
    for ax, width in zip(heat_axes, widths):
        runs = sorted(grouped[width], key=lambda run: run["start"])
        matrix = np.stack([transform(run["ce"]) for run in runs])
        vmax = np.percentile(matrix, 99.5) if separate_scales else shared_vmax
        vmin = (heat_vmin(matrix, metric) if (separate_scales and color_scale == "log")
                else shared_vmin)
        extent = [runs[0]["r"][0], runs[0]["r"][-1], len(runs), 0]
        image = ax.imshow(
            matrix, aspect="auto", interpolation="nearest", extent=extent,
            cmap="magma", vmin=vmin, vmax=vmax,
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

    color_label = color_label_for(metric, color_scale)
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
        x, median, lower, upper = distance_summary(grouped[width], metric)
        summary_ax.plot(x, median, color=color, lw=2, label=f"width {width:g}")
        summary_ax.fill_between(x, lower, upper, color=color, alpha=0.16)
    summary_ax.set_xlabel(r"Distance outside training interval in $r$")
    if metric == "ce":
        # values are log10(1 + excess in nats), so the ticks are placed there
        summary_ax.set_ylabel(r"Excess CE over in-window model at same $r$")
        summary_ax.set_yticks(np.log10(1 + np.array([0, 1, 10, 100])))
        summary_ax.set_yticklabels(["0", "1", "10", "100"])
    else:
        # values are log10 of a ratio, so decades are the natural ticks; the
        # cross-entropy placement above would put them in the wrong places
        summary_ax.set_ylabel("MSE relative to in-window model at same $r$")
        summary_ax.set_yticks([0, 1, 2, 3, 4])
        summary_ax.set_yticklabels([r"$1\times$", r"$10\times$", r"$10^2\times$",
                                    r"$10^3\times$", r"$10^4\times$"])
    summary_ax.grid(alpha=0.25)
    summary_ax.legend(title="Training interval", ncols=4)
    summary_ax.set_title("Typical extrapolation penalty (median and interquartile range)")
    summary_ax.set_ylim(bottom=0)

    fig.suptitle(
        "Where does each window-trained transformer generalize?\n"
        f"8,000 trajectories per model; {color_scale} "
        f"{'cross-entropy' if metric == 'ce' else 'mean squared error'} colour scale"
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
    parser.add_argument("--metric", choices=["ce", "mse"], default="ce")
    parser.add_argument("--output", default="figures/window_sweep_heatmaps.png")
    parser.add_argument("--color_scale", choices=["linear", "log"], default="log")
    parser.add_argument("--separate_scales", action="store_true")
    args = parser.parse_args()
    grouped = load_runs(args.runs_dir, args.metric)
    plot(grouped, args.output, args.color_scale, args.separate_scales, args.metric)
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
