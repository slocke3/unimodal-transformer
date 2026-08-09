"""Plot CE, position-histogram overlap, and their relationship for all sweeps."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import spearmanr


def load_runs(root, name_prefix=None):
    runs = []
    for path in sorted(Path(root).glob("*/eval_per_r.npz")):
        if name_prefix and not path.parent.name.startswith(name_prefix):
            continue
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        with np.load(path) as result:
            required = {"r_grid", "ce_per_r", "hist_overlap_per_r"}
            missing = required.difference(result.files)
            if missing:
                raise ValueError(f"{path} is missing histogram arrays: {sorted(missing)}")
            runs.append({
                "name": path.parent.name,
                "start": float(params["start"]),
                "end": float(params["start"] + params["width"]),
                "width": float(params["width"]),
                "r": result["r_grid"].copy(),
                "ce": result["ce_per_r"].copy(),
                "overlap": result["hist_overlap_per_r"].copy(),
            })
    if not runs:
        suffix = f" matching {name_prefix!r}" if name_prefix else ""
        raise FileNotFoundError(f"no completed runs under {root}{suffix}")
    return runs


def ce_color(ce):
    return np.log10(1.0 + np.maximum(ce, 0.0))


def interval_ticks(runs, max_ticks=7):
    n = len(runs)
    indices = np.unique(np.linspace(0, n - 1, min(n, max_ticks)).round().astype(int))
    labels = [f"[{runs[i]['start']:.2f}, {runs[i]['end']:.2f}]" for i in indices]
    return indices + 0.5, labels


def draw_heatmap_pair(ax_ce, ax_overlap, runs, ce_vmax, title):
    runs = sorted(runs, key=lambda run: (run["start"], run["end"]))
    r = runs[0]["r"]
    ce = np.stack([ce_color(run["ce"]) for run in runs])
    overlap = np.stack([run["overlap"] for run in runs])
    extent = [r[0], r[-1], len(runs), 0]

    ce_image = ax_ce.imshow(
        ce, aspect="auto", interpolation="nearest", extent=extent,
        cmap="magma", vmin=0.0, vmax=ce_vmax,
    )
    overlap_image = ax_overlap.imshow(
        overlap, aspect="auto", interpolation="nearest", extent=extent,
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    for row, run in enumerate(runs):
        for ax in (ax_ce, ax_overlap):
            ax.add_patch(Rectangle(
                (run["start"], row), run["end"] - run["start"], 1,
                fill=False, edgecolor="white", linewidth=0.7,
            ))

    ticks, labels = interval_ticks(runs)
    for ax in (ax_ce, ax_overlap):
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(r[0], r[-1])
        ax.set_xlabel(r"Evaluation parameter $r$")
    ax_ce.set_ylabel("Training interval")
    ax_overlap.set_yticklabels([])
    ax_ce.set_title(f"{title}: prediction CE")
    ax_overlap.set_title(f"{title}: train/eval occupancy overlap")
    return ce_image, overlap_image


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_window_overview(runs, output):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["width"]].append(run)
    widths = sorted(grouped, reverse=True)
    all_ce = np.concatenate([ce_color(run["ce"]) for run in runs])
    ce_vmax = np.percentile(all_ce, 99.5)

    fig, axes = plt.subplots(
        len(widths), 2, figsize=(13, 3.0 * len(widths)),
        layout="constrained", squeeze=False,
    )
    ce_image = overlap_image = None
    for row, width in enumerate(widths):
        ce_image, overlap_image = draw_heatmap_pair(
            axes[row, 0], axes[row, 1], grouped[width], ce_vmax,
            f"Window width {width:g}",
        )
    fig.colorbar(ce_image, ax=axes[:, 0], label=r"$\log_{10}(1+\mathrm{CE})$",
                 pad=0.01)
    fig.colorbar(overlap_image, ax=axes[:, 1], label="Histogram intersection",
                 pad=0.01)
    fig.suptitle(
        "Window sweep: performance and x-position coverage\n"
        "White boxes mark each model's training interval",
        fontsize=14,
    )
    save_figure(fig, output)


def plot_boundary_overview(runs, kind, output):
    runs = sorted(
        runs,
        key=(lambda run: run["end"]) if kind == "rmax"
        else (lambda run: run["start"]),
    )
    ce_vmax = np.percentile(
        np.concatenate([ce_color(run["ce"]) for run in runs]), 99.5
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    label = r"$r_{\max}$ prefix" if kind == "rmax" else r"$r_{\min}$ suffix"
    ce_image, overlap_image = draw_heatmap_pair(
        axes[0], axes[1], runs, ce_vmax, label,
    )
    fig.colorbar(ce_image, ax=axes[0], label=r"$\log_{10}(1+\mathrm{CE})$",
                 pad=0.01)
    fig.colorbar(overlap_image, ax=axes[1], label="Histogram intersection",
                 pad=0.01)
    fig.suptitle(
        f"{label} sweep: performance and x-position coverage\n"
        "White boxes mark each model's training interval",
        fontsize=14,
    )
    save_figure(fig, output)


def add_correlation_panel(ax, runs, in_distribution, title):
    overlaps, losses = [], []
    for run in runs:
        inside = ((run["r"] >= run["start"] - 1e-9)
                  & (run["r"] <= run["end"] + 1e-9))
        selected = inside if in_distribution else ~inside
        overlaps.extend(run["overlap"][selected])
        losses.extend(run["ce"][selected])
    overlaps = np.asarray(overlaps)
    losses = np.asarray(losses)
    plotted_losses = ce_color(losses)
    color = "#1B5E9A" if in_distribution else "#D85A30"

    ax.scatter(overlaps, plotted_losses, s=7, alpha=0.13, color=color,
               linewidths=0)
    edges = np.linspace(0.0, 1.0, 13)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        mask = (overlaps >= edges[i]) & (overlaps < edges[i + 1])
        if mask.sum() >= 5:
            medians[i] = np.median(plotted_losses[mask])
    ax.plot(centers, medians, color=color, lw=2.2, marker="o", ms=3,
            label="Binned median")

    valid = np.isfinite(overlaps) & np.isfinite(losses)
    rho = spearmanr(overlaps[valid], losses[valid]).statistic if valid.sum() > 1 else np.nan
    ax.text(
        0.03, 0.96, rf"Spearman $\rho={rho:.2f}$" + f"\nn={valid.sum():,}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Train/eval x-histogram intersection")
    ax.set_yticks(np.log10(1.0 + np.array([0, 1, 10, 100])))
    ax.set_yticklabels(["0", "1", "10", "100"])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower left")


def plot_correlations(runs, output, window_facets=False):
    if window_facets:
        grouped = defaultdict(list)
        for run in runs:
            grouped[run["width"]].append(run)
        groups = [(f"Window width {width:g}", grouped[width])
                  for width in sorted(grouped, reverse=True)]
    else:
        groups = [("", runs)]

    fig, axes = plt.subplots(
        len(groups), 2, figsize=(12, 3.2 * len(groups)),
        layout="constrained", squeeze=False, sharex=True, sharey=True,
    )
    for row, (group_label, group_runs) in enumerate(groups):
        prefix = f"{group_label} · " if group_label else ""
        add_correlation_panel(
            axes[row, 0], group_runs, True, prefix + "in distribution",
        )
        add_correlation_panel(
            axes[row, 1], group_runs, False, prefix + "out of distribution",
        )
        axes[row, 0].set_ylabel(r"Prediction CE (nats; $\log_{10}(1+\mathrm{CE})$ spacing)")
    fig.suptitle(
        "Does x-position coverage predict transformer performance?\n"
        "Points are model–evaluation-r pairs; lines show binned medians",
        fontsize=14,
    )
    save_figure(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_runs", default="runs_hist_window")
    parser.add_argument("--boundary_runs", default="runs_hist_boundary")
    parser.add_argument("--output_dir", default="figures_hist")
    args = parser.parse_args()

    output = Path(args.output_dir)
    window_runs = load_runs(args.window_runs)
    rmax_runs = load_runs(args.boundary_runs, "rmax_")
    rmin_runs = load_runs(args.boundary_runs, "rmin_")

    plot_window_overview(window_runs, output / "window_overview.png")
    plot_boundary_overview(rmax_runs, "rmax", output / "rmax_overview.png")
    plot_boundary_overview(rmin_runs, "rmin", output / "rmin_overview.png")
    plot_correlations(
        window_runs, output / "window_overlap_performance.png",
        window_facets=True,
    )
    plot_correlations(rmax_runs, output / "rmax_overlap_performance.png")
    plot_correlations(rmin_runs, output / "rmin_overlap_performance.png")
    print(f"saved six PNG/PDF figure pairs under {output}")


if __name__ == "__main__":
    main()
