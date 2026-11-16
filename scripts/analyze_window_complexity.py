"""Relate sliding-window training complexity to out-of-support transfer.

The analysis uses only saved ``eval_per_r.npz`` files.  For each evaluation
parameter r, a same-width in-support reference is constructed from models whose
training interval contains r.  The transfer penalty for a model trained on W is

    log10(1 + CE(W, r)) - median_{W': |W'|=|W|, r in W'} log10(1 + CE(W', r)).

This removes much of the intrinsic variation in target difficulty before losses
are averaged across targets outside W.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOL = 1e-9


def load_runs(root):
    grouped = defaultdict(list)
    for path in sorted(Path(root).glob("win_w*_s*_seed*/eval_per_r.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        with np.load(path) as result:
            run = {
                "name": path.parent.name,
                "width": float(params["width"]),
                "start": float(params["start"]),
                "end": float(params["start"] + params["width"]),
                "r": result["r_grid"].astype(float),
                "ce": result["ce_per_r"].astype(float),
            }
        grouped[run["width"]].append(run)

    if not grouped:
        raise FileNotFoundError(f"no window evaluations found under {root}")

    reference_r = next(iter(grouped.values()))[0]["r"]
    for width, runs in grouped.items():
        for run in runs:
            if not np.allclose(run["r"], reference_r):
                raise ValueError(f"evaluation grid differs for {run['name']}")
        grouped[width] = sorted(runs, key=lambda run: run["start"])
    return grouped, reference_r


def lyapunov_grid(r_values, n_steps=20_000, burn_in=1_000):
    """Vectorized version of src.maps.compute_lyapunov on a parameter grid."""
    r_values = np.asarray(r_values, dtype=float)
    x0 = np.random.default_rng(seed=42).uniform(0.1, 0.9)
    x = np.full_like(r_values, x0)
    for _ in range(burn_in):
        x = r_values * x * (1.0 - x)

    total = np.zeros_like(r_values)
    count = np.zeros_like(r_values, dtype=np.int64)
    for _ in range(n_steps):
        derivative = np.abs(r_values * (1.0 - 2.0 * x))
        valid = derivative > 1e-10
        total[valid] += np.log(derivative[valid])
        count[valid] += 1
        x = r_values * x * (1.0 - x)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def transformed_loss(ce):
    return np.log10(1.0 + np.maximum(np.asarray(ce), 0.0))


def same_width_reference(r_grid, runs):
    values = np.stack([transformed_loss(run["ce"]) for run in runs])
    support = np.stack([
        (r_grid >= run["start"] - TOL) & (r_grid <= run["end"] + TOL)
        for run in runs
    ])
    masked = np.where(support, values, np.nan)
    with np.errstate(all="ignore"):
        reference = np.nanmedian(masked, axis=0)
    if np.any(~np.isfinite(reference)):
        missing = r_grid[~np.isfinite(reference)]
        raise ValueError(
            f"training windows do not cover {len(missing)} evaluation points"
        )
    return reference


def summarize_run(run, reference, eval_lambda, dense_r, dense_lambda,
                  distance_cutoff=0.5):
    r = run["r"]
    loss = transformed_loss(run["ce"])
    delta = loss - reference
    distance = np.maximum.reduce((
        run["start"] - r,
        r - run["end"],
        np.zeros_like(r),
    ))
    outside = distance > TOL
    regular = eval_lambda <= 0.0
    chaotic = eval_lambda > 0.0
    lower = r < run["start"] - TOL
    higher = r > run["end"] + TOL
    near = outside & (distance <= distance_cutoff + TOL)

    inside_dense = ((dense_r >= run["start"] - TOL)
                    & (dense_r <= run["end"] + TOL))
    positive = np.maximum(dense_lambda[inside_dense], 0.0)
    window_lambda = dense_lambda[inside_dense]

    def mean(mask, values=delta):
        selected = np.asarray(values)[mask & np.isfinite(values)]
        return float(np.mean(selected)) if selected.size else float("nan")

    return {
        "run": run["name"],
        "width": run["width"],
        "start": run["start"],
        "end": run["end"],
        "center": 0.5 * (run["start"] + run["end"]),
        "complexity_mean_lambda": float(
            np.mean(window_lambda)
        ),
        "complexity_lambda": float(np.mean(positive)),
        "complexity_max_lambda": float(np.max(window_lambda)),
        "complexity_q90_lambda": float(np.quantile(window_lambda, 0.90)),
        "complexity_chaos_fraction": float(
            np.mean(dense_lambda[inside_dense] > 0.0)
        ),
        "mean_in_support_logloss": mean(~outside, loss),
        "mean_ood_logloss_raw": mean(outside, loss),
        "mean_ood_penalty": mean(outside),
        "mean_near_ood_penalty": mean(near),
        "mean_regular_ood_penalty": mean(outside & regular),
        "mean_chaotic_ood_penalty": mean(outside & chaotic),
        "mean_lower_ood_penalty": mean(lower),
        "mean_higher_ood_penalty": mean(higher),
        "n_ood": int(np.sum(outside)),
        "n_regular_ood": int(np.sum(outside & regular)),
        "n_chaotic_ood": int(np.sum(outside & chaotic)),
    }


def rankdata(values):
    """Ranks with average ranks for ties; sufficient for Spearman correlation."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def correlation(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), int(len(x))
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1]), int(len(x))


def binned_trend(x, y, n_bins=6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    minimum = np.min(x)
    at_minimum = np.isclose(x, minimum)
    # Chaos-fraction data have a large, scientifically meaningful point mass
    # at zero. Keep it as one group rather than splitting tied zeroes across
    # several quantile bins, then bin the positive windows separately.
    if at_minimum.sum() >= len(x) / 2 and np.any(~at_minimum):
        groups = [np.flatnonzero(at_minimum)]
        positive_order = np.flatnonzero(~at_minimum)[
            np.argsort(x[~at_minimum])
        ]
        groups.extend(
            group for group in np.array_split(
                positive_order, min(n_bins - 1, len(positive_order))
            ) if len(group)
        )
    else:
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) == 1:
            groups = [np.arange(len(x))]
        else:
            groups = []
            for i in range(len(edges) - 1):
                selected = ((x >= edges[i]) & (x < edges[i + 1])
                            if i < len(edges) - 2
                            else (x >= edges[i]) & (x <= edges[i + 1]))
                if selected.any():
                    groups.append(np.flatnonzero(selected))
    return (
        np.array([np.median(x[group]) for group in groups]),
        np.array([np.median(y[group]) for group in groups]),
        np.array([np.percentile(y[group], 25) for group in groups]),
        np.array([np.percentile(y[group], 75) for group in groups]),
    )


def plot_primary(rows, output):
    x_key = "complexity_chaos_fraction"
    y_key = "mean_ood_penalty"
    widths = sorted({row["width"] for row in rows}, reverse=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(widths)))
    color_for = dict(zip(widths, colors))

    fig, ax = plt.subplots(figsize=(6.8, 4.9), layout="constrained")
    x_all, y_all = [], []
    for width in widths:
        selected = [row for row in rows if row["width"] == width]
        x = np.array([row[x_key] for row in selected])
        y = np.array([row[y_key] for row in selected])
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[valid], y[valid], s=45, alpha=0.8,
            color=color_for[width], edgecolor="white", linewidth=0.5,
            label=f"width {width:g}",
        )
        x_all.extend(x[valid])
        y_all.extend(y[valid])

    x_all, y_all = np.asarray(x_all), np.asarray(y_all)
    groups = [np.isclose(x_all, 0.0), x_all > 0.0]
    bx = np.array([np.median(x_all[group]) for group in groups])
    by = np.array([np.median(y_all[group]) for group in groups])
    blo = np.array([np.percentile(y_all[group], 25) for group in groups])
    bhi = np.array([np.percentile(y_all[group], 75) for group in groups])
    ax.plot(bx, by, color="black", marker="o", lw=2.2, markersize=5,
            label="group median")
    ax.errorbar(bx, by, yerr=[by - blo, bhi - by], fmt="none",
                ecolor="black", capsize=4, lw=1.4)
    rho, _ = correlation(x_all, y_all, method="spearman")
    ax.text(0.97, 0.95, rf"Spearman $\rho={rho:.2f}$",
            transform=ax.transAxes, ha="right", va="top")
    ax.set_title("Out-of-support transfer and chaotic coverage of training")
    ax.set_xlabel(r"fraction of training window with $\lambda(r)>0$")
    ax.set_ylabel(r"mean OOD penalty in $\log_{10}(1+\mathrm{CE})$")
    ax.grid(alpha=0.22)
    ax.legend(title="Training interval", frameon=False, ncols=2,
              loc="upper right", bbox_to_anchor=(0.99, 0.87))
    ax.text(
        0.02, 0.02,
        "Penalty is relative to an in-support model\nat the same target parameter.",
        transform=ax.transAxes, va="bottom", fontsize=8.5, color="0.3",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_main_combination(rows, output):
    """Main-text combination: all, regular, and chaotic OOD targets."""
    panels = [
        ("mean_ood_penalty", "All OOD targets"),
        ("mean_regular_ood_penalty", r"Regular OOD targets ($\lambda\leq0$)"),
        ("mean_chaotic_ood_penalty", r"Chaotic OOD targets ($\lambda>0$)"),
    ]
    x_key = "complexity_chaos_fraction"
    widths = sorted({row["width"] for row in rows}, reverse=True)
    colors = ["#6A3D9A", "#1F78B4", "#1B9E77", "#E6AB02"]
    color_for = dict(zip(widths, colors))

    fig, axes = plt.subplots(
        1, 3, figsize=(12.8, 4.0), sharex=True, sharey=True,
        layout="constrained",
    )
    for panel_index, (ax, (y_key, title)) in enumerate(zip(axes, panels)):
        x_all, y_all = [], []
        for width in widths:
            selected = [row for row in rows if row["width"] == width]
            x = np.array([row[x_key] for row in selected])
            y = np.array([row[y_key] for row in selected])
            valid = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[valid], y[valid], s=38, alpha=0.80,
                color=color_for[width], edgecolor="white", linewidth=0.45,
            )
            x_all.extend(x[valid])
            y_all.extend(y[valid])

        x_all = np.asarray(x_all)
        y_all = np.asarray(y_all)
        groups = [np.isclose(x_all, 0.0), x_all > 0.0]
        groups = [group for group in groups if np.any(group)]
        bx = np.array([np.median(x_all[group]) for group in groups])
        by = np.array([np.median(y_all[group]) for group in groups])
        blo = np.array([np.percentile(y_all[group], 25) for group in groups])
        bhi = np.array([np.percentile(y_all[group], 75) for group in groups])
        ax.plot(bx, by, color="black", marker="o", lw=2.0, markersize=4.5)
        ax.errorbar(
            bx, by, yerr=[by - blo, bhi - by], fmt="none",
            ecolor="black", capsize=3.5, lw=1.2,
        )
        rho, _ = correlation(x_all, y_all, method="spearman")
        ax.text(
            0.97, 0.95, rf"Spearman $\rho={rho:.2f}$",
            transform=ax.transAxes, ha="right", va="top",
        )
        ax.set_title(f"({chr(ord('a') + panel_index)}) {title}")
        ax.set_xlabel(r"fraction of training window with $\lambda(r)>0$")
        ax.grid(alpha=0.22)

    axes[0].set_ylabel(r"mean OOD penalty in $\log_{10}(1+\mathrm{CE})$")
    width_handles = [
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=5.5,
            color=color_for[width], label=f"{width:g}",
        )
        for width in widths
    ]
    width_handles.append(
        plt.Line2D(
            [], [], color="black", marker="o", markersize=4.5,
            label="group median",
        )
    )
    axes[2].legend(
        handles=width_handles, title="window width", loc="lower left",
        ncols=2, fontsize=8, title_fontsize=8, frameon=True,
        framealpha=0.9, borderpad=0.35, handletextpad=0.35,
        columnspacing=0.75,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_summaries(rows, output):
    metrics = [
        ("complexity_mean_lambda", "mean_ood_penalty",
         "All out-of-support targets",
         r"mean $\lambda(r)$ in training window"),
        ("complexity_chaos_fraction", "mean_ood_penalty",
         "All out-of-support targets",
         "fraction of training window with $\\lambda(r)>0$"),
        ("complexity_chaos_fraction", "mean_regular_ood_penalty",
         r"Regular targets ($\lambda\leq0$)",
         "fraction of training window with $\\lambda(r)>0$"),
        ("complexity_chaos_fraction", "mean_chaotic_ood_penalty",
         r"Chaotic targets ($\lambda>0$)",
         "fraction of training window with $\\lambda(r)>0$"),
    ]
    widths = sorted({row["width"] for row in rows}, reverse=True)
    colors = ["#6A3D9A", "#1F78B4", "#1B9E77", "#E6AB02"]
    color_for = dict(zip(widths, colors))

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), layout="constrained")
    axes = axes.ravel()
    for ax, (x_key, y_key, title, x_label) in zip(axes, metrics):
        x_all, y_all = [], []
        for width in widths:
            selected = [row for row in rows if row["width"] == width]
            x = np.array([row[x_key] for row in selected])
            y = np.array([row[y_key] for row in selected])
            valid = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[valid], y[valid], s=34, alpha=0.78,
                color=color_for[width], edgecolor="white", linewidth=0.45,
                label=f"width {width:g}",
            )
            x_all.extend(x[valid])
            y_all.extend(y[valid])

        x_all, y_all = np.asarray(x_all), np.asarray(y_all)
        bx, by, blo, bhi = binned_trend(x_all, y_all)
        ax.plot(bx, by, color="black", marker="o", lw=2.0,
                markersize=4.5, label="binned median")
        ax.fill_between(bx, blo, bhi, color="black", alpha=0.10)
        rho, _ = correlation(x_all, y_all, method="spearman")
        ax.text(0.97, 0.95, rf"Spearman $\rho={rho:.2f}$",
                transform=ax.transAxes, ha="right", va="top")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"mean OOD penalty in $\log_{10}(1+\mathrm{CE})$")
        ax.grid(alpha=0.22)

    width_handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                   color=color_for[width], label=f"{width:g}")
        for width in widths
    ]
    axes[0].legend(
        handles=width_handles, title="window width", loc="lower left",
        ncols=2, fontsize=8, title_fontsize=8, frameon=True,
        framealpha=0.88, borderpad=0.35, handletextpad=0.35,
        columnspacing=0.75,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, path):
    path = Path(path)
    pairs = [
        ("complexity_mean_lambda", "mean_ood_penalty"),
        ("complexity_chaos_fraction", "mean_ood_penalty"),
        ("complexity_lambda", "mean_ood_penalty"),
        ("complexity_max_lambda", "mean_ood_penalty"),
        ("complexity_q90_lambda", "mean_ood_penalty"),
        ("complexity_chaos_fraction", "mean_regular_ood_penalty"),
        ("complexity_chaos_fraction", "mean_chaotic_ood_penalty"),
        ("complexity_chaos_fraction", "mean_near_ood_penalty"),
        ("center", "mean_ood_penalty"),
    ]
    lines = [
        "Window-complexity transfer analysis",
        "===================================",
        f"Windows: {len(rows)}",
        "Windows containing any positive-Lyapunov parameters: "
        f"{sum(row['complexity_chaos_fraction'] > 0 for row in rows)}",
        "Response: same-width, same-target normalized log10(1 + CE) penalty",
        "",
        "Overall correlations:",
    ]
    for x_key, y_key in pairs:
        x = [row[x_key] for row in rows]
        y = [row[y_key] for row in rows]
        pearson, n = correlation(x, y, "pearson")
        spearman, _ = correlation(x, y, "spearman")
        lines.append(
            f"  {x_key} vs {y_key}: n={n}, "
            f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
        )

    lines.extend(["", "Within-width correlations (chaos fraction vs OOD penalty):"])
    for width in sorted({row["width"] for row in rows}, reverse=True):
        selected = [row for row in rows if row["width"] == width]
        pearson, n = correlation(
            [row["complexity_chaos_fraction"] for row in selected],
            [row["mean_ood_penalty"] for row in selected],
            "pearson",
        )
        spearman, _ = correlation(
            [row["complexity_chaos_fraction"] for row in selected],
            [row["mean_ood_penalty"] for row in selected],
            "spearman",
        )
        lines.append(
            f"  width={width:g}: n={n}, Pearson={pearson:.3f}, "
            f"Spearman={spearman:.3f}"
        )

    # Descriptive partial correlation: remove a quadratic trend in window
    # center and categorical width effects from both variables.
    valid = [row for row in rows
             if np.isfinite(row["mean_ood_penalty"])]
    center = np.array([row["center"] for row in valid])
    complexity = np.array([
        row["complexity_chaos_fraction"] for row in valid
    ])
    response = np.array([row["mean_ood_penalty"] for row in valid])
    widths = sorted({row["width"] for row in valid})
    columns = [np.ones(len(valid)), center, center ** 2]
    columns.extend(
        np.array([row["width"] == width for row in valid], dtype=float)
        for width in widths[1:]
    )
    design = np.column_stack(columns)
    residual_complexity = complexity - design @ np.linalg.lstsq(
        design, complexity, rcond=None
    )[0]
    residual_response = response - design @ np.linalg.lstsq(
        design, response, rcond=None
    )[0]
    partial = float(np.corrcoef(residual_complexity, residual_response)[0, 1])
    lines.extend([
        "",
        "Descriptive confound check:",
        "  Partial Pearson correlation after removing quadratic window-center "
        f"and width effects: {partial:.3f}",
        "  (Windows overlap and are not independent replicates; this is not an "
        "inferential p-value.)",
    ])

    lines.extend(["", "Group summaries (median [IQR]):"])
    for metric in (
        "mean_ood_penalty",
        "mean_near_ood_penalty",
        "mean_regular_ood_penalty",
        "mean_chaotic_ood_penalty",
    ):
        lines.append(f"  {metric}:")
        for label, predicate in (
            ("no chaotic coverage", lambda row:
             row["complexity_chaos_fraction"] == 0.0),
            ("some chaotic coverage", lambda row:
             row["complexity_chaos_fraction"] > 0.0),
        ):
            values = np.array([
                row[metric] for row in rows
                if predicate(row) and np.isfinite(row[metric])
            ])
            quartiles = np.percentile(values, [25, 50, 75])
            lines.append(
                f"    {label}: n={len(values)}, {quartiles[1]:.3f} "
                f"[{quartiles[0]:.3f}, {quartiles[2]:.3f}]"
            )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--output", default="figures/window_complexity_transfer.png"
    )
    parser.add_argument(
        "--primary-output", default="figures/window_complexity_curve.png"
    )
    parser.add_argument(
        "--main-output", default="figures/window_complexity_main.png"
    )
    parser.add_argument(
        "--csv", default="figures/window_complexity_transfer.csv"
    )
    parser.add_argument(
        "--summary", default="figures/window_complexity_transfer_summary.txt"
    )
    parser.add_argument("--lyapunov-steps", type=int, default=20_000)
    parser.add_argument("--complexity-grid-size", type=int, default=3501)
    parser.add_argument("--distance-cutoff", type=float, default=0.5)
    args = parser.parse_args()

    grouped, eval_r = load_runs(args.runs_dir)
    dense_r = np.linspace(float(eval_r.min()), float(eval_r.max()),
                          args.complexity_grid_size)
    combined_r = np.concatenate([eval_r, dense_r])
    combined_lambda = lyapunov_grid(combined_r, n_steps=args.lyapunov_steps)
    eval_lambda = combined_lambda[:len(eval_r)]
    dense_lambda = combined_lambda[len(eval_r):]

    rows = []
    for width in sorted(grouped, reverse=True):
        runs = grouped[width]
        reference = same_width_reference(eval_r, runs)
        rows.extend([
            summarize_run(
                run, reference, eval_lambda, dense_r, dense_lambda,
                distance_cutoff=args.distance_cutoff,
            )
            for run in runs
        ])

    plot_summaries(rows, args.output)
    plot_primary(rows, args.primary_output)
    plot_main_combination(rows, args.main_output)
    write_csv(rows, args.csv)
    write_summary(rows, args.summary)
    print(f"analyzed {len(rows)} windows")
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")
    print(
        f"saved {args.primary_output} and "
        f"{Path(args.primary_output).with_suffix('.pdf')}"
    )
    print(
        f"saved {args.main_output} and "
        f"{Path(args.main_output).with_suffix('.pdf')}"
    )
    print(f"saved {args.csv}")
    print(f"saved {args.summary}")


if __name__ == "__main__":
    main()
