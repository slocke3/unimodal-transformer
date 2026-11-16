"""Relate token-level transition similarity to sliding-window transfer.

For each logistic-map parameter on the evaluation grid, this script estimates
the joint distribution of the *supervised* one-step transition,

    J_r(i, j) = P(t_{n}=i, t_{n+1}=j),

under the same finite-trajectory protocol used by the window experiments.  In
particular, the default terminal-token objective first predicts after a
50-token context, so transitions before that point are not included.

For every training window W and held-out target r, it compares the model's
same-target-normalized loss penalty with four representation-level shifts:

  * nearest_transition_js: min_{r' in W} JS(J_r, J_{r'}),
  * nearest transition-support distance under Manhattan and Euclidean norms,
  * pooled_transition_js:  JS(J_r, J_W),
  * unseen_transition_mass: target mass on transitions absent from training,
  * occupancy_js: JS between the current-token marginals of J_r and J_W.

The pooled training distribution and its support are reconstructed from the
actual r-values, split, initial-condition seed, trajectory length, and context
length saved for each run.  The nearest-map comparison uses standardized
process estimates on the common evaluation grid so that sampling noise and the
number of trajectories assigned to an individual training r do not determine
the retrieval distance.
"""

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOL = 1e-9


def transformed_loss(ce):
    return np.log10(1.0 + np.maximum(np.asarray(ce, dtype=float), 0.0))


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
                "train_r": result["train_r"].astype(float),
                "params": params,
            }
        grouped[run["width"]].append(run)

    if not grouped:
        raise FileNotFoundError(f"no window evaluations found under {root}")

    reference_r = next(iter(grouped.values()))[0]["r"]
    for width, runs in grouped.items():
        for run in runs:
            if not np.allclose(run["r"], reference_r):
                raise ValueError(f"evaluation grid differs for {run['name']}")
        grouped[width] = sorted(runs, key=lambda item: item["start"])
    return grouped, reference_r


def same_width_reference(r_grid, runs):
    values = np.stack([transformed_loss(run["ce"]) for run in runs])
    support = np.stack([
        (r_grid >= run["start"] - TOL) & (r_grid <= run["end"] + TOL)
        for run in runs
    ])
    with np.errstate(all="ignore"):
        reference = np.nanmedian(np.where(support, values, np.nan), axis=0)
    if np.any(~np.isfinite(reference)):
        raise ValueError("training windows do not cover the evaluation grid")
    return reference


def normalized(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts)


def entropy_rows(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=-1)


def js_to_one(distributions, target):
    """Jensen--Shannon divergence from every row to one target (nats)."""
    distributions = np.asarray(distributions, dtype=float)
    target = np.asarray(target, dtype=float)
    midpoint = 0.5 * (distributions + target[None, :])
    return (
        entropy_rows(midpoint)
        - 0.5 * entropy_rows(distributions)
        - 0.5 * entropy_rows(target[None, :])[0]
    )


def pairwise_js(distributions):
    distributions = np.asarray(distributions, dtype=float)
    result = np.empty((len(distributions), len(distributions)), dtype=float)
    for i, target in enumerate(distributions):
        result[i] = js_to_one(distributions, target)
    # Remove tiny negative values caused by roundoff and enforce symmetry.
    result = np.maximum(0.0, 0.5 * (result + result.T))
    np.fill_diagonal(result, 0.0)
    return result


def supervised_transition_counts(r_values, x0s, context_len, traj_len,
                                 n_bins):
    """Counts for the terminal targets used by the original objective.

    DiscreteMapDataset creates a trajectory with ``traj_len + 1`` states and
    sliding windows t=0,...,traj_len-context_len-1.  The supervised transition
    is therefore from state t+context_len-1 to t+context_len.
    """
    r_values = np.asarray(r_values, dtype=float)
    x = np.asarray(x0s, dtype=float).copy()
    if x.shape != r_values.shape:
        raise ValueError("r_values and x0s must have the same shape")
    counts = np.zeros(n_bins * n_bins, dtype=np.int64)
    first_step = context_len - 1
    final_step = traj_len - 2
    for step in range(traj_len):
        x_next = r_values * x * (1.0 - x)
        if first_step <= step <= final_step:
            current = np.clip((x * n_bins).astype(np.int64), 0, n_bins - 1)
            following = np.clip(
                (x_next * n_bins).astype(np.int64), 0, n_bins - 1
            )
            counts += np.bincount(
                current * n_bins + following, minlength=n_bins * n_bins
            )
        x = x_next
    return counts.reshape(n_bins, n_bins)


def context_transition_counts(r_values, x0s, context_len, traj_len, n_bins):
    """Counts of adjacent pairs as exposed across all sliding examples.

    Unlike ``supervised_transition_counts``, this counts every adjacent pair in
    every context--target example.  A transition near the center of a trajectory
    appears in more overlapping examples and is therefore weighted more heavily,
    exactly as it is in the dataset presented to the transformer.
    """
    r_values = np.asarray(r_values, dtype=float)
    x = np.asarray(x0s, dtype=float).copy()
    if x.shape != r_values.shape:
        raise ValueError("r_values and x0s must have the same shape")
    counts = np.zeros(n_bins * n_bins, dtype=np.int64)
    n_windows = traj_len - context_len
    for step in range(traj_len - 1):
        x_next = r_values * x * (1.0 - x)
        first_window = max(0, step - (context_len - 1))
        final_window = min(step, n_windows - 1)
        weight = max(0, final_window - first_window + 1)
        if weight:
            current = np.clip((x * n_bins).astype(np.int64), 0, n_bins - 1)
            following = np.clip(
                (x_next * n_bins).astype(np.int64), 0, n_bins - 1
            )
            counts += weight * np.bincount(
                current * n_bins + following, minlength=n_bins * n_bins
            )
        x = x_next
    return counts.reshape(n_bins, n_bins)


def standardized_processes(r_grid, n_trajectories, context_len, traj_len,
                           n_bins, seed):
    """Estimate terminal and context J_r with common initial conditions."""
    rng = np.random.default_rng(seed)
    x0s = rng.uniform(0.05, 0.95, size=n_trajectories)
    terminal_counts = np.empty(
        (len(r_grid), n_bins, n_bins), dtype=np.int64
    )
    context_counts = np.empty_like(terminal_counts)
    for index, r in enumerate(r_grid):
        r_values = np.full(n_trajectories, r)
        terminal_counts[index] = supervised_transition_counts(
            r_values, x0s, context_len, traj_len, n_bins
        )
        context_counts[index] = context_transition_counts(
            r_values, x0s, context_len, traj_len, n_bins
        )
    flat = terminal_counts.reshape(len(r_grid), -1).astype(float)
    flat /= flat.sum(axis=1, keepdims=True)
    context_flat = context_counts.reshape(len(r_grid), -1).astype(float)
    context_flat /= context_flat.sum(axis=1, keepdims=True)
    return terminal_counts, flat, context_counts, context_flat


def grid_support_distances(counts, metric="manhattan"):
    """Distance from each token-pair cell to represented support.

    Distances are measured on the two-dimensional token-transition grid and
    converted to state-space units by dividing token-index distances by the
    number of bins.  ``euclidean`` is an exact sensitivity check for the
    default four-neighbour Manhattan distance.
    """
    support = np.asarray(counts) > 0
    n_bins = support.shape[0]
    if metric == "euclidean":
        support_coordinates = np.argwhere(support)
        if not len(support_coordinates):
            raise ValueError("transition support is empty")
        grid_coordinates = np.column_stack(
            np.indices(support.shape).reshape(2, -1)
        )
        squared = np.sum(
            (grid_coordinates[:, None, :]
             - support_coordinates[None, :, :]) ** 2,
            axis=2,
        )
        return np.sqrt(squared.min(axis=1)).reshape(support.shape) / n_bins
    if metric != "manhattan":
        raise ValueError(f"unknown grid metric: {metric}")
    distance = np.full((n_bins, n_bins), 2 * n_bins, dtype=np.int32)
    queue = deque()
    for i, j in np.argwhere(support):
        distance[i, j] = 0
        queue.append((int(i), int(j)))
    while queue:
        i, j = queue.popleft()
        candidate = distance[i, j] + 1
        for ii, jj in ((i - 1, j), (i + 1, j),
                       (i, j - 1), (i, j + 1)):
            if (0 <= ii < n_bins and 0 <= jj < n_bins
                    and candidate < distance[ii, jj]):
                distance[ii, jj] = candidate
                queue.append((ii, jj))
    return distance.astype(float) / n_bins


def directed_support_distance_matrix(process_counts, processes,
                                     metric="manhattan"):
    """Expected token-grid distance from target mass to source support."""
    fields = np.stack([
        grid_support_distances(counts, metric=metric).reshape(-1)
        for counts in process_counts
    ])
    return processes @ fields.T


def training_r_and_x0(run):
    """Recreate the exact training split and initial conditions for one run."""
    params = run["params"]
    train_r = run["train_r"]
    total = int(params["n_train_traj"])
    seed = int(params["seed"])
    n_per, remainder = divmod(total, len(train_r))

    rng = np.random.default_rng(seed)
    repeats = np.full(len(train_r), n_per, dtype=int)
    if remainder:
        repeats[rng.permutation(len(train_r))[:remainder]] += 1
    all_r = np.repeat(train_r, repeats)
    rng.shuffle(all_r)

    n_val = int(float(params.get("val_frac", 0.15)) * len(all_r))
    max_val = params.get("max_val_traj")
    if max_val not in (None, 0):
        n_val = min(n_val, int(max_val))
    training_r = all_r[n_val:]

    # DiscreteMapDataset initializes a fresh generator with this same seed and,
    # when r_values are supplied, draws x0 immediately.
    dataset_rng = np.random.default_rng(seed)
    x0s = dataset_rng.uniform(0.05, 0.95, size=len(training_r))
    return training_r, x0s


def reconstruct_training_counts(run, n_bins):
    """Recreate terminal and context transition counts for one run."""
    params = run["params"]
    training_r, x0s = training_r_and_x0(run)
    terminal = supervised_transition_counts(
        training_r,
        x0s,
        int(params["context_len"]),
        int(params["traj_len"]),
        n_bins,
    )
    context = context_transition_counts(
        training_r,
        x0s,
        int(params["context_len"]),
        int(params["traj_len"]),
        n_bins,
    )
    return terminal, context


def lyapunov_grid(r_values, n_steps=20_000, burn_in=1_000):
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


def rankdata(values):
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


def correlation(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), int(len(x))
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1]), int(len(x))


def make_pair_rows(grouped, r_grid, process_counts, processes, process_js,
                   context_process_counts, context_processes,
                   support_distance, support_distance_euclidean,
                   eval_lambda, n_bins, reference_grouped=None):
    rows = []
    terminal_occupancy = process_counts.sum(axis=2).astype(float)
    terminal_occupancy /= terminal_occupancy.sum(axis=1, keepdims=True)
    context_occupancy = context_process_counts.sum(axis=2).astype(float)
    context_occupancy /= context_occupancy.sum(axis=1, keepdims=True)
    for width in sorted(grouped, reverse=True):
        runs = grouped[width]
        # A targeted replication does not tile the full family. It can use a
        # fixed, separately supplied reference sweep, so adding seeds/windows
        # does not change the normalization of previously evaluated models.
        reference_runs = runs if reference_grouped is None else reference_grouped[width]
        reference = same_width_reference(r_grid, reference_runs)
        for run in runs:
            train_counts, train_context_counts = reconstruct_training_counts(
                run, n_bins
            )
            train_process = normalized(train_counts.reshape(-1))
            train_occupancy = normalized(train_counts.sum(axis=1))
            train_context_process = normalized(
                train_context_counts.reshape(-1)
            )
            train_context_occupancy = normalized(
                train_context_counts.sum(axis=1)
            )
            source = ((r_grid >= run["start"] - TOL)
                      & (r_grid <= run["end"] + TOL))
            source_indices = np.flatnonzero(source)
            if not len(source_indices):
                raise ValueError(f"no source-grid parameters in {run['name']}")

            pooled_js = js_to_one(processes, train_process)
            context_pooled_js = js_to_one(
                context_processes, train_context_process
            )
            occupancy_js = js_to_one(terminal_occupancy, train_occupancy)
            context_occupancy_js = js_to_one(
                context_occupancy, train_context_occupancy
            )
            context_occupancy_wasserstein = (
                np.abs(
                    np.cumsum(context_occupancy, axis=1)
                    - np.cumsum(train_context_occupancy)[None, :]
                ).sum(axis=1)
                / n_bins
            )
            unseen = processes[:, train_counts.reshape(-1) == 0].sum(axis=1)
            context_unseen = context_processes[
                :, train_context_counts.reshape(-1) == 0
            ].sum(axis=1)

            nearest_local = np.argmin(process_js[:, source_indices], axis=1)
            nearest_indices = source_indices[nearest_local]
            nearest_js = process_js[np.arange(len(r_grid)), nearest_indices]
            nearest_support_local = np.argmin(
                support_distance[:, source_indices], axis=1
            )
            nearest_support_indices = source_indices[nearest_support_local]
            nearest_support_distance = support_distance[
                np.arange(len(r_grid)), nearest_support_indices
            ]
            nearest_euclidean_local = np.argmin(
                support_distance_euclidean[:, source_indices], axis=1
            )
            nearest_euclidean_indices = source_indices[
                nearest_euclidean_local
            ]
            nearest_euclidean_distance = support_distance_euclidean[
                np.arange(len(r_grid)), nearest_euclidean_indices
            ]
            distance = np.maximum.reduce((
                run["start"] - r_grid,
                r_grid - run["end"],
                np.zeros_like(r_grid),
            ))
            delta = transformed_loss(run["ce"]) - reference

            for i, target_r in enumerate(r_grid):
                if distance[i] <= TOL:
                    continue
                rows.append({
                    "run": run["name"],
                    "width": run["width"],
                    "start": run["start"],
                    "end": run["end"],
                    "target_r": float(target_r),
                    "target_lambda": float(eval_lambda[i]),
                    "parameter_distance": float(distance[i]),
                    "loss_penalty": float(delta[i]),
                    "nearest_transition_js": float(nearest_js[i]),
                    "nearest_transition_r": float(r_grid[nearest_indices[i]]),
                    "nearest_transition_support_distance": float(
                        nearest_support_distance[i]
                    ),
                    "nearest_support_r": float(
                        r_grid[nearest_support_indices[i]]
                    ),
                    "nearest_transition_support_distance_euclidean": float(
                        nearest_euclidean_distance[i]
                    ),
                    "nearest_support_r_euclidean": float(
                        r_grid[nearest_euclidean_indices[i]]
                    ),
                    "pooled_transition_js": float(pooled_js[i]),
                    "unseen_transition_mass": float(unseen[i]),
                    "occupancy_js": float(occupancy_js[i]),
                    "context_transition_js": float(context_pooled_js[i]),
                    "context_unseen_transition_mass": float(
                        context_unseen[i]
                    ),
                    "context_occupancy_js": float(context_occupancy_js[i]),
                    "context_occupancy_wasserstein": float(
                        context_occupancy_wasserstein[i]
                    ),
                })
    return rows


def quantile_trend(x, y, n_bins=12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    order = np.argsort(x, kind="mergesort")
    at_minimum = np.isclose(x, x[order[0]])
    at_maximum = np.isclose(x, x[order[-1]])
    if at_minimum.sum() >= len(x) / 2 and np.any(~at_minimum):
        remainder = np.flatnonzero(~at_minimum)
        remainder = remainder[np.argsort(x[remainder], kind="mergesort")]
        groups = [np.flatnonzero(at_minimum)]
        groups.extend(group for group in np.array_split(
            remainder, min(n_bins - 1, len(remainder))
        ) if len(group))
    elif at_maximum.sum() >= len(x) / 2 and np.any(~at_maximum):
        remainder = np.flatnonzero(~at_maximum)
        remainder = remainder[np.argsort(x[remainder], kind="mergesort")]
        groups = [group for group in np.array_split(
            remainder, min(n_bins - 1, len(remainder))
        ) if len(group)]
        groups.append(np.flatnonzero(at_maximum))
    else:
        groups = [group for group in np.array_split(
            order, min(n_bins, len(order))
        ) if len(group)]
    return (
        np.array([np.median(x[group]) for group in groups]),
        np.array([np.median(y[group]) for group in groups]),
        np.array([np.percentile(y[group], 25) for group in groups]),
        np.array([np.percentile(y[group], 75) for group in groups]),
    )


def plot_metric_comparison(rows, path):
    metrics = [
        ("nearest_transition_support_distance",
         "Transition-support distance",
         "Nearest transition-support distance"),
        ("context_occupancy_js", "Context-state mismatch",
         "Context-token occupancy JS"),
        ("parameter_distance", "Parameter-space baseline",
         r"Distance from training interval in $r$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    y = np.array([row["loss_penalty"] for row in rows])
    width_values = sorted({row["width"] for row in rows}, reverse=True)
    colors = ["#6A3D9A", "#1F78B4", "#1B9E77", "#E6AB02"]
    color_for = dict(zip(width_values, colors))

    for ax, (key, title, label) in zip(axes, metrics):
        x = np.array([row[key] for row in rows])
        # Plot a deterministic subsample per width to keep the PDF lightweight.
        for width in width_values:
            indices = np.flatnonzero(
                np.array([row["width"] == width for row in rows])
            )
            step = max(1, int(np.ceil(len(indices) / 700)))
            indices = indices[::step]
            ax.scatter(x[indices], y[indices], s=10, alpha=0.24,
                       color=color_for[width], edgecolors="none",
                       label=f"width {width:g}")
        bx, by, low, high = quantile_trend(x, y)
        ax.plot(bx, by, color="black", marker="o", ms=3.5, lw=1.7,
                label="quantile median")
        ax.fill_between(bx, low, high, color="black", alpha=0.10)
        rho, n = correlation(x, y)
        ax.text(0.97, 0.95, rf"Spearman $\rho={rho:.2f}$",
                transform=ax.transAxes, ha="right", va="top")
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_ylabel(r"OOD penalty in $\log_{10}(1+\mathrm{CE})$")
        ax.grid(alpha=0.2)

    width_handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                   color=color_for[width], label=f"{width:g}")
        for width in width_values
    ]
    axes[0].legend(
        handles=width_handles, title="window width", loc="upper left",
        ncols=2, fontsize=8, title_fontsize=8, frameon=True,
        framealpha=0.9, borderpad=0.35, handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.suptitle(
        "Out-of-support error versus represented transition similarity",
        fontsize=14,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_process_matrix(r_grid, process_js, support_distance, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), layout="constrained")
    panels = [
        (process_js, "Exact token-transition JS", "JS divergence (nats)"),
        (support_distance, "Directed transition-support distance",
         "Mean Manhattan distance (state-space units)"),
    ]
    for ax, (values, title, color_label) in zip(axes, panels):
        image = ax.imshow(
            values,
            origin="lower",
            extent=[r_grid[0], r_grid[-1], r_grid[0], r_grid[-1]],
            aspect="auto",
            cmap="magma",
        )
        ax.plot(r_grid, r_grid, color="white", lw=0.8, alpha=0.8)
        ax.set_xlabel(r"Comparison parameter $r'$ ")
        ax.set_ylabel(r"Target parameter $r$")
        ax.set_title(title)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(color_label)
    fig.suptitle("Pairwise similarity of represented one-step processes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_windows(rows, dense_r, dense_lambda):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    summaries = []
    metric_keys = [
        "loss_penalty",
        "nearest_transition_js",
        "nearest_transition_support_distance",
        "nearest_transition_support_distance_euclidean",
        "pooled_transition_js",
        "unseen_transition_mass",
        "context_occupancy_js",
        "parameter_distance",
    ]
    for name, selected in grouped.items():
        first = selected[0]
        inside = ((dense_r >= first["start"] - TOL)
                  & (dense_r <= first["end"] + TOL))
        summary = {
            "run": name,
            "width": first["width"],
            "start": first["start"],
            "end": first["end"],
            "center": 0.5 * (first["start"] + first["end"]),
            "chaos_fraction": float(np.mean(dense_lambda[inside] > 0.0)),
            "mean_positive_lambda": float(np.mean(
                np.maximum(dense_lambda[inside], 0.0)
            )),
        }
        for key in metric_keys:
            summary[f"mean_{key}"] = float(np.mean([
                row[key] for row in selected
            ]))
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (item["width"], item["start"]))


def plot_window_summary(rows, path):
    panels = [
        ("chaos_fraction", "mean_nearest_transition_support_distance",
         r"Chaotic coverage versus represented support",
         r"fraction of window with $\lambda(r)>0$",
         "mean nearest transition-support distance"),
        ("mean_nearest_transition_support_distance", "mean_loss_penalty",
         "Transition coverage versus transfer",
         "mean nearest transition-support distance",
         r"mean OOD penalty in $\log_{10}(1+\mathrm{CE})$"),
        ("mean_context_occupancy_js", "mean_loss_penalty",
         "Context-state coverage versus transfer",
         "mean context-token occupancy JS",
         r"mean OOD penalty in $\log_{10}(1+\mathrm{CE})$"),
    ]
    widths = sorted({row["width"] for row in rows}, reverse=True)
    colors = ["#6A3D9A", "#1F78B4", "#1B9E77", "#E6AB02"]
    color_for = dict(zip(widths, colors))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), layout="constrained")
    for panel_index, (ax, (x_key, y_key, title, x_label, y_label)) in enumerate(
            zip(axes, panels)):
        all_x, all_y = [], []
        for width in widths:
            selected = [row for row in rows if row["width"] == width]
            x = np.array([row[x_key] for row in selected])
            y = np.array([row[y_key] for row in selected])
            ax.scatter(x, y, s=42, alpha=0.78, color=color_for[width],
                       edgecolor="white", linewidth=0.45,
                       label=f"width {width:g}")
            all_x.extend(x)
            all_y.extend(y)
        rho, _ = correlation(all_x, all_y)
        if panel_index == 0:
            ax.text(0.03, 0.95, rf"Spearman $\rho={rho:.2f}$",
                    transform=ax.transAxes, ha="left", va="top")
        else:
            ax.text(0.97, 0.95, rf"Spearman $\rho={rho:.2f}$",
                    transform=ax.transAxes, ha="right", va="top")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
    width_handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                   color=color_for[width], label=f"{width:g}")
        for width in widths
    ]
    axes[0].legend(
        handles=width_handles, title="window width", loc="upper right",
        ncols=2, fontsize=8, title_fontsize=8, frameon=True,
        framealpha=0.9, borderpad=0.35, handletextpad=0.35,
        columnspacing=0.75,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_main_combination(rows, window_rows, path):
    """Main-text combination of pair- and window-level transfer analyses."""
    fig = plt.figure(figsize=(12.8, 8.2), layout="constrained")
    grid = fig.add_gridspec(4, 6, height_ratios=[0.07, 1, 0.07, 1])
    pair_header = fig.add_subplot(grid[0, :])
    pair_header.axis("off")
    pair_header.text(
        0.5, 0.5,
        f"Window--target-level analysis ({len(rows):,} OOD pairs)",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
    )
    top_axes = [
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:6]),
    ]
    window_header = fig.add_subplot(grid[2, :])
    window_header.axis("off")
    window_header.text(
        0.5, 0.5,
        f"Training-window-level analysis ({len(window_rows)} windows)",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
    )
    bottom_axes = [
        fig.add_subplot(grid[3, 0:3]),
        fig.add_subplot(grid[3, 3:6]),
    ]
    width_values = sorted({row["width"] for row in rows}, reverse=True)
    colors = ["#6A3D9A", "#1F78B4", "#1B9E77", "#E6AB02"]
    color_for = dict(zip(width_values, colors))

    pair_metrics = [
        (
            "nearest_transition_support_distance",
            "Transition-support distance",
            "Nearest transition-support mismatch (lower = closer)",
        ),
        (
            "context_occupancy_js",
            "Context-state mismatch",
            "Context-state JS mismatch (lower = closer)",
        ),
        (
            "parameter_distance",
            "Parameter-space baseline",
            r"Distance outside training interval in $r$",
        ),
    ]
    pair_response = np.array([row["loss_penalty"] for row in rows])
    row_widths = np.array([row["width"] for row in rows])
    for panel_index, (ax, (key, title, x_label)) in enumerate(
            zip(top_axes, pair_metrics)):
        x = np.array([row[key] for row in rows])
        for width in width_values:
            indices = np.flatnonzero(row_widths == width)
            step = max(1, int(np.ceil(len(indices) / 700)))
            indices = indices[::step]
            ax.scatter(
                x[indices], pair_response[indices], s=10, alpha=0.24,
                color=color_for[width], edgecolors="none",
            )
        bx, by, low, high = quantile_trend(x, pair_response)
        ax.plot(bx, by, color="black", marker="o", ms=3.5, lw=1.7)
        ax.fill_between(bx, low, high, color="black", alpha=0.10)
        rho, _ = correlation(x, pair_response)
        ax.text(
            0.97, 0.95, rf"Spearman $\rho={rho:.2f}$",
            transform=ax.transAxes, ha="right", va="top",
        )
        ax.set_title(f"({chr(ord('a') + panel_index)}) {title}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"OOD penalty in $\log_{10}(1+\mathrm{CE})$")
        ax.grid(alpha=0.2)

    window_panels = [
        (
            "chaos_fraction", "mean_nearest_transition_support_distance",
            "Chaotic coverage versus represented support",
            r"fraction of window with $\lambda(r)>0$",
            "mean transition-support mismatch (lower = closer)",
        ),
        (
            "mean_nearest_transition_support_distance", "mean_loss_penalty",
            "Transition coverage versus transfer",
            "mean transition-support mismatch (lower = closer)",
            r"mean OOD penalty in $\log_{10}(1+\mathrm{CE})$",
        ),
    ]
    for offset, (ax, (x_key, y_key, title, x_label, y_label)) in enumerate(
            zip(bottom_axes, window_panels), start=3):
        all_x, all_y = [], []
        for width in width_values:
            selected = [
                row for row in window_rows if row["width"] == width
            ]
            x = np.array([row[x_key] for row in selected])
            y = np.array([row[y_key] for row in selected])
            ax.scatter(
                x, y, s=42, alpha=0.78, color=color_for[width],
                edgecolor="white", linewidth=0.45,
            )
            all_x.extend(x)
            all_y.extend(y)
        rho, _ = correlation(all_x, all_y)
        anchor_x = 0.03 if offset == 3 else 0.97
        alignment = "left" if offset == 3 else "right"
        ax.text(
            anchor_x, 0.95, rf"Spearman $\rho={rho:.2f}$",
            transform=ax.transAxes, ha=alignment, va="top",
        )
        ax.set_title(f"({chr(ord('a') + offset)}) {title}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)

    width_handles = [
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=5.5,
            color=color_for[width], label=f"{width:g}",
        )
        for width in width_values
    ]
    top_axes[0].legend(
        handles=width_handles, title="window width", loc="upper left",
        ncols=2, fontsize=8, title_fontsize=8, frameon=True,
        framealpha=0.9, borderpad=0.35, handletextpad=0.35,
        columnspacing=0.75,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def append_window_summary(lines, rows):
    lines.extend([
        "",
        "Window-level associations (n=53 overlapping windows):",
    ])
    pairs = [
        ("chaos_fraction", "mean_nearest_transition_support_distance"),
        ("chaos_fraction",
         "mean_nearest_transition_support_distance_euclidean"),
        ("chaos_fraction", "mean_context_occupancy_js"),
        ("mean_nearest_transition_support_distance", "mean_loss_penalty"),
        ("mean_nearest_transition_support_distance_euclidean",
         "mean_loss_penalty"),
        ("mean_context_occupancy_js", "mean_loss_penalty"),
    ]
    for x_key, y_key in pairs:
        pearson, _ = correlation(
            [row[x_key] for row in rows], [row[y_key] for row in rows],
            "pearson",
        )
        spearman, _ = correlation(
            [row[x_key] for row in rows], [row[y_key] for row in rows],
        )
        lines.append(
            f"  {x_key} vs {y_key}: Pearson={pearson:.3f}, "
            f"Spearman={spearman:.3f}"
        )
    lines.append("  Group medians [no chaotic coverage, some chaotic coverage]:")
    for key in (
        "mean_loss_penalty",
        "mean_nearest_transition_support_distance",
        "mean_nearest_transition_support_distance_euclidean",
        "mean_context_occupancy_js",
    ):
        values = []
        counts = []
        for has_chaos in (False, True):
            selected = [
                row[key] for row in rows
                if (row["chaos_fraction"] > 0.0) == has_chaos
            ]
            values.append(float(np.median(selected)))
            counts.append(len(selected))
        lines.append(
            f"    {key}: {values[0]:.3f} (n={counts[0]}), "
            f"{values[1]:.3f} (n={counts[1]})"
        )

    center = np.array([row["center"] for row in rows])
    widths = sorted({row["width"] for row in rows})
    design = np.column_stack(
        [np.ones(len(rows)), center, center ** 2]
        + [
            np.array([row["width"] == width for row in rows], dtype=float)
            for width in widths[1:]
        ]
    )

    def residual(key):
        values = np.array([row[key] for row in rows])
        return values - design @ np.linalg.lstsq(
            design, values, rcond=None
        )[0]

    lines.extend([
        "  Partial Pearson after removing quadratic window-center and width",
        "  effects (descriptive confound check):",
    ])
    for x_key, y_key in pairs:
        value = np.corrcoef(residual(x_key), residual(y_key))[0, 1]
        lines.append(f"    {x_key} vs {y_key}: {value:.3f}")


def write_summary(rows, path, settings, window_rows=None):
    metrics = [
        "nearest_transition_js",
        "nearest_transition_support_distance",
        "nearest_transition_support_distance_euclidean",
        "pooled_transition_js",
        "unseen_transition_mass",
        "occupancy_js",
        "context_transition_js",
        "context_unseen_transition_mass",
        "context_occupancy_js",
        "context_occupancy_wasserstein",
        "parameter_distance",
    ]
    response = np.array([row["loss_penalty"] for row in rows])
    lines = [
        "Window transition-similarity analysis",
        "=====================================",
        f"OOD window-target pairs: {len(rows)}",
        f"Process trajectories per r: {settings['n_process_trajectories']}",
        f"Bins: {settings['n_bins']}",
        f"Context length: {settings['context_len']}",
        f"Trajectory length: {settings['traj_len']}",
        "Response: same-width, same-target normalized log10(1 + CE) penalty",
        "Terminal metrics: one-step transitions at the supervised target",
        "Context metrics: all adjacent pairs as weighted by sliding examples",
        "",
        "Overall pair-level correlations:",
    ]
    for key in metrics:
        values = np.array([row[key] for row in rows])
        pearson, n = correlation(values, response, "pearson")
        spearman, _ = correlation(values, response, "spearman")
        lines.append(
            f"  {key}: n={n}, Pearson={pearson:.3f}, Spearman={spearman:.3f}"
        )

    manhattan = np.array([
        row["nearest_transition_support_distance"] for row in rows
    ])
    euclidean = np.array([
        row["nearest_transition_support_distance_euclidean"] for row in rows
    ])
    metric_rho, _ = correlation(manhattan, euclidean)
    retrieval_agreement = np.mean([
        np.isclose(
            row["nearest_support_r"], row["nearest_support_r_euclidean"]
        )
        for row in rows
    ])
    lines.extend([
        "",
        "Transition-support distance metric sensitivity:",
        "  Distances are normalized by the number of bins.",
        f"  Manhattan vs Euclidean Spearman: {metric_rho:.3f}",
        "  Same nearest source-grid parameter: "
        f"{retrieval_agreement:.1%}",
    ])

    nearest = np.array([row["nearest_transition_js"] for row in rows])
    unseen = np.array([row["unseen_transition_mass"] for row in rows])
    context_unseen = np.array([
        row["context_unseen_transition_mass"] for row in rows
    ])
    lines.extend([
        "",
        "Exact-support saturation diagnostics:",
        "  nearest_transition_js at maximum log(2): "
        f"{np.mean(nearest >= np.log(2) - 1e-8):.1%}",
        "  unseen_transition_mass >= 0.999: "
        f"{np.mean(unseen >= 0.999):.1%}",
        "  context_unseen_transition_mass >= 0.999: "
        f"{np.mean(context_unseen >= 0.999):.1%}",
    ])

    # Descriptive partial rank correlations.  Residualizing ranks is the
    # standard linear representation of a partial Spearman correlation.
    ranked_response = rankdata(response)
    ranked_parameter_distance = rankdata([
        row["parameter_distance"] for row in rows
    ])
    widths = sorted({row["width"] for row in rows})
    width_indicators = np.column_stack([
        np.array([row["width"] == width for row in rows], dtype=float)
        for width in widths[1:]
    ])
    controls = np.column_stack([
        ranked_parameter_distance, width_indicators
    ])

    def residual(values, design):
        design = np.column_stack([np.ones(len(values)), design])
        return values - design @ np.linalg.lstsq(
            design, values, rcond=None
        )[0]

    residual_response = residual(ranked_response, controls)
    lines.extend([
        "",
        "Partial Spearman correlations controlling parameter distance and",
        "categorical window width:",
    ])
    for key in metrics:
        if key == "parameter_distance":
            continue
        ranked_predictor = rankdata([row[key] for row in rows])
        value = np.corrcoef(
            residual(ranked_predictor, controls), residual_response
        )[0, 1]
        lines.append(f"  {key}: {value:.3f}")

    model_keys = [
        "parameter_distance",
        "context_occupancy_js",
        "nearest_transition_support_distance",
        "nearest_transition_js",
    ]
    predictors = np.column_stack([
        rankdata([row[key] for row in rows]) for key in model_keys
    ])
    predictors = (
        (predictors - predictors.mean(axis=0)) / predictors.std(axis=0)
    )
    standardized_response = (
        (ranked_response - ranked_response.mean()) / ranked_response.std()
    )
    design = np.column_stack([
        np.ones(len(rows)), predictors, width_indicators
    ])
    coefficients = np.linalg.lstsq(
        design, standardized_response, rcond=None
    )[0]
    fitted = design @ coefficients
    r_squared = 1.0 - np.sum(
        (standardized_response - fitted) ** 2
    ) / np.sum(
        (standardized_response - standardized_response.mean()) ** 2
    )
    lines.extend([
        "",
        "Descriptive rank-linear model with width indicators:",
        f"  R^2={r_squared:.3f}",
    ])
    for key, coefficient in zip(model_keys, coefficients[1:]):
        lines.append(f"  standardized coefficient {key}: {coefficient:.3f}")

    lines.extend(["", "Pair-level Spearman correlations by window width:"])
    for width in sorted({row["width"] for row in rows}, reverse=True):
        selected = [row for row in rows if row["width"] == width]
        y = [row["loss_penalty"] for row in selected]
        pieces = []
        for key in metrics:
            rho, _ = correlation([row[key] for row in selected], y)
            pieces.append(f"{key}={rho:.3f}")
        lines.append(f"  width={width:g}, n={len(selected)}: " + ", ".join(pieces))

    lines.extend([
        "",
        "Median within-window Spearman correlation across target parameters",
        "(IQR across windows; this asks whether each metric orders targets",
        "within a fixed trained model):",
    ])
    names = sorted({row["run"] for row in rows})
    for key in metrics:
        values = []
        for name in names:
            selected = [row for row in rows if row["run"] == name]
            rho, _ = correlation(
                [row[key] for row in selected],
                [row["loss_penalty"] for row in selected],
            )
            if np.isfinite(rho):
                values.append(rho)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        lines.append(
            f"  {key}: n_windows={len(values)}, {median:.3f} "
            f"[{q25:.3f}, {q75:.3f}]"
        )

    lines.extend([
        "",
        "Predictor correlations (Spearman):",
    ])
    for i, key_a in enumerate(metrics):
        for key_b in metrics[i + 1:]:
            rho, _ = correlation(
                [row[key_a] for row in rows],
                [row[key_b] for row in rows],
            )
            lines.append(f"  {key_a} vs {key_b}: {rho:.3f}")

    lines.extend([
        "",
        "Caution: window-target pairs share targets and heavily overlapping",
        "training intervals. These correlations are descriptive, not",
        "independent-sample inferential tests.",
    ])
    if window_rows is not None:
        append_window_summary(lines, window_rows)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--n-process-trajectories", type=int, default=512)
    parser.add_argument("--process-seed", type=int, default=202709)
    parser.add_argument("--lyapunov-steps", type=int, default=20_000)
    parser.add_argument(
        "--output", default="figures/window_transition_similarity.png"
    )
    parser.add_argument(
        "--matrix-output", default="figures/window_transition_js_matrix.png"
    )
    parser.add_argument(
        "--window-output",
        default="figures/window_transition_window_summary.png",
    )
    parser.add_argument(
        "--main-output", default="figures/window_transition_main.png"
    )
    parser.add_argument(
        "--csv", default="figures/window_transition_similarity.csv"
    )
    parser.add_argument(
        "--summary", default="figures/window_transition_similarity_summary.txt"
    )
    parser.add_argument(
        "--window-csv",
        default="figures/window_transition_window_summary.csv",
    )
    parser.add_argument(
        "--cache", default="figures/window_transition_process_cache.npz"
    )
    args = parser.parse_args()

    grouped, r_grid = load_runs(args.runs_dir)
    example = next(iter(grouped.values()))[0]["params"]
    context_len = int(example["context_len"])
    traj_len = int(example["traj_len"])
    for runs in grouped.values():
        for run in runs:
            params = run["params"]
            if (int(params["context_len"]) != context_len
                    or int(params["traj_len"]) != traj_len
                    or int(params["n_bins"]) != args.n_bins):
                raise ValueError("window runs do not share process settings")

    cache_path = Path(args.cache)
    use_cache = False
    cache_needs_write = False
    support_distance_euclidean = None
    if cache_path.exists():
        with np.load(cache_path) as cached:
            required = {
                "process_counts", "processes", "process_js",
                "context_process_counts", "context_processes",
                "support_distance",
            }
            use_cache = (
                required.issubset(cached.files)
                and np.allclose(cached["r_grid"], r_grid)
                and int(cached["n_bins"]) == args.n_bins
                and int(cached["context_len"]) == context_len
                and int(cached["traj_len"]) == traj_len
                and int(cached["n_process_trajectories"])
                == args.n_process_trajectories
                and int(cached["process_seed"]) == args.process_seed
            )
            if use_cache:
                process_counts = cached["process_counts"]
                processes = cached["processes"]
                process_js = cached["process_js"]
                context_process_counts = cached["context_process_counts"]
                context_processes = cached["context_processes"]
                support_distance = cached["support_distance"]
                if "support_distance_euclidean" in cached.files:
                    support_distance_euclidean = cached[
                        "support_distance_euclidean"
                    ]
    if not use_cache:
        print("estimating token-transition processes", flush=True)
        (process_counts, processes,
         context_process_counts, context_processes) = standardized_processes(
            r_grid,
            args.n_process_trajectories,
            context_len,
            traj_len,
            args.n_bins,
            args.process_seed,
        )
        print("computing pairwise JS divergences", flush=True)
        process_js = pairwise_js(processes)
        print("computing pairwise transition-support distances", flush=True)
        support_distance = directed_support_distance_matrix(
            process_counts, processes, metric="manhattan"
        )
        cache_needs_write = True

    if support_distance_euclidean is None:
        print(
            "computing Euclidean transition-support distances",
            flush=True,
        )
        support_distance_euclidean = directed_support_distance_matrix(
            process_counts, processes, metric="euclidean"
        )
        cache_needs_write = True

    if cache_needs_write:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            r_grid=r_grid,
            n_bins=args.n_bins,
            context_len=context_len,
            traj_len=traj_len,
            n_process_trajectories=args.n_process_trajectories,
            process_seed=args.process_seed,
            process_counts=process_counts,
            processes=processes,
            process_js=process_js,
            context_process_counts=context_process_counts,
            context_processes=context_processes,
            support_distance=support_distance,
            support_distance_euclidean=support_distance_euclidean,
        )

    eval_lambda = lyapunov_grid(r_grid, n_steps=args.lyapunov_steps)
    print("reconstructing training transitions and assembling pairs", flush=True)
    rows = make_pair_rows(
        grouped,
        r_grid,
        process_counts,
        processes,
        process_js,
        context_process_counts,
        context_processes,
        support_distance,
        support_distance_euclidean,
        eval_lambda,
        args.n_bins,
    )
    settings = {
        "n_process_trajectories": args.n_process_trajectories,
        "n_bins": args.n_bins,
        "context_len": context_len,
        "traj_len": traj_len,
    }
    dense_r = np.linspace(float(r_grid.min()), float(r_grid.max()), 3501)
    dense_lambda = lyapunov_grid(
        dense_r, n_steps=args.lyapunov_steps
    )
    window_rows = summarize_windows(rows, dense_r, dense_lambda)
    plot_metric_comparison(rows, args.output)
    plot_process_matrix(r_grid, process_js, support_distance,
                        args.matrix_output)
    plot_window_summary(window_rows, args.window_output)
    plot_main_combination(rows, window_rows, args.main_output)
    write_csv(rows, args.csv)
    write_csv(window_rows, args.window_csv)
    write_summary(rows, args.summary, settings, window_rows=window_rows)
    print(f"analyzed {len(rows)} OOD window-target pairs")
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")
    print(
        f"saved {args.matrix_output} and "
        f"{Path(args.matrix_output).with_suffix('.pdf')}"
    )
    print(
        f"saved {args.window_output} and "
        f"{Path(args.window_output).with_suffix('.pdf')}"
    )
    print(
        f"saved {args.main_output} and "
        f"{Path(args.main_output).with_suffix('.pdf')}"
    )
    print(f"saved {args.csv}")
    print(f"saved {args.window_csv}")
    print(f"saved {args.summary}")


if __name__ == "__main__":
    main()
