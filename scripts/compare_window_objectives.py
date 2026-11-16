"""Compare restricted-window transfer correlations for CE and scalar MSE.

The represented-state and transition predictors are objective-independent, so
this script reuses the pair-level predictors computed by
``analyze_window_transition_similarity.py`` and replaces only the response.
For each objective, the response is normalized against models of the same
window width whose training interval contains the target parameter.

CE response:
    log10(1 + CE_W(r)) - median_{W': r in W'} log10(1 + CE_W'(r))

MSE response:
    log10(MSE_W(r)) - median_{W': r in W'} log10(MSE_W'(r))
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TOL = 1e-9


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
        return float("nan")
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def residual(values, controls):
    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    design = np.column_stack([np.ones(len(values)), controls])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def partial_spearman(x, y, controls):
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    ranked_controls = np.column_stack([
        rankdata(controls[:, i]) for i in range(controls.shape[1])
    ])
    return correlation(
        residual(ranked_x, ranked_controls),
        residual(ranked_y, ranked_controls),
        method="pearson",
    )


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_objective_runs(root, objective):
    grouped = defaultdict(list)
    by_name = {}
    for path in sorted(Path(root).glob("win_w*_s*_seed*/eval_per_r.npz")):
        with (path.parent / "params.json").open() as handle:
            params = json.load(handle)
        with np.load(path) as result:
            r_grid = result["r_grid"].astype(float)
            if objective == "ce":
                raw = result["ce_per_r"].astype(float)
                transformed = np.log10(1.0 + np.maximum(raw, 0.0))
            elif objective == "mse":
                # Scalar-output evaluations store RMS; square before logging.
                raw = result["rms_per_r"].astype(float) ** 2
                transformed = np.log10(np.maximum(raw, 1e-12))
            else:
                raise ValueError(f"unknown objective: {objective}")
        run = {
            "name": path.parent.name,
            "width": float(params["width"]),
            "start": float(params["start"]),
            "end": float(params["start"] + params["width"]),
            "r": r_grid,
            "transformed": transformed,
        }
        grouped[run["width"]].append(run)
        by_name[run["name"]] = run
    if not by_name:
        raise FileNotFoundError(f"no runs found under {root}")
    for width in grouped:
        grouped[width] = sorted(grouped[width], key=lambda run: run["start"])
    return grouped, by_name


def objective_penalties(root, objective):
    grouped, by_name = load_objective_runs(root, objective)
    penalties = {}
    for width, runs in grouped.items():
        r_grid = runs[0]["r"]
        for run in runs:
            if not np.allclose(run["r"], r_grid):
                raise ValueError(f"evaluation grid differs for {run['name']}")
        values = np.stack([run["transformed"] for run in runs])
        support = np.stack([
            (r_grid >= run["start"] - TOL)
            & (r_grid <= run["end"] + TOL)
            for run in runs
        ])
        with np.errstate(all="ignore"):
            reference = np.nanmedian(np.where(support, values, np.nan), axis=0)
        if np.any(~np.isfinite(reference)):
            raise ValueError(f"width {width:g} does not cover evaluation grid")
        for run in runs:
            penalties[run["name"]] = run["transformed"] - reference
    return by_name, penalties


def attach_response(base_rows, by_name, penalties):
    result = []
    for row in base_rows:
        name = row["run"]
        if name not in by_name:
            raise KeyError(f"missing objective run {name}")
        run = by_name[name]
        target = float(row["target_r"])
        matches = np.flatnonzero(np.isclose(run["r"], target, atol=1e-10))
        if len(matches) != 1:
            raise ValueError(f"could not uniquely match r={target} for {name}")
        copied = dict(row)
        copied["loss_penalty"] = float(penalties[name][matches[0]])
        result.append(copied)
    return result


def width_controls(rows):
    widths = sorted({float(row["width"]) for row in rows})
    indicators = np.column_stack([
        np.array([float(row["width"]) == width for row in rows], dtype=float)
        for width in widths[1:]
    ])
    parameter_distance = np.array([
        float(row["parameter_distance"]) for row in rows
    ])
    return np.column_stack([parameter_distance, indicators])


def window_rows(pair_rows, window_metadata):
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[row["run"]].append(row)
    metrics = [
        "loss_penalty",
        "nearest_transition_support_distance",
        "nearest_transition_support_distance_euclidean",
        "context_occupancy_js",
        "parameter_distance",
    ]
    result = []
    for name, selected in grouped.items():
        meta = window_metadata[name]
        item = {
            "run": name,
            "width": float(meta["width"]),
            "center": float(meta["center"]),
            "chaos_fraction": float(meta["chaos_fraction"]),
        }
        for metric in metrics:
            item[f"mean_{metric}"] = float(np.mean([
                float(row[metric]) for row in selected
            ]))
        result.append(item)
    return sorted(result, key=lambda row: (row["width"], row["center"]))


def residual_window_correlation(rows, x_key, y_key):
    center = np.array([row["center"] for row in rows])
    widths = sorted({row["width"] for row in rows})
    controls = np.column_stack(
        [center, center ** 2]
        + [
            np.array([row["width"] == width for row in rows], dtype=float)
            for width in widths[1:]
        ]
    )
    return correlation(
        residual([row[x_key] for row in rows], controls),
        residual([row[y_key] for row in rows], controls),
        method="pearson",
    )


def analyze(rows, windows):
    response = np.array([float(row["loss_penalty"]) for row in rows])
    controls = width_controls(rows)
    metrics = {
        "parameter_distance": "parameter_distance",
        "context_occupancy_js": "context_occupancy_js",
        "support_l1": "nearest_transition_support_distance",
        "support_l2": "nearest_transition_support_distance_euclidean",
    }
    result = {}
    for short, key in metrics.items():
        values = np.array([float(row[key]) for row in rows])
        result[f"pair_spearman_{short}"] = correlation(values, response)
        if short != "parameter_distance":
            result[f"pair_partial_{short}"] = partial_spearman(
                values, response, controls
            )

    window_metrics = {
        "context_occupancy_js": "mean_context_occupancy_js",
        "support_l1": "mean_nearest_transition_support_distance",
        "support_l2": "mean_nearest_transition_support_distance_euclidean",
    }
    mean_response = [row["mean_loss_penalty"] for row in windows]
    result["window_spearman_chaos"] = correlation(
        [row["chaos_fraction"] for row in windows], mean_response
    )
    result["window_residual_chaos"] = residual_window_correlation(
        windows, "chaos_fraction", "mean_loss_penalty"
    )
    for short, key in window_metrics.items():
        result[f"window_spearman_{short}"] = correlation(
            [row[key] for row in windows], mean_response
        )
        result[f"window_residual_{short}"] = residual_window_correlation(
            windows, key, "mean_loss_penalty"
        )
    return result


def write_long_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(next(iter(results.values())))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["statistic", *results])
        for key in keys:
            writer.writerow([key, *[f"{results[obj][key]:.6f}" for obj in results]])


def write_summary(results, path, ce_check):
    lines = [
        "Restricted-window objective/output robustness",
        "=============================================",
        "CE response: same-width, same-target normalized log10(1 + CE)",
        "MSE response: same-width, same-target normalized log10(MSE)",
        "Pair-level n=14658; window-level n=53 overlapping windows",
        f"Maximum CE reconstruction difference: {ce_check:.3e}",
        "",
    ]
    for key in sorted(results["CE"]):
        lines.append(
            f"{key}: CE={results['CE'][key]:.3f}, "
            f"MSE={results['MSE'][key]:.3f}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-csv", default="figures/window_transition_similarity.csv"
    )
    parser.add_argument(
        "--window-csv", default="figures/window_transition_window_summary.csv"
    )
    parser.add_argument("--ce-runs", default="runs_win_fix_ce")
    parser.add_argument("--mse-runs", default="runs_win_fix_mse")
    parser.add_argument(
        "--output-csv", default="figures/window_objective_robustness.csv"
    )
    parser.add_argument(
        "--summary", default="figures/window_objective_robustness_summary.txt"
    )
    args = parser.parse_args()

    base_rows = read_csv(args.pair_csv)
    metadata_rows = read_csv(args.window_csv)
    metadata = {row["run"]: row for row in metadata_rows}

    objective_rows = {}
    objective_windows = {}
    for label, root, objective in (
        ("CE", args.ce_runs, "ce"),
        ("MSE", args.mse_runs, "mse"),
    ):
        by_name, penalties = objective_penalties(root, objective)
        objective_rows[label] = attach_response(base_rows, by_name, penalties)
        objective_windows[label] = window_rows(objective_rows[label], metadata)

    ce_check = max(
        abs(float(old["loss_penalty"]) - float(new["loss_penalty"]))
        for old, new in zip(base_rows, objective_rows["CE"])
    )
    if ce_check > 1e-8:
        raise ValueError(
            f"reconstructed CE response differs from stored analysis: {ce_check}"
        )

    results = {
        label: analyze(objective_rows[label], objective_windows[label])
        for label in ("CE", "MSE")
    }
    write_long_csv(results, args.output_csv)
    write_summary(results, args.summary, ce_check)
    print(f"saved {args.output_csv}")
    print(f"saved {args.summary}")


if __name__ == "__main__":
    main()
