"""Descriptive controls and seed-aware summaries for restricted-window transfer.

The original 53-window sweep supplies a FROZEN same-width/target reference.
New seeds are averaged within each interval before correlations are computed;
they do not become additional independent windows. No iid-pair p-values or
confidence intervals are reported. Optional new runs use the original process
cache and the original analysis's exact training-data reconstruction.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_window_transition_similarity as transition

SUPPORT = "nearest_transition_support_distance"
STATE = "context_occupancy_js"
RESPONSE = "loss_penalty"
METRICS = [SUPPORT, SUPPORT + "_euclidean"]


def read_csv(path):
    with open(path) as f:
        return [{k: v if k == "run" else float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def values(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def window_key(row):
    return (round(float(row["width"]), 7), round(float(row["start"]), 7))


def aggregate_seeds(rows):
    """One row per interval/target, averaging equal-weight run-level metrics."""
    groups = defaultdict(list)
    for row in rows:
        groups[(*window_key(row), round(row["target_r"], 12))].append(row)
    out = []
    for (width, start, _), group in sorted(groups.items()):
        names = [r["run"] for r in group]
        if len(names) != len(set(names)):
            raise ValueError("duplicate run/target pair: " + str(names))
        result = {k: float(np.mean([r[k] for r in group]))
                  for k in group[0] if k != "run"}
        result.update(run=f"w{width:g}_s{start:g}", n_seeds=len(group))
        out.append(result)
    return out


def corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return corr(transition.rankdata(x), transition.rankdata(y))


def absorb_groups(matrix, groups, tolerance=1e-11):
    """Residualize against categorical fixed effects by alternating projections.

    This is equivalent to projecting out all dummy columns jointly, including
    for the incomplete window-by-target grid. Sequential demeaning just once
    is NOT equivalent, so iterate to convergence and fail rather than silently
    returning an unconverged projection.
    """
    residual = matrix.copy()
    residual -= residual.mean(axis=0)
    codes = [np.unique(g, return_inverse=True)[1] for g in groups]
    for _ in range(1000):
        before = residual.copy()
        for g in codes:
            counts = np.bincount(g)
            means = np.column_stack([
                np.bincount(g, weights=residual[:, j]) / counts
                for j in range(residual.shape[1])])
            residual -= means[g]
        if np.max(np.abs(residual - before)) < tolerance:
            return residual
    raise RuntimeError("categorical-effect projection did not converge")


def partial_rank(rows, predictor, controls=(), categories=()):
    matrix = np.column_stack([transition.rankdata(values(rows, key))
                              for key in (predictor, RESPONSE, *controls)])
    scale = matrix.std(axis=0)
    matrix = (matrix - matrix.mean(axis=0)) / np.where(scale > 0, scale, 1)
    matrix = absorb_groups(matrix, [[r[k] for r in rows] for k in categories])
    residual = matrix[:, :2]
    if controls:
        c = matrix[:, 2:]
        residual = residual - c @ np.linalg.lstsq(c, residual, rcond=None)[0]
    return corr(residual[:, 0], residual[:, 1])


CONTROL_SPECS = [
    ("Pooled", (), ()),
    ("Distance + width", ("parameter_distance",), ("width",)),
    ("Distance + width + state", ("parameter_distance", STATE), ("width",)),
    ("Also target effects", ("parameter_distance", STATE), ("width", "target_r")),
    ("Target + window effects + state/distance", ("parameter_distance", STATE),
     ("run", "target_r")),
]


def analyze(rows):
    results = []
    for metric in METRICS:
        for label, controls, categories in CONTROL_SPECS:
            results.append(dict(association=label, metric=metric,
                                n_pairs=len(rows), n_windows=len({r["run"] for r in rows}),
                                rho=partial_rank(rows, metric, controls, categories)))
        for width in sorted(set(values(rows, "width"))):
            sub = [r for r in rows if r["width"] == width]
            results.append(dict(association=f"Width {width:g}", metric=metric,
                                n_pairs=len(sub), n_windows=len({r["run"] for r in sub}),
                                rho=partial_rank(sub, metric)))
    return results


def within_window(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    result = []
    for name, sub in sorted(grouped.items()):
        first = sub[0]
        result.append(dict(run=name, width=first["width"], start=first["start"],
                           n_targets=len(sub), n_seeds=first["n_seeds"],
                           support_rho=spearman(values(sub, SUPPORT), values(sub, RESPONSE)),
                           state_rho=spearman(values(sub, STATE), values(sub, RESPONSE))))
    return result


def common_target_summary(pairs, reference, grid):
    """Compare the prespecified replication intervals on identical targets.

    The upper endpoint is the smallest source start in the replication set.
    Hence every target is regular and out of support for EVERY selected model.
    Both the penalty and unnormalized transformed loss are retained.
    """
    from make_window_replication_jobs import selected_windows
    selected = {(w,s): group for w,s,group in selected_windows()}
    edge = min(s for w,s in selected)
    grouped = defaultdict(list)
    for row in pairs:
        key = window_key(row)
        if key in selected and row['target_r'] < edge - transition.TOL:
            grouped[key].append(row)
    refs = {w: transition.same_width_reference(grid, runs) for w,runs in reference.items()}
    result = []
    for (width,start), sub in sorted(grouped.items()):
        penalties = values(sub, RESPONSE)
        idx = np.array([np.argmin(abs(grid-r['target_r'])) for r in sub])
        ell = penalties + refs[width][idx]
        result.append(dict(width=width, start=start, group=selected[(width,start)],
                           n_seeds=sub[0]['n_seeds'], n_targets=len(sub),
                           target_min=float(grid.min()), target_upper_exclusive=edge,
                           mean_penalty=float(penalties.mean()),
                           mean_log10_one_plus_ce=float(ell.mean())))
    return result


def plot_controls(results, window_rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = [s[0] for s in CONTROL_SPECS]
    for j, (metric, color, name) in enumerate(zip(METRICS, ["#2878b5", "#d55e00"],
                                                 ["Manhattan", "Euclidean"])):
        vals = [next(r["rho"] for r in results if r["association"] == label
                     and r["metric"] == metric) for label in labels]
        axes[0].plot(vals, np.arange(len(labels)) + (j - .5) * .14, "o", color=color, label=name)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rank correlation after specified controls")
    axes[0].axvline(0, color=".65", lw=.8)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].set_title("(a) Pooled and adjusted associations", loc="left", fontsize=11)
    for i, width in enumerate(sorted({r["width"] for r in window_rows})):
        sub = [r["support_rho"] for r in window_rows if r["width"] == width]
        jitter = np.linspace(-.17, .17, len(sub))
        axes[1].scatter(i + jitter, sub, s=25, alpha=.8)
        axes[1].plot([i-.22, i+.22], [np.nanmedian(sub)]*2, color="black", lw=2)
    axes[1].set_xticks(range(4), ["0.125", "0.25", "0.5", "1"])
    axes[1].set_xlabel("Training-window width")
    axes[1].set_ylabel("Within-window Spearman correlation")
    axes[1].axhline(0, color=".65", lw=.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("(b) Ordering targets within each window", loc="left", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_old_rows(rows, reference, grid):
    """Reject stale CSVs (e.g. early stopping) rather than mixing protocols."""
    by_name = {r["name"]: r for group in reference.values() for r in group}
    refs = {w: transition.same_width_reference(grid, group) for w, group in reference.items()}
    if {r["run"] for r in rows} != set(by_name):
        raise ValueError("CSV and reference run inventories differ")
    for row in rows:
        run = by_name[row["run"]]
        i = int(np.argmin(abs(grid - row["target_r"])))
        expected = transition.transformed_loss(run["ce"])[i] - refs[run["width"]][i]
        if abs(grid[i] - row["target_r"]) > 1e-10 or not np.isclose(expected, row[RESPONSE], atol=1e-10):
            raise ValueError("CSV loss differs from frozen reference for " + row["run"])


def append_new_runs(rows, roots, reference, grid, cache_path):
    if not roots:
        return rows
    all_new = defaultdict(list)
    names = {r["run"] for r in rows}
    identities = {(window_key(r), int(r['params']['seed']))
                  for group in reference.values() for r in group}
    expected = next(iter(reference.values()))[0]["params"]
    checks = dict(context_len=50, traj_len=150, n_bins=64, burn_in=0,
                  output_mode="bins", n_train_traj=8000, max_steps=160000,
                  all_positions=False, input_mode="bins", synonyms=1, noise_bins=0,
                  dropout=.1, d_model=128, n_heads=4, n_layers=4, max_val_traj=600)
    for root in roots:
        new, new_grid = transition.load_runs(root)
        if not np.array_equal(new_grid, grid):
            raise ValueError("new evaluation grid differs")
        for width, group in new.items():
            if width not in reference:
                raise ValueError("no fixed reference for width " + str(width))
            for run in group:
                if run["name"] in names:
                    raise ValueError("duplicate run: " + run["name"])
                names.add(run["name"])
                identity = (window_key(run), int(run['params']['seed']))
                if identity in identities:
                    raise ValueError("duplicate interval/seed: " + str(identity))
                identities.add(identity)
                for k, value in checks.items():
                    if run["params"].get(k, value) != expected.get(k, value):
                        raise ValueError(f"incompatible {k}: {run['name']}")
                if run["params"].get("family", "logistic") != "logistic":
                    raise ValueError("requires logistic runs")
                for k in ('n_bins_in', 'n_bins_out'):
                    if (run['params'].get(k) or run['params']['n_bins']) != 64:
                        raise ValueError("requires 64-bin inputs and outputs")
                if run['params']['m'] != reference[width][0]['params']['m']:
                    raise ValueError("training-parameter density differs: " + run['name'])
                all_new[width].append(run)
    with np.load(cache_path) as c:
        for key, value in [("n_bins", 64), ("context_len", 50), ("traj_len", 150),
                           ("n_process_trajectories", 512), ("process_seed", 202709)]:
            if int(c[key]) != value:
                raise ValueError("unexpected process cache " + key)
        if not np.array_equal(c["r_grid"], grid):
            raise ValueError("cache grid differs")
        generated = transition.make_pair_rows(
            all_new, grid, c["process_counts"], c["processes"], c["process_js"],
            c["context_process_counts"], c["context_processes"], c["support_distance"],
            c["support_distance_euclidean"], transition.lyapunov_grid(grid), 64,
            reference_grouped=reference)
    return rows + generated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="figures/window_transition_similarity.csv")
    ap.add_argument("--reference-runs", default="runs_win_fix_ce")
    ap.add_argument("--additional-runs", nargs="*", default=[])
    ap.add_argument("--cache", default="figures/window_transition_process_cache.npz")
    ap.add_argument("--output-dir", default="figures_rebuttal/window_transfer")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.pairs)
    reference, grid = transition.load_runs(args.reference_runs)
    validate_old_rows(rows, reference, grid)
    rows = append_new_runs(rows, args.additional_runs, reference, grid, args.cache)
    pairs = aggregate_seeds(rows)
    results = analyze(pairs)
    within = within_window(pairs)
    dense_r = np.linspace(grid.min(), grid.max(), 3501)
    dense_lambda = transition.lyapunov_grid(dense_r)
    windows = transition.summarize_windows(pairs, dense_r, dense_lambda)
    raw_windows = transition.summarize_windows(rows, dense_r, dense_lambda)
    common = common_target_summary(pairs, reference, grid)
    # Seed variation of each run's mean; error bars must not use pooled targets
    # or mistake multiple seeds of one interval for different windows.
    for window in windows:
        seeds = [r for r in raw_windows if window_key(r) == window_key(window)]
        losses = values(seeds, "mean_loss_penalty")
        window.update(n_seeds=len(seeds), penalty_seed_min=float(losses.min()),
                      penalty_seed_max=float(losses.max()),
                      penalty_seed_sd=float(losses.std(ddof=1)) if len(seeds)>1 else float("nan"))
    for name, records in [("pair_controls", results), ("within_window", within),
                          ("window_means", windows), ("run_means", raw_windows),
                          ("common_regular_targets", common),
                          ("pairs_by_run", rows), ("pairs_seed_mean", pairs)]:
        transition.write_csv(records, out / (name + ".csv"))
    plot_controls(results, within, out / "transition_controls.png")
    lines = ["Restricted-window transfer: descriptive checks", "",
             f"Runs: {len({r['run'] for r in rows})}; distinct intervals: {len(windows)}; "
             f"interval-target pairs after seed averaging: {len(pairs)}",
             f"Frozen reference: {args.reference_runs}",
             "Seeds averaged within intervals; no iid-pair significance tests.",
             "State control is context-token occupancy JS, not an oracle conditional model.", ""]
    for metric in METRICS:
        lines.append(metric)
        lines += [f"  {r['association']}: {r['rho']:.3f}" for r in results if r['metric']==metric]
    for key in ["support_rho", "state_rho"]:
        v=values(within,key)
        lines.append(f"Within-window {key}: median {np.nanmedian(v):.3f}, "
                     f"IQR {np.nanquantile(v,.25):.3f} to {np.nanquantile(v,.75):.3f}, "
                     f"positive {np.sum(v>0)}/{np.sum(np.isfinite(v))}")
    for name, subset in [("all",windows), ("positive chaotic coverage",[r for r in windows if r['chaos_fraction']>0]),
                         ("zero chaotic coverage",[r for r in windows if r['chaos_fraction']==0])]:
        lines.append(f"Window-mean support/penalty ({name}, n={len(subset)}): "
                     f"rho={spearman(values(subset,'mean_'+SUPPORT),values(subset,'mean_loss_penalty')):.3f}")
    lines += ["", "Prespecified replication subset on COMMON regular targets:"]
    for width in sorted(set(r['width'] for r in common)):
        parts=[]
        for group in ('regular','chaotic'):
            sub=[r for r in common if r['width']==width and r['group']==group]
            parts.append(f"{group} mean penalty={values(sub,'mean_penalty').mean():.3f} "
                         f"({len(sub)} intervals)")
        lines.append(f"  width={width:g}: " + "; ".join(parts))
    lines += ["", "These controls distinguish pooled/between-window association from ordering",
              "targets within a fixed window. Fixed effects absorb all average target and",
              "window differences, including variation that could be part of the mechanism.",
              "Neither persistence nor disappearance establishes a causal explanation.",
              "Same-width normalization uses empirical trained models, not a Bayes error floor.",
              "Finite-trajectory support overlap does not require matching conditional probabilities."]
    (out / "summary.txt").write_text("\n".join(lines)+"\n")
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
