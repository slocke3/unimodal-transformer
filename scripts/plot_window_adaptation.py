"""Where each window-trained model adapts beyond its training range.

An absolute error cutoff cannot answer this. Just outside a window, applying the
nearest training map already gets the error small, and below r = 1 every map
sends the collapsed orbit to 0, so a cutoff scores those cells as successes
without the model adapting at all. The bar that separates adaptation from
knowing the training range is the clamped baseline: the map at the nearest edge
of the training window, which is the best any single in-window map can do, since
its error (r' - r)^2 E[(x(1-x))^2] is convex in r'. It is given the exact state
x_T, so it is the stronger version of that baseline for a binned-input model.

Each out-of-window cell is scored by how far the model gets from the clamped
error C down to the error S of a same-width model trained on a window containing
r, in decades:  a = log(C / M) / log(C / S).  a = 0 is no better than the
training range alone, a = 1 is as good as having been trained there. Cells where
C is already within --room decades of S have nothing to adapt to and are left
grey; that covers the training windows themselves, a thin margin beside each,
and r < 1.

Square-loss runs only: the baseline is in map units. The evaluation contexts are
regenerated exactly as evaluate_per_r draws them (seed + 7), so C is computed on
the same trajectories the models were scored on.
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

from src.maps import iterate_map


def load(runs_dir):
    runs, shared = [], None
    for path in sorted(Path(runs_dir).glob("win_w*_s*_seed*/eval_per_r.npz")):
        params = json.loads((path.parent / "params.json").read_text())
        setup = tuple(params[k] for k in ("context_len", "traj_len", "burn_in",
                                          "n_eval_per_r", "seed", "full_lo",
                                          "full_hi", "r_eval_points"))
        assert shared in (None, setup), f"{path.parent} was evaluated differently"
        shared = setup
        with np.load(path) as result:
            runs.append(dict(
                width=float(params["width"]), start=float(params["start"]),
                end=float(params["start"] + params["width"]),
                mse=result["rms_per_r"] ** 2,
                inside=result["in_window"].astype(bool),
                r=result["r_grid"],
            ))
    if not runs:
        raise FileNotFoundError(f"no window-sweep results under {runs_dir}")
    return runs, dict(zip(("context_len", "traj_len", "burn_in", "n_eval_per_r",
                           "seed"), shared))


def state_moment(r_grid, context_len, traj_len, burn_in, n_eval_per_r, seed):
    """E[(x_T (1 - x_T))^2] over the evaluation windows at each r, where x_T is
    the last context element -- the exact draws of evaluate_per_r."""
    rng = np.random.default_rng(seed + 7)
    moment = np.empty(len(r_grid))
    for i, r in enumerate(r_grid):
        last = []
        for _ in range(n_eval_per_r):
            traj = iterate_map(rng.uniform(0.05, 0.95), r, burn_in + traj_len)[burn_in:]
            last.append(traj[context_len - 1 : len(traj) - 1])
        last = np.concatenate(last)
        moment[i] = np.mean((last * (1.0 - last)) ** 2)
    return moment


def score(runs, setup, room):
    r = runs[0]["r"]
    moment = state_moment(r, **setup)
    by_width = defaultdict(list)
    for run in runs:
        by_width[run["width"]].append(run)
    for group in by_width.values():
        log_mse = np.stack([np.log(run["mse"]) for run in group])
        inside = np.stack([run["inside"] for run in group])
        # a same-width specialist at each r; overlapping windows are averaged in log
        specialist = np.exp(np.nanmean(np.where(inside, log_mse, np.nan), axis=0))
        for run in group:
            clamp = (r - np.clip(r, run["start"], run["end"])) ** 2 * moment
            with np.errstate(divide="ignore", invalid="ignore"):
                scored = ~run["inside"] & (np.log10(clamp / specialist) >= room)
                a = np.log(clamp / run["mse"]) / np.log(clamp / specialist)
            run["a"] = np.where(scored, a, np.nan)
    return by_width


def plot(by_width, output, room):
    widths = sorted(by_width, reverse=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), layout="constrained")
    for ax, width in zip(axes.flat, widths):
        runs = sorted(by_width[width], key=lambda run: run["start"])
        r = runs[0]["r"]
        matrix = np.ma.masked_invalid(np.stack([np.clip(run["a"], 0, 1) for run in runs]))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                          extent=[r[0], r[-1], len(runs), 0], cmap=cmap,
                          vmin=0, vmax=1)
        for row, run in enumerate(runs):
            ax.add_patch(Rectangle((run["start"], row), width, 1, fill=False,
                                   edgecolor="#1a1a1a", linewidth=0.8))
        ax.set_title(f"Width {width:g} · {len(runs)} training windows")
        ax.set_xlabel(r"Evaluation parameter $r$")
        ax.set_ylabel("Training window (ordered by start)")
        ax.set_yticks([0.5, len(runs) - 0.5])
        ax.set_yticklabels([f"{runs[0]['start']:.3g}", f"{runs[-1]['start']:.3g}"])

    colorbar = fig.colorbar(image, ax=axes, pad=0.015, shrink=0.9)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_ticklabels(["0  no better than\n    training range", "0.25", "0.5",
                             "0.75", "1  as good as a model\n    trained at this r"])
    colorbar.set_label("Adaptation: share of the gap closed, in decades of MSE")
    fig.suptitle(
        "Where does each window-trained transformer adapt beyond its training range?\n"
        "0 = the clamped baseline (best map inside the training window); "
        "1 = a same-width model trained on a window containing $r$.\n"
        f"Grey: the clamped baseline is already within {10 ** room:g}× of that model, "
        "so there is nothing to adapt to. Outlined boxes are training ranges.",
        fontsize=12.5,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs_win_fix_mse")
    parser.add_argument("--output", default="figures_window/window_fixed_mse_adaptation.png")
    parser.add_argument("--room", type=float, default=1.0,
                        help="decades the clamped baseline must sit above the "
                             "specialist for a cell to be scored")
    args = parser.parse_args()
    runs, setup = load(args.runs_dir)
    by_width = score(runs, setup, args.room)
    values = np.concatenate([run["a"] for run in runs])
    scored = np.isfinite(values)
    print(f"{scored.sum()} scored out-of-window cells: "
          f"{(values[scored] > 0).mean():.1%} beat the clamped baseline, "
          f"{(values[scored] >= 0.5).sum()} at least halfway, "
          f"median {np.median(values[scored]):.2f}")
    plot(by_width, args.output, args.room)
    print(f"saved {args.output} and {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
