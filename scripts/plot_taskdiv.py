"""Figure-2 analogue for the task-diversity sweeps.

Modelled on Figure 2 of Raventos et al., "Pretraining task diversity and the
emergence of non-Bayesian in-context learning for regression" (NeurIPS 2023):
loss against the number of pretraining tasks, shown separately for tasks seen
during pretraining (top row) and new tasks (bottom row).

Here a task is a parameter value r of the logistic family. Two sweeps run over
the same task counts:

  uncontrolled  one trajectory per r, so the task count and the total amount of
                data grow together (the original budget-sweep design)
  controlled    total trajectories pinned at --controlled_total, so raising the
                task count *reduces* trajectories per task and only diversity
                changes

Both sweeps train an identical fixed number of gradient steps, and each run
reports two models read off that one trajectory: the final weights (fixed-step
protocol) and the best-validation checkpoint (early-stopping protocol).

Seen tasks are measured directly at the training r-values (`ce_at_train_r`),
not inferred from distance to a grid. New tasks use the fixed 300-point
full-range grid, identical for every run and so comparable across the sweep.

Torch-free: reads eval_per_r.npz / params.json only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROTOCOL = {
    "final":   {"color": "#D85A30", "ls": "-",  "marker": "o",
                "label": "Fixed steps (final)"},
    "bestval": {"color": "#1B2A4A", "ls": "--", "marker": "s",
                "label": "Early stopping (best val)"},
}
SWEEP_TITLE = {
    "uncontrolled": "Uncontrolled: 1 trajectory per $r$\n"
                    "(task count and total data grow together)",
    "controlled":   "Controlled: total trajectories fixed\n"
                    "(only task count changes)",
}


def load_sweep(root):
    """Return {m: run} for one sweep directory."""
    runs = {}
    for path in sorted(Path(root).glob("*/eval_per_r.npz")):
        params_path = path.parent / "params.json"
        if not params_path.exists():
            continue
        with open(params_path) as handle:
            params = json.load(handle)
        with np.load(path) as z:
            run = {"name": path.parent.name, "params": params,
                   "m": int(params["m"]),
                   "total": int(params["n_train_traj_actual"]),
                   "new_final": z["ce_per_r"].copy()}
            run["seen_final"] = (z["ce_at_train_r"].copy()
                                 if "ce_at_train_r" in z.files else None)
            run["new_bestval"] = (z["ce_per_r_bestval"].copy()
                                  if "ce_per_r_bestval" in z.files else None)
            run["seen_bestval"] = (z["ce_at_train_r_bestval"].copy()
                                   if "ce_at_train_r_bestval" in z.files else None)
        runs[run["m"]] = run
    return runs


def _series(runs, row, protocol):
    """(task counts, mean CE) for one row/protocol, skipping missing arrays."""
    ms, vals = [], []
    for m in sorted(runs):
        arr = runs[m].get(f"{row}_{protocol}")
        if arr is None or len(arr) == 0:
            continue
        ms.append(m)
        vals.append(float(np.mean(arr)))
    return np.array(ms), np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncontrolled_runs", default="runs_div_uncontrolled")
    ap.add_argument("--controlled_runs", default="runs_div_controlled")
    ap.add_argument("--out_dir", default="figures_taskdiv")
    a = ap.parse_args()

    sweeps = {"uncontrolled": load_sweep(a.uncontrolled_runs),
              "controlled": load_sweep(a.controlled_runs)}

    print("=" * 72)
    for key, runs in sweeps.items():
        print(f"{key}: {len(runs)} runs | task counts {sorted(runs)}")
        for m in sorted(runs):
            r = runs[m]
            seen = (f"{np.mean(r['seen_final']):.4f}"
                    if r["seen_final"] is not None else "n/a")
            print(f"    m={m:<6} total={r['total']:<6} "
                  f"traj/r={r['total'] / m:<7.1f} "
                  f"bestval={'yes' if r['new_bestval'] is not None else 'NO':<3} "
                  f"new={np.mean(r['new_final']):.4f} seen={seen}")
    print("=" * 72)

    if not any(sweeps.values()):
        raise FileNotFoundError("no runs found in either sweep directory")

    rows = ["seen", "new"]
    cols = [k for k in ("uncontrolled", "controlled") if sweeps[k]]
    fig, axes = plt.subplots(2, len(cols), figsize=(6.0 * len(cols), 8.4),
                             squeeze=False, sharex=True)

    for j, key in enumerate(cols):
        runs = sweeps[key]
        for i, row in enumerate(rows):
            ax = axes[i][j]
            for protocol, style in PROTOCOL.items():
                ms, vals = _series(runs, row, protocol)
                if len(ms) == 0:
                    continue
                ax.plot(ms, vals, style["marker"] + style["ls"],
                        color=style["color"], lw=1.8, ms=5,
                        mfc="white" if protocol == "bestval" else style["color"],
                        label=style["label"])
            ax.axhline(np.log(64), color="gray", ls=":", lw=1.1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both", lw=0.4)
            if i == 0:
                ax.set_title(SWEEP_TITLE[key], fontsize=10.5)
            if j == 0:
                label = ("Seen tasks\n(evaluated at training $r$)" if row == "seen"
                         else "New tasks\n(full-range grid)")
                ax.set_ylabel(f"{label}\nMean cross-entropy (nats)", fontsize=9.5)
            if i == len(rows) - 1:
                ax.set_xlabel("Number of training tasks (distinct $r$ values)")
            ax.legend(fontsize=7.5)

    # Share the y-range within each row so the two sweeps are directly comparable.
    for i in range(2):
        lims = [axes[i][j].get_ylim() for j in range(len(cols))]
        lo = min(l[0] for l in lims)
        hi = max(l[1] for l in lims)
        for j in range(len(cols)):
            axes[i][j].set_ylim(lo, hi)

    total = next((r["total"] for runs in [sweeps["controlled"]] if runs
                  for r in [runs[max(runs)]]), None)
    fig.suptitle("Task diversity and in-context generalization on the logistic "
                 "family\nDotted line: uniform prediction over 64 bins"
                 + (f"   |   controlled sweep holds total trajectories at {total}"
                    if total else ""),
                 fontsize=11)
    fig.tight_layout()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"task_diversity_controlled.{ext}", dpi=160,
                    bbox_inches="tight")
    print(f"wrote {out}/task_diversity_controlled.png")


if __name__ == "__main__":
    main()
