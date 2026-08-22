"""Figure-2-style view of the budget sweep: loss vs number of training tasks.

Modelled on Figure 2 of Raventos et al., "Pretraining task diversity and the
emergence of non-Bayesian in-context learning for regression" (NeurIPS 2023),
where loss is plotted against the number of pretraining tasks, separately for
tasks seen during pretraining and for new tasks.

Here a "task" is a parameter value r of the logistic family, and N is the number
of distinct r-values the model was trained on. Seen / new is resolved by each
evaluation r's distance to the nearest training r: the eval grid is a fixed
300-point grid shared by every run, so a grid point sitting within SEEN_TOL of a
training r is effectively a task the model saw.

IMPORTANT CAVEAT, stated on the figure itself: these runs used one trajectory
per r (m == N == n_train_traj), so N sets task diversity AND total data at the
same time. Raventos et al. hold total data fixed and vary only the task count.
Separating the two needs the `diversity` mode of gen_jobs.py, which fixes
--n_train_traj and sweeps --m.

Torch-free: reads eval_per_r.npz / params.json only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# An eval r within this distance of a training r counts as a "seen" task.
SEEN_TOL = 1e-3
# Distance bin edges for the continuous generalization-gap panel.
DIST_EDGES = np.array([0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])

ARM_STYLE = {
    "fixed": {"color": "#D85A30", "label": "Fixed steps"},
    "early": {"color": "#1B2A4A", "label": "Early stopping"},
}


def load_runs(root):
    runs = {}
    for path in sorted(Path(root).glob("budget_N*/eval_per_r.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        arm = "fixed" if params.get("max_steps") is not None else "early"
        with np.load(path) as result:
            r = result["r_grid"].copy()
            train_r = np.sort(result["train_r"].copy())
            ce = result["ce_per_r"].copy()
        # distance from each eval r to the nearest r the model trained on
        idx = np.searchsorted(train_r, r).clip(1, len(train_r) - 1)
        dist = np.minimum(np.abs(r - train_r[idx - 1]), np.abs(r - train_r[idx]))
        runs.setdefault(arm, {})[int(params["n_train_traj_actual"])] = {
            "r": r, "ce": ce, "dist": dist, "train_r": train_r,
        }
    if not runs:
        raise FileNotFoundError(f"no budget runs under {root}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_budget")
    ap.add_argument("--out_dir", default="figures_budget")
    a = ap.parse_args()

    runs = load_runs(a.runs_dir)
    arms = [x for x in ("fixed", "early") if x in runs]
    budgets = sorted(runs[arms[0]])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # -- panel A: loss vs task count, seen vs new -------------------------
    ax = axes[0]
    for arm in arms:
        style = ARM_STYLE[arm]
        seen, new, n_seen, n_new = [], [], [], []
        for n in budgets:
            run = runs[arm][n]
            m_seen = run["dist"] <= SEEN_TOL
            seen.append(run["ce"][m_seen].mean() if m_seen.any() else np.nan)
            new.append(run["ce"][~m_seen].mean() if (~m_seen).any() else np.nan)
            n_seen.append(int(m_seen.sum()))
            n_new.append(int((~m_seen).sum()))
        ax.plot(budgets, seen, "o-", color=style["color"], lw=1.8,
                label=f"{style['label']} — seen tasks")
        ax.plot(budgets, new, "s--", color=style["color"], lw=1.4, alpha=0.7,
                mfc="white", label=f"{style['label']} — new tasks")
        if arm == arms[0]:
            for n, ns, nn in zip(budgets, n_seen, n_new):
                ax.annotate(f"{ns}/{nn}", (n, seen[budgets.index(n)]),
                            textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=6, color="#666666")

    ax.axhline(np.log(64), color="gray", ls=":", lw=1.2)
    ax.text(budgets[0], np.log(64) * 1.05, "uniform over 64 bins",
            fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of training tasks $N$ (distinct $r$ values)")
    ax.set_ylabel("Mean cross-entropy (nats)")
    ax.set_title("Loss vs task count, seen vs new tasks", fontsize=11)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    # -- panel B: the generalization gap, resolved continuously -----------
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    centers = np.sqrt(DIST_EDGES[:-1] * np.maximum(DIST_EDGES[1:], 1e-12))
    centers[0] = DIST_EDGES[1] / 2
    for i, n in enumerate(budgets):
        run = runs["fixed"][n]
        means = []
        for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
            sel = (run["dist"] > lo) & (run["dist"] <= hi)
            means.append(run["ce"][sel].mean() if sel.sum() >= 3 else np.nan)
        ax.plot(centers, means, "o-", lw=1.5, ms=4,
                color=cmap(i / max(1, len(budgets) - 1)), label=f"N={n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Distance from eval $r$ to nearest training $r$")
    ax.set_ylabel("Mean cross-entropy (nats)")
    ax.set_title("Generalization gap vs distance to training data\n"
                 "(fixed-step arm)", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    fig.suptitle("Task diversity and generalization across the logistic family"
                 "   —   CAVEAT: these runs use 1 trajectory per $r$, so $N$ "
                 "varies task count and total data together", fontsize=10.5)
    fig.tight_layout()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"task_diversity.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}/task_diversity.png")


if __name__ == "__main__":
    main()
