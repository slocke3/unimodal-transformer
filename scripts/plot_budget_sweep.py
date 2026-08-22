"""Plot the data-budget sweep: fixed-step vs early-stopping training protocols.

Two figures:

  budget_ce_vs_r      per-r cross-entropy for every budget N, one panel per arm,
                      plus mean CE vs N with both arms on the same axes.
  budget_overfitting  train/val curves against gradient steps, one panel per N.
                      The fixed arm runs the same step budget at every N, so the
                      val curve turning up at small N *is* the overfitting the
                      sweep is measuring; the early arm's stop point is marked
                      on the same step axis for comparison.

Torch-free: reads eval_per_r.npz / history.json / params.json only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import numpy as np

# Period-doubling accumulation point: r above this is the chaotic band.
R_INF = 3.5699

ARM_STYLE = {
    "fixed": {"color": "#D85A30", "label": "Fixed steps (final model)"},
    "early": {"color": "#1B2A4A", "label": "Early stopping (best-val model)"},
}


def steps_per_epoch(params):
    """Optimizer steps per epoch, from the run's own recorded configuration."""
    n_traj = params["n_train_traj_actual"]
    n_train = n_traj - int(params["val_frac"] * n_traj)
    n_examples = n_train * (params["traj_len"] - params["context_len"])
    return max(1, -(-n_examples // params["batch_size"]))


def load_runs(root):
    """Load budget runs as {arm: {N: run}}."""
    runs = {}
    for path in sorted(Path(root).glob("budget_N*/eval_per_r.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        with open(path.parent / "history.json") as handle:
            history = json.load(handle)
        arm = "fixed" if params.get("max_steps") is not None else "early"
        with np.load(path) as result:
            run = {
                "name": path.parent.name,
                "n": int(params["n_train_traj_actual"]),
                "params": params,
                "history": history,
                "spe": steps_per_epoch(params),
                "r": result["r_grid"].copy(),
                "ce": result["ce_per_r"].copy(),
                "acc": result["acc_per_r"].copy(),
            }
        runs.setdefault(arm, {})[run["n"]] = run
    if not runs:
        raise FileNotFoundError(f"no budget runs found under {root}")
    return runs


def budget_colors(budgets):
    norm = LogNorm(vmin=min(budgets), vmax=max(budgets))
    cmap = plt.get_cmap("viridis")
    return {n: cmap(norm(n)) for n in budgets}, ScalarMappable(norm=norm, cmap=cmap)


# ---------------------------------------------------------------------------
# Figure 1: CE across r, per arm, and the summary curve
# ---------------------------------------------------------------------------

def plot_ce_vs_r(runs, save_stem):
    arms = [a for a in ("fixed", "early") if a in runs]
    budgets = sorted({n for arm in arms for n in runs[arm]})
    colors, _ = budget_colors(budgets)

    fig, axes = plt.subplots(1, len(arms) + 1,
                             figsize=(5.2 * (len(arms) + 1), 4.2))
    axes = np.atleast_1d(axes)

    for ax, arm in zip(axes, arms):
        for n in budgets:
            run = runs[arm].get(n)
            if run is None:
                continue
            ax.plot(run["r"], run["ce"], lw=1.1, color=colors[n], label=f"N={n}")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)
        ax.axvspan(R_INF, 4.0, color="#F44336", alpha=0.06, lw=0)
        ax.set_xlabel(r"Parameter $r$")
        ax.set_ylabel("Cross-entropy (nats)")
        ax.set_title(ARM_STYLE[arm]["label"], fontsize=11)
        ax.legend(fontsize=7, ncol=2)

    ax = axes[-1]
    for arm in arms:
        ns = sorted(runs[arm])
        style = ARM_STYLE[arm]
        means = [runs[arm][n]["ce"].mean() for n in ns]
        # The median is ~1e-6: most r are periodic and predicted essentially
        # perfectly, so the full-range mean is carried by the chaotic band.
        # Report that band separately rather than a median that says nothing.
        chaotic = [runs[arm][n]["ce"][runs[arm][n]["r"] >= R_INF].mean()
                   for n in ns]
        ax.plot(ns, means, "o-", color=style["color"], lw=1.7,
                label=f"{style['label']} — full range")
        ax.plot(ns, chaotic, "s--", color=style["color"], lw=1.3, alpha=0.7,
                label=f"{style['label']} — chaotic band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training budget $N$ (trajectories = distinct $r$)")
    ax.set_ylabel("Mean cross-entropy (nats)")
    ax.set_title("How much data to learn the family?", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    fig.suptitle("Data-budget sweep: cross-entropy across the logistic family",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{save_stem}.{ext}", dpi=160, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: the overfitting curves (why the protocol matters)
# ---------------------------------------------------------------------------

def plot_overfitting(runs, save_stem):
    budgets = sorted({n for arm in runs for n in runs[arm]}, reverse=True)
    ncols = min(3, len(budgets))
    nrows = -(-len(budgets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows),
                             squeeze=False)

    for ax, n in zip(axes.ravel(), budgets):
        fixed = runs.get("fixed", {}).get(n)
        early = runs.get("early", {}).get(n)

        if fixed is not None:
            hist = fixed["history"]
            total = hist.get("total_steps", fixed["params"]["max_steps"])
            n_logged = len(hist["train_losses"])
            interval = max(1, total // max(1, fixed["params"]["log_points"]))
            steps = np.arange(1, n_logged + 1) * interval
            ax.plot(steps, hist["train_losses"], lw=1.4,
                    color=ARM_STYLE["fixed"]["color"], label="fixed: train")
            ax.plot(steps, hist["val_losses"], lw=1.4, ls="--",
                    color=ARM_STYLE["fixed"]["color"], label="fixed: val")
            best = int(np.argmin(hist["val_losses"]))
            ax.plot(steps[best], hist["val_losses"][best], "o", ms=5,
                    color=ARM_STYLE["fixed"]["color"], mfc="white",
                    label="fixed: val minimum")

        if early is not None:
            hist = early["history"]
            steps = np.arange(1, len(hist["train_losses"]) + 1) * early["spe"]
            ax.plot(steps, hist["train_losses"], lw=1.2, alpha=0.8,
                    color=ARM_STYLE["early"]["color"], label="early: train")
            val_steps = np.arange(1, len(hist["val_losses"]) + 1) * early["spe"]
            ax.plot(val_steps, hist["val_losses"], lw=1.2, ls="--", alpha=0.8,
                    color=ARM_STYLE["early"]["color"], label="early: val")
            ax.axvline(hist["best_epoch"] * early["spe"], lw=1.0, ls=":",
                       color=ARM_STYLE["early"]["color"],
                       label=f"early stop (epoch {hist['best_epoch']})")

        spe = (fixed or early)["spe"]
        ax.set_yscale("log")
        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("Cross-entropy (nats)")
        ax.set_title(f"N={n}  ({spe} steps/epoch)", fontsize=11)
        ax.legend(fontsize=6.5)

    for ax in axes.ravel()[len(budgets):]:
        ax.set_visible(False)

    fig.suptitle("Fixed-step training overfits small budgets; early stopping "
                 "reports the best point instead", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{save_stem}.{ext}", dpi=160, bbox_inches="tight")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_budget")
    ap.add_argument("--out_dir", default="figures_budget")
    a = ap.parse_args()

    runs = load_runs(a.runs_dir)
    for arm in sorted(runs):
        print(f"{arm}: N={sorted(runs[arm])}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_ce_vs_r(runs, str(out / "budget_ce_vs_r"))
    plot_overfitting(runs, str(out / "budget_overfitting"))
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
