"""
N x L ablation sweep: how does next-token cross-entropy depend on bin count N
(symbolic vs metric resolution) and context length L (the Markov / temporal-
locality hypothesis)?

Hypothesis: the tokenized process is closer to Markov at coarse N, so CE should
be roughly flat in L at small N and fan out (longer context helps) at large N.

This module is the driver: it generates per-run configs, trains each model,
evaluates CE as a function of r (the bifurcation parameter), and plots the
result as small multiples (one panel per N, one curve per L). The trained
checkpoints are saved so downstream evals (cross-family, dissociation,
mech-interp) can reuse them without retraining.

torch is imported lazily inside the train/eval functions so the config-generation
and plotting paths import without it.
"""
import os
import json

import numpy as np

from .maps import iterate_map, tokenize_trajectory

# The reduced grid we settled on.
DEFAULT_NS = (2, 16, 64)
DEFAULT_LS = (2, 8, 32, 64)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def make_sweep_configs(base_cfg, ns=DEFAULT_NS, ls=DEFAULT_LS):
    """Yield (run_name, cfg) for each (N, L) cell, derived from a base config.

    base_cfg is the parsed base.yaml dict; we override n_bins and context_len
    and give each run a descriptive name. Everything else (model width/depth,
    optimizer, data size) is held fixed so N and L are the only variables.
    """
    import copy
    runs = []
    for n in ns:
        for l in ls:
            cfg = copy.deepcopy(base_cfg)
            cfg["data"]["n_bins"] = int(n)
            cfg["data"]["context_len"] = int(l)
            run_name = f"sweep_N{n}_L{l}"
            cfg["run_name"] = run_name
            runs.append((run_name, cfg))
    return runs


# ---------------------------------------------------------------------------
# CE as a function of r  (the primary figure's data)
# ---------------------------------------------------------------------------

def ce_vs_r(model, device, n_bins, context_len, r_grid=None,
            n_eval_per_r=20, traj_len=150, burn_in=50, seed=99):
    """Mean next-token cross-entropy per r for a trained model.

    Returns (r_grid, ce_per_r). Uses a burn-in so trajectories are on-attractor
    (consistent with how lambda is computed). No Lyapunov grid is needed: the
    figure's x-axis is r, not lambda.
    """
    import torch
    import torch.nn as nn

    if r_grid is None:
        r_grid = np.linspace(0.5, 4.0, 200)
    model.eval()
    rng = np.random.default_rng(seed)
    crit = nn.CrossEntropyLoss()
    ce = np.empty(len(r_grid))

    with torch.no_grad():
        for i, r in enumerate(r_grid):
            contexts, targets = [], []
            for _ in range(n_eval_per_r):
                x0 = rng.uniform(0.05, 0.95)
                traj = iterate_map(x0, r, burn_in + traj_len)[burn_in:]
                tok = tokenize_trajectory(traj, n_bins)
                for t in range(len(tok) - context_len - 1):
                    contexts.append(tok[t:t + context_len])
                    targets.append(tok[t + context_len])
            ctx = torch.as_tensor(np.array(contexts), dtype=torch.long, device=device)
            tgt = torch.as_tensor(np.array(targets), dtype=torch.long, device=device)
            ce[i] = float(crit(model(ctx), tgt).item())
            if (i + 1) % 50 == 0:
                print(f"    ce_vs_r {i+1}/{len(r_grid)}")
    return r_grid, ce


def save_ce_result(out_dir, run_name, n_bins, context_len, r_grid, ce):
    """Append one run's CE(r) to a JSON results file (incremental, disconnect-safe)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sweep_ce_results.json")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results[run_name] = {
        "n_bins": int(n_bins), "context_len": int(context_len),
        "r_grid": list(map(float, r_grid)), "ce": list(map(float, ce)),
    }
    with open(path, "w") as f:
        json.dump(results, f)
    return path


# ---------------------------------------------------------------------------
# Plotting: small multiples, one panel per N, one curve per L
# ---------------------------------------------------------------------------

def plot_sweep(results, ns=DEFAULT_NS, ls=DEFAULT_LS, save_path=None, figsize=None):
    """results: dict run_name -> {n_bins, context_len, r_grid, ce} (from JSON).

    One panel per N (own y-scale, raw CE), one colored curve per L.
    """
    import matplotlib.pyplot as plt
    ns = [n for n in ns if any(v["n_bins"] == n for v in results.values())]
    figsize = figsize or (4.3 * len(ns), 4.0)
    fig, axes = plt.subplots(1, len(ns), figsize=figsize)
    if len(ns) == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis")
    l_color = {l: cmap(i / max(1, len(ls) - 1)) for i, l in enumerate(ls)}

    for ax, n in zip(axes, ns):
        for l in ls:
            key = f"sweep_N{n}_L{l}"
            if key not in results:
                continue
            d = results[key]
            ax.plot(d["r_grid"], d["ce"], color=l_color[l], lw=1.2,
                    alpha=0.85, label=f"L={l}")
        ax.set_title(f"N = {n}")
        ax.set_xlabel(r"$r$")
        ax.set_ylabel("Cross-entropy (nats)")
        ax.legend(fontsize=8, title="context", loc="upper left")
    fig.suptitle("CE vs $r$ across bin count $N$ and context length $L$", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig