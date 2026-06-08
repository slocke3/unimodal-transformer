"""
Evaluate a trained model: per-r cross-entropy vs Lyapunov, trajectory rollouts,
and cross-family generalization.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/base_best.pt --config outputs/checkpoints/base_config.yaml
"""
import argparse
import os

import numpy as np
import torch
import yaml

from src.dataset import make_eval_grid
from src.model import DiscreteTrajectoryTransformer
from src.evaluation import (
    evaluate_per_r, evaluate_general,
    plot_ce_vs_lyapunov, plot_trajectory_comparison,
)
from src.maps import FAMILIES, compute_lyapunov_general


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", nargs="+", default=["quadratic", "tent", "sine", "cubic"],
                        help="Map families to evaluate generalization on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    e = cfg["evaluation"]
    p = cfg["paths"]
    run_name = cfg.get("run_name", "run")

    os.makedirs(p["figure_dir"], exist_ok=True)
    os.makedirs(p["cache_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    m = cfg["model"]
    model = DiscreteTrajectoryTransformer(
        n_bins=d["n_bins"], context_len=d["context_len"],
        d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"],
        d_ff=m.get("d_ff"), dropout=m["dropout"],
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.4f})")

    # --- Lyapunov grid ---
    lya_cache = os.path.join(p["cache_dir"], "lyapunov_grid.npz")
    if os.path.exists(lya_cache):
        print("Loading cached Lyapunov grid...")
        cache = np.load(lya_cache)
        r_grid, lyapunovs = cache["r_grid"], cache["lyapunovs"]
    else:
        r_grid, lyapunovs = make_eval_grid(
            n_points=e["n_eval_points"],
            r_range=tuple(d["r_range"]),
            n_lyapunov_steps=e["n_lyapunov_steps"],
        )
        np.savez(lya_cache, r_grid=r_grid, lyapunovs=lyapunovs)

    # --- Per-r CE evaluation ---
    ce_cache = os.path.join(p["cache_dir"], f"{run_name}_ce_per_r.npz")
    if os.path.exists(ce_cache):
        print("Loading cached per-r evaluation...")
        cache = np.load(ce_cache)
        ce_per_r, acc_per_r = cache["ce_per_r"], cache["acc_per_r"]
    else:
        print("Evaluating per r...")
        ce_per_r, acc_per_r = evaluate_per_r(
            model=model, r_grid=r_grid, device=device,
            context_len=d["context_len"], n_bins=d["n_bins"],
            burn_in=d["burn_in"], n_eval_per_r=e["n_eval_per_r"],
            traj_len=d["traj_len"],
        )
        np.savez(ce_cache, ce_per_r=ce_per_r, acc_per_r=acc_per_r)

    fig = plot_ce_vs_lyapunov(
        r_grid=r_grid, lyapunovs=lyapunovs, ce_per_r=ce_per_r,
        save_path=os.path.join(p["figure_dir"], f"{run_name}_ce_vs_lyapunov.png"),
    )

    # --- Cross-family generalization ---
    for name in args.families:
        if name not in FAMILIES:
            print(f"Unknown family '{name}', skipping.")
            continue
        fam = FAMILIES[name]
        ce_cache_f = os.path.join(p["cache_dir"], f"{run_name}_ce_{name}.npz")
        if os.path.exists(ce_cache_f):
            print(f"Loading cached: {name}")
            cache = np.load(ce_cache_f)
            ce_f, acc_f = cache["ce"], cache["acc"]
        else:
            print(f"Evaluating family: {name}")
            ce_f, acc_f = evaluate_general(
                model=model, map_fn=fam["map_fn"], params=fam["params"],
                device=device, context_len=d["context_len"], n_bins=d["n_bins"],
                burn_in=d["burn_in"], traj_len=d["traj_len"],
            )
            np.savez(ce_cache_f, ce=ce_f, acc=acc_f)
        print(f"  {name}: mean CE = {ce_f.mean():.4f}")

    # --- Trajectory rollout plot ---
    r_values_to_plot = [2.5, 3.2, 3.5, 3.7, 3.83, 3.95]
    plot_trajectory_comparison(
        model=model, r_values=r_values_to_plot, device=device,
        context_len=d["context_len"], n_bins=d["n_bins"],
        rollout_steps=100, burn_in=d["burn_in"],
        save_path=os.path.join(p["figure_dir"], f"{run_name}_trajectories.png"),
    )
    print("Done.")


if __name__ == "__main__":
    main()