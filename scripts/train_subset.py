"""
Train ONE transformer on a chosen subset of r-values and evaluate it across the
whole family. Designed for a SLURM job array: the scheduler invokes this script
once per job with a different (m, a, start, ...) tuple; each job trains one model
and writes checkpoint + per-r eval to disk.

Two experiment modes, both via --placement:

  * sliding window (experiment 1): --placement uniform_r --start A --width W --m M
        trains on M r-values sampled uniformly in [start, start+width];
        eval across the full range shows in-window vs complement (extrapolation).

  * placement strategy (experiment 2): --placement {uniform_r,uniform_lambda,
        chaotic,bifurcation} over the full range, fixed budget M
        -> "can dynamical invariants guide sample-efficient training?"

CONFOUND CONTROL: total training data is held fixed via --n_train_traj (a total
trajectory budget split evenly over the M r-values), so comparisons across M / a
/ placement vary *where* the data is, not *how much*.

Outputs (in --out_dir):
  params.json     the full run spec + resolved training r-values
  best.pt         best-val checkpoint
  history.json    train/val loss curves
  eval_per_r.npz  r_grid, ce_per_r, acc_per_r, in_window mask  (the key result)

Example (one job):
  python scripts/train_subset.py --placement uniform_r --start 0.5 --width 0.4 \
      --m 8 --context_len 50 --n_bins 64 --n_train_traj 8000 \
      --seed 0 --out_dir runs/w0.4_s0.5_m8
"""
import argparse
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Placement strategies: given a budget m, return the training r-values
# ---------------------------------------------------------------------------

def _lambda_grid(r_lo, r_hi, n=1500, n_steps=20000):
    from src.maps import compute_lyapunov
    rg = np.linspace(r_lo, r_hi, n)
    lam = np.array([compute_lyapunov(r, n_steps=n_steps) for r in rg])
    return rg, lam


def place_r(placement, m, start, width, seed=0, full_range=(0.5, 4.0)):
    """Return m training r-values for the given placement strategy."""
    rng = np.random.default_rng(seed)
    lo, hi = start, start + width
    if placement == "uniform_r":
        # evenly spaced in r within the window (edges included)
        return np.linspace(lo, hi, m)
    if placement == "uniform_r_random":
        return np.sort(rng.uniform(lo, hi, size=m))

    # --- full-range invariant-guided strategies (experiment 2) ---
    r_lo, r_hi = full_range
    if placement == "uniform_lambda":
        rg, lam = _lambda_grid(r_lo, r_hi)
        # place m points at equal quantiles of lambda (monotone-ish resample)
        order = np.argsort(lam)
        q = np.linspace(0, 1, m)
        idx = order[np.round(q * (len(order) - 1)).astype(int)]
        return np.sort(rg[idx])
    if placement == "chaotic":
        # all budget in the chaotic band [3.57, 4.0]
        return np.linspace(3.5699, r_hi, m)
    if placement == "bifurcation":
        # density concentrated near the period-doubling accumulation r_inf
        r_inf = 3.5699
        # geometric spacing toward r_inf from both sides, plus chaotic band
        left = r_inf - np.geomspace(1e-3, r_inf - r_lo, m // 2)
        right = np.linspace(r_inf, r_hi, m - m // 2)
        return np.sort(np.concatenate([left, right]))
    raise ValueError(f"unknown placement '{placement}'")


# ---------------------------------------------------------------------------
# Main: build data from the r-set, train, eval across the full range
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # what to train on
    p.add_argument("--placement", default="uniform_r")
    p.add_argument("--start", type=float, default=0.5)
    p.add_argument("--width", type=float, default=0.4)
    p.add_argument("--m", type=int, default=8, help="number of training r-values")
    p.add_argument("--full_lo", type=float, default=0.5)
    p.add_argument("--full_hi", type=float, default=4.0)
    # data / tokenization (also sweepable axes)
    p.add_argument("--context_len", type=int, default=50)
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--traj_len", type=int, default=150)
    p.add_argument("--burn_in", type=int, default=0)
    p.add_argument("--n_train_traj", type=int, default=8000,
                   help="TOTAL training trajectories, split evenly over the m r-values")
    p.add_argument("--val_frac", type=float, default=0.15)
    # model / training
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    # eval
    p.add_argument("--r_eval_points", type=int, default=300)
    p.add_argument("--n_eval_per_r", type=int, default=30)
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # -- resolve the training r-set (fixed total data, split over m) ---------
    train_r = np.asarray(place_r(args.placement, args.m, args.start, args.width,
                                 seed=args.seed, full_range=(args.full_lo, args.full_hi)),
                         dtype=float)
    n_per_r = max(1, args.n_train_traj // len(train_r))
    train_r_lo, train_r_hi = float(train_r.min()), float(train_r.max())

    # torch imports after arg parsing (fast --help on the login node)
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import DiscreteMapDataset
    from src.model import DiscreteTrajectoryTransformer
    from src.trainer import Trainer, TrainerConfig
    from src.evaluation import evaluate_per_r

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # expand: n_per_r trajectories per training r; split into train/val by trajectory
    rng = np.random.default_rng(args.seed)
    all_r = np.repeat(train_r, n_per_r)
    rng.shuffle(all_r)
    n_val = int(args.val_frac * len(all_r))
    val_r, tr_r = all_r[:n_val], all_r[n_val:]

    def loader(rvals, shuffle, seed):
        ds = DiscreteMapDataset(r_values=rvals, context_len=args.context_len,
                                burn_in=args.burn_in, traj_len=args.traj_len,
                                n_bins=args.n_bins, seed=seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = loader(tr_r, True, args.seed)
    val_loader = loader(val_r, False, args.seed + 1)

    model = DiscreteTrajectoryTransformer(
        n_bins=args.n_bins, context_len=args.context_len, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)

    tcfg = TrainerConfig(lr=args.lr, weight_decay=args.weight_decay,
                         max_epochs=args.max_epochs, patience=args.patience,
                         save_dir=args.out_dir)
    trainer = Trainer(model, train_loader, val_loader, config=tcfg, run_name="best")
    print(f"[train_subset] placement={args.placement} m={args.m} "
          f"window=[{train_r_lo:.3f},{train_r_hi:.3f}] n_per_r={n_per_r} device={device}", flush=True)
    history = trainer.train()
    trainer.load_best()

    # -- eval across the WHOLE family (in-window + complement) ---------------
    r_grid = np.linspace(args.full_lo, args.full_hi, args.r_eval_points)
    ce_per_r, acc_per_r = evaluate_per_r(
        model=model, r_grid=r_grid, device=device, context_len=args.context_len,
        n_bins=args.n_bins, burn_in=args.burn_in, n_eval_per_r=args.n_eval_per_r,
        traj_len=args.traj_len, seed=args.seed + 7)

    # in-window mask (for the sliding-window / extrapolation analysis)
    in_window = (r_grid >= train_r_lo - 1e-9) & (r_grid <= train_r_hi + 1e-9)

    np.savez(os.path.join(args.out_dir, "eval_per_r.npz"),
             r_grid=r_grid, ce_per_r=ce_per_r, acc_per_r=acc_per_r,
             in_window=in_window, train_r=train_r)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump({**vars(args), "train_r": train_r.tolist(), "n_per_r": n_per_r,
                   "train_r_lo": train_r_lo, "train_r_hi": train_r_hi,
                   "n_params": model.count_parameters(),
                   "wall_sec": round(time.time() - t0, 1)}, f, indent=2)

    oo = ~in_window
    print(f"[train_subset] DONE in {time.time()-t0:.0f}s | "
          f"mean CE in-window={ce_per_r[in_window].mean():.3f} "
          f"complement={ce_per_r[oo].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
