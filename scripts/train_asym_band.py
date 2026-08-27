"""
Train ONE transformer on a band of the asymmetric unimodal family, then
evaluate it across the whole (alpha, R) plane.

The family is

    g(x) = R (x/x_c)^alpha ((1-x)/(1-x_c))^beta,   beta = 2 - alpha,
                                                   x_c  = alpha/2

with R the peak height. R = r/4 and alpha = 1 recover the logistic map exactly,
so alpha is a pure "move the peak" knob holding the critical point quadratic.

Training band: alpha ~ U[1 - band_width, 1], R ~ U[R_lo, R_hi]. A band width of
0 trains on the quadratic family alone, which is the zero-shot arm: train on
logistic, test on asymmetric.

Why alpha <= 1: near the origin g ~ x^alpha, so g'(0) = 0 for alpha > 1 and the
origin becomes superattracting. Above alpha = 1 a growing share of orbits die
there (at alpha=1.4, R=1, all of them), and a dead orbit is a constant token
sequence that is trivially predictable. Including alpha > 1 would therefore make
out-of-band loss look better the more degenerate the task became. The evaluation
still records a dead fraction per grid point so any such contamination is visible.

Outputs (in --out_dir):
  params.json         run spec, including the resolved band
  best_final.pt / best_bestval.pt   final and best-val checkpoints
  history.json        loss curves
  eval_asym.npz       CE/accuracy/dead-fraction over the (alpha, R) grid, for
                      both checkpoints, plus the in-band mask
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    # training band
    p.add_argument("--family", choices=["asym", "tilted"], default="asym",
                   help="asym: g(x)=R(x/x_c)^a((1-x)/(1-x_c))^(2-a), band is "
                        "one-sided [1-w, 1] since a>1 is degenerate. "
                        "tilted: f(x)=C x(1-x)(1+s(x-1/2)), s=0 IS logistic and "
                        "both endpoints stay regular, so the band is SYMMETRIC "
                        "[-w, +w] and has held-out regions on both sides.")
    p.add_argument("--band_width", type=float, default=0.0,
                   help="asym: train on alpha ~ U[1-w, 1]. tilted: s ~ U[-w, +w]")
    p.add_argument("--heldout_d", type=float, default=0.2,
                   help="fixed DISTANCE beyond the band edge at which the "
                        "primary held-out probes sit. Holding the distance "
                        "fixed (rather than the position) stops a widening "
                        "band from making its own test easier.")
    p.add_argument("--R_lo", type=float, default=0.125,
                   help="peak-height range (0.125 = logistic r=0.5)")
    p.add_argument("--R_hi", type=float, default=1.0,
                   help="1.0 = logistic r=4")
    p.add_argument("--n_tasks", type=int, default=8000,
                   help="distinct (R, alpha) pairs drawn for training")
    p.add_argument("--n_train_traj", type=int, default=32000,
                   help="total trajectories, split evenly over the tasks")
    # data / tokenization
    p.add_argument("--context_len", type=int, default=50)
    p.add_argument("--n_bins", type=int, default=64)
    p.add_argument("--traj_len", type=int, default=150)
    p.add_argument("--burn_in", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--max_val_traj", type=int, default=600)
    # model / training
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=160000)
    p.add_argument("--log_points", type=int, default=80)
    # eval grid over the (alpha, R) plane
    p.add_argument("--alpha_eval_lo", type=float, default=0.5)
    p.add_argument("--alpha_eval_hi", type=float, default=1.0)
    p.add_argument("--n_alpha_eval", type=int, default=26)
    p.add_argument("--p_eval_lo", type=float, default=None,
                   help="tilted: lowest s in the eval grid (default -p_eval_hi)")
    p.add_argument("--p_eval_hi", type=float, default=1.2)
    p.add_argument("--n_R_eval", type=int, default=100)
    p.add_argument("--n_eval_per_point", type=int, default=30)
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    if args.family == "asym":
        # alpha > 1 makes the origin superattracting and below ~0.15 orbits
        # collapse, so [0.2, 1.0] is the usable span and the band is one-sided.
        if not 0.0 <= args.band_width <= 0.8:
            raise ValueError("asym: band_width must be in [0, 0.8]")
        band_lo, band_hi = 1.0 - args.band_width, 1.0
    else:
        # tilted: both endpoints stay regular, so the band is symmetric about
        # the logistic map at s=0. f'(0) crosses 1 near s=1.45; stay well inside.
        if not 0.0 <= args.band_width <= 1.0:
            raise ValueError("tilted: band_width must be in [0, 1.0]")
        band_lo, band_hi = -args.band_width, args.band_width

    # -- draw the training tasks --------------------------------------------
    rng = np.random.default_rng(args.seed)
    alpha_lo = band_lo
    if args.band_width == 0.0:
        alphas = np.full(args.n_tasks, band_hi)
    else:
        alphas = rng.uniform(band_lo, band_hi, size=args.n_tasks)
    Rs = rng.uniform(args.R_lo, args.R_hi, size=args.n_tasks)
    tasks = np.column_stack([Rs, alphas])

    n_per_task, n_extra = divmod(args.n_train_traj, args.n_tasks)
    if n_per_task < 1:
        raise ValueError(f"n_train_traj ({args.n_train_traj}) < n_tasks "
                         f"({args.n_tasks})")
    counts = np.full(args.n_tasks, n_per_task, dtype=int)
    if n_extra:
        counts[rng.permutation(args.n_tasks)[:n_extra]] += 1
    all_params = np.repeat(tasks, counts, axis=0)
    rng.shuffle(all_params)

    n_val = int(args.val_frac * len(all_params))
    if args.max_val_traj > 0:
        n_val = min(n_val, args.max_val_traj)
    val_params, train_params = all_params[:n_val], all_params[n_val:]

    # torch imports after arg parsing (fast --help on the login node)
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import AsymMapDataset
    from src.model import DiscreteTrajectoryTransformer
    from src.trainer import Trainer, TrainerConfig
    from src.evaluation import evaluate_asym_grid

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def loader(params, shuffle, seed):
        ds = AsymMapDataset(params, context_len=args.context_len,
                            burn_in=args.burn_in, traj_len=args.traj_len,
                            n_bins=args.n_bins, seed=seed,
                            family=args.family)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)

    train_loader = loader(train_params, True, args.seed)
    val_loader = loader(val_params, False, args.seed + 1)
    train_token_hist = train_loader.dataset.token_counts.copy()

    model = DiscreteTrajectoryTransformer(
        n_bins=args.n_bins, context_len=args.context_len, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)

    tcfg = TrainerConfig(lr=args.lr, weight_decay=args.weight_decay,
                         max_steps=args.max_steps, log_points=args.log_points,
                         save_dir=args.out_dir)
    trainer = Trainer(model, train_loader, val_loader, config=tcfg,
                      run_name="best")
    print(f"[train_asym] band=[{band_lo:.3f}, {band_hi:.3f}] "
          f"R in [{args.R_lo:.3f}, {args.R_hi:.3f}] tasks={args.n_tasks} "
          f"traj={len(all_params)} steps={args.max_steps} device={device}",
          flush=True)
    history = trainer.train()
    trainer.load_best()

    # -- evaluate over the whole (alpha, R) plane ----------------------------
    if args.family == "asym":
        alpha_grid = np.linspace(args.alpha_eval_lo, args.alpha_eval_hi,
                                 args.n_alpha_eval)
    else:
        lo = args.p_eval_lo if args.p_eval_lo is not None else -args.p_eval_hi
        alpha_grid = np.linspace(lo, args.p_eval_hi, args.n_alpha_eval)
    # the fixed-distance probes must land exactly on the grid
    probes = [p for p in (band_lo - args.heldout_d, band_hi + args.heldout_d)
              if alpha_grid.min() - 1e-9 <= p <= alpha_grid.max() + 1e-9]
    heldout_mask = np.array([any(abs(g - p) < 1e-6 for p in probes)
                             for g in alpha_grid])
    print(f"[train_asym] family={args.family} band=[{band_lo:.3f},{band_hi:.3f}] "
          f"fixed-distance probes at {[round(p,3) for p in probes]} "
          f"(on grid: {heldout_mask.sum()})", flush=True)
    R_grid = np.linspace(args.R_lo, args.R_hi, args.n_R_eval)

    def run_eval(tag):
        ce, acc, dead, imp, flr = evaluate_asym_grid(
            model=model, alpha_grid=alpha_grid, R_grid=R_grid, device=device,
            context_len=args.context_len, n_bins=args.n_bins,
            burn_in=args.burn_in, n_eval_per_point=args.n_eval_per_point,
            traj_len=args.traj_len, seed=args.seed + 7,
            family=args.family, return_implied=True)
        in_band = ((alpha_grid >= band_lo - 1e-9)
                   & (alpha_grid <= band_hi + 1e-9))[:, None] & np.ones_like(ce, bool)
        print(f"[train_asym] eval[{tag}] in-band CE={ce[in_band].mean():.4f} "
              f"out-of-band CE="
              f"{ce[~in_band].mean() if (~in_band).any() else float('nan'):.4f} "
              f"max dead-frac={dead.max():.2f}", flush=True)
        return ce, acc, dead, imp, flr

    ce_final, acc_final, dead, imp_final, floor_grid = run_eval("final")

    bestval_path = os.path.join(args.out_dir, "best_bestval.pt")
    extra = {}
    if os.path.exists(bestval_path):
        ckpt = torch.load(bestval_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        ce_bv, acc_bv, _, imp_bv, _ = run_eval("bestval")
        extra = {"ce_bestval": ce_bv, "acc_bestval": acc_bv,
                 "implied_rms_bestval": imp_bv}

    # in_band is a property of alpha alone; store the 1-D mask
    in_band_alpha = (alpha_grid >= band_lo - 1e-9) & \
                    (alpha_grid <= band_hi + 1e-9)

    np.savez(os.path.join(args.out_dir, "eval_asym.npz"),
             alpha_grid=alpha_grid, R_grid=R_grid,
             ce_final=ce_final, acc_final=acc_final, dead_frac=dead,
             in_band_alpha=in_band_alpha, band_width=args.band_width,
             alpha_lo=band_lo, band_lo=band_lo, band_hi=band_hi,
             family=args.family, heldout_mask=heldout_mask,
             heldout_d=args.heldout_d, probes=np.array(probes),
             implied_rms=imp_final, binning_floor=floor_grid,
             train_tasks=tasks,
             train_token_hist=train_token_hist, **extra)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump({**vars(args), "alpha_lo": alpha_lo,
                   "n_train_traj_actual": len(all_params),
                   "n_params": model.count_parameters(),
                   "wall_sec": round(time.time() - t0, 1)}, f, indent=2)

    ib = in_band_alpha
    print(f"[train_asym] DONE in {time.time()-t0:.0f}s | "
          f"in-band={ce_final[ib].mean():.4f} "
          f"out={ce_final[~ib].mean() if (~ib).any() else float('nan'):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
