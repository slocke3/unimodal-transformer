"""Train ONE continuous-input transformer on the full logistic family and
evaluate it across r. The input is the raw value (sinusoidal embedding, no
input binning); the output is a softmax over --n_out bins. Sweeping --n_out
across jobs answers "which output discretization is best?" on a common footing
(reference-grid CE + implied-map RMS; see src/continuous_eval.py).

Designed for the same SLURM job array as train_subset.py -- one job per
(n_out, seed). Torch is imported after arg parsing so --help is fast.

Example (one job):
  python scripts/train_continuous.py --n_out 64 --n_train_traj 8000 \
      --context_len 50 --seed 0 --out_dir runs_cont/cont_nout64_seed0
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    # output resolution (the swept axis) and continuous-input knobs
    p.add_argument("--n_out", type=int, default=64, help="output softmax bins")
    p.add_argument("--n_freqs", type=int, default=64, help="sinusoidal input frequencies")
    p.add_argument("--max_freq", type=float, default=256.0,
                   help="max input frequency = finest resolvable x-scale")
    p.add_argument("--ref_bins", type=int, default=256,
                   help="reference grid for comparable CE (multiple of every n_out)")
    # family / data
    p.add_argument("--full_lo", type=float, default=0.5)
    p.add_argument("--full_hi", type=float, default=4.0)
    p.add_argument("--context_len", type=int, default=50)
    p.add_argument("--traj_len", type=int, default=150)
    p.add_argument("--burn_in", type=int, default=0)
    p.add_argument("--n_train_traj", type=int, default=8000)
    p.add_argument("--val_frac", type=float, default=0.15)
    # model / training (matches the discrete baseline)
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
    # eval / bookkeeping
    p.add_argument("--r_eval_points", type=int, default=300)
    p.add_argument("--n_eval_per_r", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    import torch
    from torch.utils.data import DataLoader
    from src.continuous_dataset import ContinuousMapDataset
    from src.continuous_model import ContinuousInputTransformer
    from src.continuous_eval import evaluate_continuous_per_r
    from src.trainer import Trainer, TrainerConfig

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # full-range uniform-random r, one trajectory each; split by trajectory
    rng = np.random.default_rng(args.seed)
    rs = rng.uniform(args.full_lo, args.full_hi, size=args.n_train_traj)
    rng.shuffle(rs)
    n_val = int(args.val_frac * len(rs))
    val_r, tr_r = rs[:n_val], rs[n_val:]

    def loader(rvals, shuffle, seed):
        ds = ContinuousMapDataset(r_values=rvals, n_out=args.n_out,
                                  context_len=args.context_len, burn_in=args.burn_in,
                                  traj_len=args.traj_len, seed=seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = loader(tr_r, True, args.seed)
    val_loader = loader(val_r, False, args.seed + 1)

    model = ContinuousInputTransformer(
        n_out=args.n_out, context_len=args.context_len, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        n_freqs=args.n_freqs, max_freq=args.max_freq)

    tcfg = TrainerConfig(lr=args.lr, weight_decay=args.weight_decay,
                         max_epochs=args.max_epochs, patience=args.patience,
                         save_dir=args.out_dir)
    trainer = Trainer(model, train_loader, val_loader, config=tcfg, run_name="best")
    print(f"[train_continuous] n_out={args.n_out} n_freqs={args.n_freqs} "
          f"max_freq={args.max_freq} n_train={len(tr_r)} device={device}", flush=True)
    history = trainer.train()
    trainer.load_best()

    r_grid = np.linspace(args.full_lo, args.full_hi, args.r_eval_points)
    ce_ref, ce_native, rms = evaluate_continuous_per_r(
        model, r_grid, device, args.context_len, args.n_out,
        ref_bins=args.ref_bins, burn_in=args.burn_in,
        n_eval_per_r=args.n_eval_per_r, traj_len=args.traj_len, seed=args.seed + 7)

    np.savez(os.path.join(args.out_dir, "eval_continuous.npz"),
             r_grid=r_grid, ce_ref_per_r=ce_ref, ce_native_per_r=ce_native,
             rms_per_r=rms, n_out=args.n_out, ref_bins=args.ref_bins)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump({**vars(args), "n_params": model.count_parameters(),
                   "n_train_traj_actual": len(tr_r),
                   "wall_sec": round(time.time() - t0, 1)}, f, indent=2)

    print(f"[train_continuous] DONE {time.time()-t0:.0f}s | n_out={args.n_out} | "
          f"mean CE_ref={np.nanmean(ce_ref):.3f} | mean RMS={np.nanmean(rms):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
