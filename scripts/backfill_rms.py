"""Add implied-map RMS to sweeps trained before the metric existed.

runs_div_controlled is the input=64 / output=64-bin arm of the tokenisation
sweep, so it is reused rather than retrained -- but it predates rms_per_r, and
RMS is the only metric comparable across the binned-output and square-loss
halves of that sweep. The checkpoints are all present, so this recomputes it
from them instead of spending 7 training jobs.

The evaluation grid, the seen-r sample and the seeds are read back out of each
run's own npz, so the numbers land on exactly the grid the CE numbers used.
evaluate_per_r draws its initial conditions sequentially along the grid, so
passing the full grid with the original seed reproduces the original stream
exactly -- which is checked here by recomputing CE and comparing it to what was
stored. If that matches, the RMS is on the same footing as the CE it joins.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_div_controlled")
    ap.add_argument("--n_eval_per_r", type=int, default=30)
    a = ap.parse_args()

    import torch
    from src.model import DiscreteTrajectoryTransformer
    from src.evaluation import evaluate_per_r

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        npz_path = os.path.join(d, "eval_per_r.npz")
        par_path = os.path.join(d, "params.json")
        if not (os.path.exists(npz_path) and os.path.exists(par_path)):
            continue
        z = dict(np.load(npz_path))
        if "rms_per_r" in z:
            print(f"[backfill] {os.path.basename(d)}: already has rms, skipping")
            continue
        p = json.load(open(par_path))
        nb, L = p["n_bins"], p["context_len"]
        model = DiscreteTrajectoryTransformer(
            n_bins=nb, context_len=L, d_model=p["d_model"],
            n_heads=p["n_heads"], n_layers=p["n_layers"],
            dropout=p["dropout"]).to(dev)

        for tag, ckpt_name, suffix in (("final", "best_final.pt", ""),
                                       ("bestval", "best_bestval.pt", "_bestval")):
            ck = os.path.join(d, ckpt_name)
            if not os.path.exists(ck):
                print(f"[backfill] {os.path.basename(d)}: no {ckpt_name}")
                continue
            model.load_state_dict(torch.load(
                ck, map_location=dev, weights_only=True)["model_state_dict"])
            kw = dict(model=model, device=dev, context_len=L, n_bins=nb,
                      burn_in=p.get("burn_in", 0), n_eval_per_r=a.n_eval_per_r,
                      traj_len=p["traj_len"], return_rms=True)
            ce, _, rms = evaluate_per_r(r_grid=z["r_grid"],
                                        seed=p["seed"] + 7, **kw)
            _, _, rms_seen = evaluate_per_r(r_grid=z["seen_r"],
                                            seed=p["seed"] + 13, **kw)
            stored = z.get("ce_per_r" + suffix)
            if stored is not None:
                dev_max = float(np.nanmax(np.abs(stored - ce)))
                flag = "OK" if dev_max < 1e-6 else f"MISMATCH {dev_max:.2e}"
                print(f"[backfill] {os.path.basename(d)} [{tag}] "
                      f"CE reproduction: {flag}", flush=True)
            z[f"rms_per_r{suffix}"] = rms
            z[f"rms_at_train_r{suffix}"] = rms_seen
            print(f"[backfill] {os.path.basename(d)} [{tag}] "
                  f"new-grid RMS={np.nanmean(rms):.5f} "
                  f"seen-r RMS={np.nanmean(rms_seen):.5f}", flush=True)
        np.savez(npz_path, **z)
    print("[backfill] done")


if __name__ == "__main__":
    main()
