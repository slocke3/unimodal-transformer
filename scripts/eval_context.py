"""Re-evaluate trained models at shorter effective context, without retraining.

Method (i) of the context-length question. Rather than truncating the input --
which would move the prediction from position L-1 of the trained length to a
position these models never predicted at, since they are trained on the final
position only with learned positional embeddings -- attention is masked so that
the query stays where it was trained and only the older keys disappear. The mask
is verified to block at every layer: scrambling the hidden positions leaves the
output bit-identical.

This measures how much of its context a model trained at the full length actually
uses. It is NOT the same question as how well a model trained at that shorter
length would do, which needs its own runs; comparing the two is the point.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories, or globs, to re-evaluate")
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 20, 30, 40, 50])
    ap.add_argument("--n_eval_per_r", type=int, default=30)
    ap.add_argument("--ckpt", default="best_final.pt")
    ap.add_argument("--out", default="figures_taskdiv/eval_context.npz")
    a = ap.parse_args()

    import torch
    from src.model import (DiscreteTrajectoryTransformer,
                           ContinuousTrajectoryTransformer)
    from src.evaluation import evaluate_per_r

    dirs = sorted({d for pat in a.runs for d in glob.glob(pat)})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"contexts": np.array(a.contexts)}
    print(f"[eval_context] {len(dirs)} runs x {len(a.contexts)} contexts on {dev}",
          flush=True)

    for d in dirs:
        p = json.load(open(os.path.join(d, "params.json")))
        z = np.load(os.path.join(d, "eval_per_r.npz"))
        n_in = p.get("n_bins_in") or p["n_bins"]
        n_out = p.get("n_bins_out") or p["n_bins"]
        modes = dict(input_mode=p.get("input_mode", "bins"), n_bins_in=n_in,
                     n_bins_out=n_out, output_mode=p.get("output_mode", "bins"),
                     synonyms=p.get("synonyms", 1),
                     input_noise=p.get("input_noise", 0.0))
        if modes["input_mode"] == "continuous":
            model = ContinuousTrajectoryTransformer(
                n_bins=n_out, context_len=p["context_len"], d_model=p["d_model"],
                n_heads=p["n_heads"], n_layers=p["n_layers"],
                dropout=p["dropout"], output_mode=modes["output_mode"]).to(dev)
        else:
            model = DiscreteTrajectoryTransformer(
                n_bins=p["n_bins"], context_len=p["context_len"],
                d_model=p["d_model"], n_heads=p["n_heads"],
                n_layers=p["n_layers"], dropout=p["dropout"],
                n_bins_in=n_in * modes["synonyms"], n_bins_out=n_out,
                output_mode=modes["output_mode"]).to(dev)
        model.load_state_dict(torch.load(os.path.join(d, a.ckpt),
                                         map_location=dev,
                                         weights_only=True)["model_state_dict"])
        tag = os.path.basename(d)
        for L in a.contexts:
            kw = dict(model=model, device=dev, context_len=p["context_len"],
                      n_bins=p["n_bins"], burn_in=p.get("burn_in", 0),
                      n_eval_per_r=a.n_eval_per_r, traj_len=p["traj_len"],
                      return_rms=True, attend_last=L, **modes)
            ce, _, rms = evaluate_per_r(r_grid=z["r_grid"], seed=p["seed"] + 7, **kw)
            ce_s, _, rms_s = evaluate_per_r(r_grid=z["seen_r"],
                                            seed=p["seed"] + 13, **kw)
            for k, v in (("ce_new", ce), ("rms_new", rms),
                         ("ce_seen", ce_s), ("rms_seen", rms_s)):
                out[f"{tag}_L{L}_{k}"] = v
            print(f"  {tag:28s} L={L:3d}  new RMS {np.nanmean(rms):.6f}  "
                  f"seen RMS {np.nanmean(rms_s):.6f}", flush=True)
        out[f"{tag}_m"] = np.array(p["m"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **out)
    print(f"[eval_context] wrote {a.out}")


if __name__ == "__main__":
    main()
