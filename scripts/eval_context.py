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
    import torch.nn as nn
    from src.model import (DiscreteTrajectoryTransformer,
                           ContinuousTrajectoryTransformer)
    from src.maps import iterate_map, tokenize_trajectory

    def eval_all_contexts(model, r_grid, p, modes, seed, contexts):
        """Every context on one pass of the data.

        The first version of this called evaluate_per_r once per (model,
        context) pair, which regenerated the whole trajectory set each time --
        twenty contexts meant twenty identical generations, and the job timed
        out. The data does not depend on the context, so it is built once per r
        and every context is then just another forward pass over it.
        """
        L = p["context_len"]
        n_in = modes["n_bins_in"]
        rng = np.random.default_rng(seed)
        mse = {c: np.empty(len(r_grid)) for c in contexts}
        for i, r in enumerate(r_grid):
            rc, rt = [], []
            for _ in range(a.n_eval_per_r):
                traj = iterate_map(rng.uniform(0.05, 0.95),
                                   r, p.get("burn_in", 0) + p["traj_len"])
                for t in range(len(traj) - (L + 1)):
                    rc.append(traj[t:t + L]); rt.append(traj[t + L])
            rc = np.asarray(rc, dtype=np.float64)
            truth = torch.tensor(np.asarray(rt), dtype=torch.float32, device=dev)
            if modes["input_mode"] == "continuous":
                ctx = torch.tensor(rc, dtype=torch.float32, device=dev)
            else:
                tok = tokenize_trajectory(rc, n_in)
                if modes["synonyms"] > 1:
                    tok = tok * modes["synonyms"] + rng.integers(
                        0, modes["synonyms"], size=tok.shape)
                ctx = torch.tensor(tok, dtype=torch.long, device=dev)
            with torch.no_grad():
                for c in contexts:
                    pred = model(ctx, attend_last=c)
                    mse[c][i] = ((pred - truth) ** 2).mean().item()
        return mse

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
        assert modes["output_mode"] == "scalar", \
            "this path reports MSE directly; binned output would need the " \
            "softmax-mean implied map instead"
        tag = os.path.basename(d)
        new = eval_all_contexts(model, z["r_grid"], p, modes, p["seed"] + 7,
                                a.contexts)
        seen = eval_all_contexts(model, z["seen_r"], p, modes, p["seed"] + 13,
                                 a.contexts)
        for L in a.contexts:
            out[f"{tag}_L{L}_mse_new"] = new[L]
            out[f"{tag}_L{L}_mse_seen"] = seen[L]
            # also under the old names, as per-r RMS, so existing readers keep
            # working; they must square BEFORE averaging over r, not after
            out[f"{tag}_L{L}_rms_new"] = np.sqrt(new[L])
            out[f"{tag}_L{L}_rms_seen"] = np.sqrt(seen[L])
            print(f"  {tag:28s} L={L:3d}  new MSE {np.nanmean(new[L]):.3e}  "
                  f"seen MSE {np.nanmean(seen[L]):.3e}", flush=True)
        out[f"{tag}_m"] = np.array(p["m"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **out)
    print(f"[eval_context] wrote {a.out}")


if __name__ == "__main__":
    main()
