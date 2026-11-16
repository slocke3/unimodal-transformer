"""Does the model use history beyond the last observation? (reviewer FWzs)

FWzs proposed contexts from different maps whose final states are nearly
identical but whose correct next states differ (the next state depends on r).
A model that predicts from the last token alone must give such contexts the
same prediction; a model that identifies the map from the history should move
its prediction with the truth.

For each trained run this draws fresh logistic trajectories at r ~ U(0.5, 4)
(new tasks, not the training r's) with one context window per trajectory,
tokenized exactly as in training, and records the model's predicted next state
(the scalar output, or the softmax-weighted bin centre for cross-entropy
models). Two measurements:

  matched pairs   Within each final input bin, contexts are sorted by their
                  continuous final state and consecutive ones are paired when
                  those states differ by < 1/(4 N_in) and the true next states
                  differ by more than one output bin (1/64). Regressing the
                  predicted difference on the true difference across pairs
                  gives a slope: 0 = last-token-only, 1 = fully tracks r.
  within-bin R^2  1 - SSE(model) / SSE(best last-token-only predictor), where
                  the latter is each final bin's mean true next state on these
                  same contexts (an oracle for that strategy). Positive values
                  mean the model beats every predictor that sees only the last
                  token.

Results are also split by regime: r < 3 (fixed point), 3-3.57 (periodic),
> 3.57 (chaotic band, including its periodic windows). Needs torch and the run's
checkpoint (<out_dir>/best_best.pt, the final weights in fixed-step mode):

  python scripts/rebuttal/probe_matched_final.py runs_ctx_short/L*_m8000_seed0 \
      runs_probe/* --out runs_rebuttal_probe/matched_final.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.maps import tokenize_trajectory  # noqa: E402
from src.model import DiscreteTrajectoryTransformer  # noqa: E402

REGIMES = {"all": (0.5, 4.0), "fixed": (0.5, 3.0), "periodic": (3.0, 3.5699),
           "chaotic": (3.5699, 4.0)}


def load_model(run, device):
    p = json.load(open(os.path.join(run, "params.json")))
    n_in = p["n_bins_in"] or p["n_bins"]
    n_out = p["n_bins_out"] or p["n_bins"]
    model = DiscreteTrajectoryTransformer(
        n_bins=p["n_bins"], context_len=p["context_len"], d_model=p["d_model"],
        n_heads=p["n_heads"], n_layers=p["n_layers"], dropout=p["dropout"],
        n_bins_in=n_in * p.get("synonyms", 1), n_bins_out=n_out,
        output_mode=p.get("output_mode", "bins"))
    ck = torch.load(os.path.join(run, "best_best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), p, n_in, n_out


def contexts(L, traj_len, n, seed):
    """One window per trajectory at a uniformly random start, as in training."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 4.0, n)
    x = rng.uniform(0.05, 0.95, n)
    traj = np.empty((n, traj_len + 1))
    traj[:, 0] = x
    for i in range(traj_len):
        x = r * x * (1.0 - x)
        traj[:, i + 1] = x
    t = rng.integers(0, traj_len + 1 - (L + 1), n)
    idx = t[:, None] + np.arange(L + 1)[None, :]
    win = np.take_along_axis(traj, idx, axis=1)
    return win[:, :L], win[:, L], r


@torch.no_grad()
def predict(model, tok, n_out, out_mode, device, bs=8192):
    centres = torch.tensor((np.arange(n_out) + 0.5) / n_out, dtype=torch.float32,
                           device=device)
    out = []
    for i in range(0, len(tok), bs):
        x = torch.tensor(tok[i:i + bs], dtype=torch.long, device=device)
        y = model(x)
        out.append((y if out_mode == "scalar" else torch.softmax(y, -1) @ centres).cpu())
    return torch.cat(out).numpy().astype(float)


def measures(last_tok, x_last, y, yhat, n_in, sel):
    lt, xl, yy, yh = last_tok[sel], x_last[sel], y[sel], yhat[sel]
    # within-bin R^2 against the oracle last-token-only predictor
    means = np.bincount(lt, weights=yy, minlength=n_in) / np.maximum(
        np.bincount(lt, minlength=n_in), 1)
    sse_tok = np.sum((yy - means[lt]) ** 2)
    r2 = 1 - np.sum((yy - yh) ** 2) / sse_tok if sse_tok > 0 else np.nan
    # matched pairs
    order = np.lexsort((xl, lt))
    a, b = order[:-1], order[1:]
    keep = ((lt[a] == lt[b]) & (np.abs(xl[a] - xl[b]) < 0.25 / n_in)
            & (np.abs(yy[a] - yy[b]) > 1 / 64))
    dy, dyh = (yy[a] - yy[b])[keep], (yh[a] - yh[b])[keep]
    slope = np.sum(dy * dyh) / np.sum(dy ** 2) if keep.sum() > 10 else np.nan
    corr = np.corrcoef(dy, dyh)[0, 1] if keep.sum() > 10 else np.nan
    return r2, slope, corr, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=400_000, help="contexts per run")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="runs_rebuttal_probe/matched_final.csv")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"   # MPS gave wrong outputs (Nov 2026)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for run in a.runs:
        if not os.path.exists(os.path.join(run, "best_best.pt")):
            print(f"[skip] {run}: no checkpoint")
            continue
        model, p, n_in, n_out = load_model(run, device)
        L = p["context_len"]
        ctx, y, r = contexts(L, p.get("traj_len", 150), a.n, a.seed)
        tok = tokenize_trajectory(ctx, n_in)
        yhat = predict(model, tok, n_out, p.get("output_mode", "bins"), device)
        for name, (lo, hi) in REGIMES.items():
            sel = (r >= lo) & (r < hi)
            r2, slope, corr, npairs = measures(tok[:, -1], ctx[:, -1], y, yhat, n_in, sel)
            rows.append(dict(run=run, L=L, m=p["m"], n_bins_in=n_in,
                             output=p.get("output_mode", "bins"), seed=p["seed"],
                             d_model=p["d_model"], n_layers=p["n_layers"], regime=name,
                             within_bin_r2=r2, pair_slope=slope, pair_corr=corr,
                             n_pairs=npairs, mse=float(np.mean((yhat[sel] - y[sel]) ** 2))))
            print(f"{run:45s} {name:8s} R2_within={r2:+.3f} slope={slope:.3f} "
                  f"corr={corr:.3f} pairs={npairs}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
