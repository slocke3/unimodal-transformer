"""Is in-context identification limited by the CONTEXT LENGTH?

On the tilted family the models sit 6-8x above the ideal binned predictor even
*inside* their training band, where the asymmetric-family models sat essentially
at it. One explanation is information, not generalization: identifying the map
means pinning down a function on [0,1] from L=50 observed transitions over a
64-bin alphabet -- fewer observations than alphabet symbols. If that is the
binding constraint then no model can approach 1.0, and a flat held-out curve
says nothing about the transition.

This truncates the context of ALREADY TRAINED models and re-reads the implied
return map. No retraining.

Positional handling matters and is easy to get wrong. The model reads its
prediction off the LAST position, which during training was always position 49.
Feeding a length-L sequence naively uses positions 0..L-1, so the readout lands
at position L-1 -- one the model never used for prediction, which confounds
"less information" with "unfamiliar readout position". The primary measurement
therefore keeps the original absolute positions (50-L .. 49); the naive variant
is computed alongside as a control, and a gap between them is a positional
artifact rather than an information effect.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.maps import iterate_family, family_map_vec, tokenize_trajectory, detokenize
from src.binfloor import ideal_bin_reference


def contexts_for(R, p, family, n_bins, context_len, n_traj, traj_len, seed):
    rng = np.random.default_rng(seed)
    ctx = []
    for _ in range(n_traj):
        x0 = rng.uniform(0.05, 0.95)
        t = tokenize_trajectory(iterate_family(x0, R, p, traj_len, family), n_bins)
        for s in range(len(t) - context_len - 1):
            ctx.append(t[s:s + context_len])
    return np.asarray(ctx)


def implied(model, ctx, L, n_bins, device, keep_positions=True):
    """Implied E[x_{n+1}] using only the last L tokens of each context."""
    import torch
    sub = ctx[:, -L:]
    x = torch.as_tensor(sub, dtype=torch.long, device=device)
    with torch.no_grad():
        if keep_positions:
            # keep the ORIGINAL absolute positions so the readout stays at the
            # index the model was trained to predict from
            full = ctx.shape[1]
            pos = torch.arange(full - L, full, device=device)
            h = model.token_embed(x) + model.pos_embedding.embedding(pos).unsqueeze(0)
            h = model.transformer(h, mask=model.causal_mask[:L, :L], is_causal=True)
            logits = model.output_head(h[:, -1, :])
        else:
            logits = model(x)
        pr = torch.softmax(logits, dim=-1).cpu().numpy()
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return pr @ centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs_tilt/asym_w0.1_seed0",
                             "runs_tilt/asym_w0.4_seed0",
                             "runs_tilt/asym_w0.7_seed0"])
    ap.add_argument("--Ls", type=int, nargs="+",
                    default=[3, 5, 8, 12, 18, 25, 35, 50])
    ap.add_argument("--R", type=float, nargs="+", default=[0.7, 1.0])
    ap.add_argument("--n_traj", type=int, default=25)
    ap.add_argument("--out_dir", default="figures_tilt_prelim")
    a = ap.parse_args()

    import torch
    from src.model import DiscreteTrajectoryTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    for run in a.runs:
        P = json.load(open(Path(run) / "params.json"))
        fam, nb, CL = P["family"], P["n_bins"], P["context_len"]
        w = P["band_width"]
        model = DiscreteTrajectoryTransformer(
            n_bins=nb, context_len=CL, d_model=P["d_model"],
            n_heads=P["n_heads"], n_layers=P["n_layers"],
            dropout=P["dropout"]).to(device)
        ck = torch.load(Path(run) / "best_final.pt", map_location=device,
                        weights_only=True)
        model.load_state_dict(ck["model_state_dict"]); model.eval()

        cells = [("in band", 0.0), ("in band", round(w / 2, 3)),
                 ("held out", round(w + 0.2, 3)), ("held out", round(-w - 0.2, 3))]
        print(f"\n=== {run}  family={fam} band=[{-w:g},{w:g}] ===")
        print("%9s %8s %6s " % ("cell", "s", "R") +
              "".join("%8d" % L for L in a.Ls))
        rows = {}
        for tag, sval in cells:
            for R in a.R:
                ref = float(ideal_bin_reference([sval], [R], nb, fam,
                                                n_steps=15000)[0, 0])
                ctx = contexts_for(R, sval, fam, nb, CL, a.n_traj,
                                   P["traj_len"], seed=int(abs(sval) * 97) + 3)
                last_x = detokenize(ctx[:, -1], nb)
                truth = family_map_vec(last_x, R, sval, fam)
                ratios, naive = [], []
                for L in a.Ls:
                    e = implied(model, ctx, L, nb, device, True)
                    ratios.append(float(np.sqrt(np.mean((e - truth) ** 2)) / ref))
                    e2 = implied(model, ctx, L, nb, device, False)
                    naive.append(float(np.sqrt(np.mean((e2 - truth) ** 2)) / ref))
                rows[(tag, sval, R)] = (ratios, naive)
                print("%9s %8.2f %6.2f " % (tag, sval, R) +
                      "".join("%8.1f" % v for v in ratios))
        out[run] = {"w": w, "rows": rows}

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, len(out), figsize=(5.4 * len(out), 4.7),
                             squeeze=False, sharey=True)
    for k, (run, res) in enumerate(out.items()):
        ax = axes[0][k]
        for (tag, sval, R), (ratios, naive) in res["rows"].items():
            c = "#1B2A4A" if tag == "in band" else "#D85A30"
            ax.plot(a.Ls, ratios, "-o", ms=4, lw=1.6, color=c, alpha=0.85,
                    label=f"{tag} s={sval:g} R={R:g}")
            ax.plot(a.Ls, naive, ":", lw=1.0, color=c, alpha=0.4)
        ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Context length $L$ used at test time")
        if k == 0:
            ax.set_ylabel("Implied-map RMS / ideal binned predictor")
        ax.set_title(f"w={res['w']:g}   (trained at $L$=50)", fontsize=10.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.legend(fontsize=6.5)
    fig.suptitle("Is identification context-limited?  Solid: original absolute "
                 "positions kept (readout at 49).  Dotted: naive truncation "
                 "(readout at $L-1$) — a control for positional artifacts.",
                 fontsize=11)
    fig.tight_layout()
    o = Path(a.out_dir); o.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(o / f"context_length_probe.{ext}", dpi=160,
                    bbox_inches="tight")
    print(f"\nwrote {o}/context_length_probe.png")


if __name__ == "__main__":
    main()
