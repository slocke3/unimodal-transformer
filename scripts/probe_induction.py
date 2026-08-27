"""Does the model implement the task-INDEPENDENT counting estimator?

The optimal in-context predictor for any next-token process is: find earlier
occurrences of the current token, and predict what followed them. That routine
is what induction heads implement, and crucially it does not depend on which
map generated the sequence -- it is the analogue of OLS in Goddard et al., the
kind of solution that can generalize out of task distribution.

If the model uses it, the probability mass it puts on the in-context
continuation set should be high AND roughly equal in band and out of band,
because the routine is task-agnostic. If instead it applies a learned family
prior, that mass should collapse off-band along with everything else.

Reference: a context-only counting predictor computed directly from the same
context, which is what a pure induction head would give.
"""
import argparse, json
from pathlib import Path
import numpy as np

from src.maps import iterate_family, tokenize_trajectory


def contexts(R, p, family, n_bins, L, n_traj, traj_len, seed):
    rng = np.random.default_rng(seed)
    ctx, tgt = [], []
    for _ in range(n_traj):
        t = tokenize_trajectory(
            iterate_family(rng.uniform(.05, .95), R, p, traj_len, family), n_bins)
        for s in range(len(t) - L - 1):
            ctx.append(t[s:s + L]); tgt.append(t[s + L])
    return np.asarray(ctx), np.asarray(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs_tilt/asym_w0.1_seed0", "runs_tilt/asym_w0.7_seed0"])
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--n_traj", type=int, default=40)
    a = ap.parse_args()

    import torch
    from src.model import DiscreteTrajectoryTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for run in a.runs:
        P = json.load(open(Path(run) / "params.json"))
        fam, nb, L, w = P["family"], P["n_bins"], P["context_len"], P["band_width"]
        m = DiscreteTrajectoryTransformer(
            n_bins=nb, context_len=L, d_model=P["d_model"], n_heads=P["n_heads"],
            n_layers=P["n_layers"], dropout=P["dropout"]).to(dev)
        m.load_state_dict(torch.load(Path(run) / "best_final.pt", map_location=dev,
                                     weights_only=True)["model_state_dict"])
        m.eval()
        print(f"\n=== {run}  band=[{-w:g},{w:g}] ===")
        print("%9s %7s | %14s %14s %13s %12s"
              % ("cell", "s", "P(model->cont)", "P(count->cont)", "chance",
                 "n w/ match"))
        for tag, sv in (("in band", 0.0), ("in band", round(w/2, 3)),
                        ("held out", round(w+0.2, 3)), ("held out", round(-w-0.2, 3))):
            ctx, tgt = contexts(a.R, sv, fam, nb, L, a.n_traj, P["traj_len"],
                                seed=int(abs(sv)*131)+5)
            with torch.no_grad():
                pr = torch.softmax(
                    m(torch.as_tensor(ctx, dtype=torch.long, device=dev)),
                    dim=-1).cpu().numpy()
            mm, cc, nmatch = [], [], 0
            for i in range(len(ctx)):
                cur = ctx[i, -1]
                j = np.where(ctx[i, :-1] == cur)[0]
                if len(j) == 0:
                    continue
                nmatch += 1
                cont = ctx[i, j + 1]                      # what followed, in context
                sup = np.unique(cont)
                mm.append(pr[i, sup].sum())               # model's mass on that set
                cnt = np.bincount(cont, minlength=nb).astype(float)
                cc.append((cnt / cnt.sum())[tgt[i]])      # counting predictor's p(truth)
            print("%9s %7.2f | %14.3f %14.3f %13.3f %12d"
                  % (tag, sv, np.mean(mm), np.mean(cc), len(np.unique(ctx)) and 1.0/nb,
                     nmatch))


if __name__ == "__main__":
    main()
