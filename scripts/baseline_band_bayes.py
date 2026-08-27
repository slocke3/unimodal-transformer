"""Reference predictors for the held-out region, analogous to Figure 3 of
Goddard et al. (arXiv:2506.05574).

That figure plots far-OOD test loss against task diversity and compares the
transformer against the *optimal in-task-distribution Bayesian solution* — the
best predictor that knows only the pretraining task distribution. The
transformer beating it is the evidence that out-of-distribution generalization
is happening even below the transition.

Three references are computed here, all model-free, all on the same held-out
tasks the transformer is scored on:

  band_bayes    Bayes over an order-1 (Markov) model class restricted to the
                TRAINING BAND. It infers which map it is looking at from the
                context, but its prior lives entirely inside alpha in [1-w, 1],
                so on a held-out alpha it must explain the data with a map it
                wrongly believes is in-band. This is the analogue of the paper's
                dashed "Optimal Bayes*" line, and the one to beat.
  context_only  order-1 transition matrix estimated from the context window
                itself (49 transitions), smoothed. No prior over tasks at all —
                what pure in-context Markov estimation buys you.
  oracle        order-1 transition matrix of the TRUE map. Best achievable
                within the Markov-1 class; a floor, not a competitor.

Order-1 is a good model class here: the maps are deterministic, so the only
uncertainty in the next bin comes from within-bin spread, and the transition
matrices are near-deterministic curves. The transformer can beat this class
only by using longer history to pin the parameters down more sharply, which is
exactly the ability under test.

Smoothing is tuned per predictor to minimize its own loss, so each reference is
given its best shot rather than being handicapped.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.maps import iterate_asym, tokenize_trajectory


def transition_matrix(R, alpha, n_bins, n_steps=120_000, burn_in=2_000, seed=5):
    x = np.random.default_rng(seed).uniform(0.2, 0.8)
    traj = iterate_asym(x, R, alpha, burn_in + n_steps)[burn_in:]
    t = tokenize_trajectory(traj, n_bins)
    C = np.bincount(t[:-1] * n_bins + t[1:],
                    minlength=n_bins * n_bins).astype(float)
    return C.reshape(n_bins, n_bins)


def normalize(counts, eps):
    P = counts + eps
    return P / P.sum(axis=1, keepdims=True)


def make_contexts(R, alpha, n_bins, context_len, n_traj, traj_len, seed):
    rng = np.random.default_rng(seed)
    ctx, tgt = [], []
    for _ in range(n_traj):
        x0 = rng.uniform(0.05, 0.95)
        t = tokenize_trajectory(iterate_asym(x0, R, alpha, traj_len), n_bins)
        for s in range(len(t) - context_len - 1):
            ctx.append(t[s:s + context_len])
            tgt.append(t[s + context_len])
    return np.asarray(ctx), np.asarray(tgt)


def ce_from_probs(p_next, targets):
    """Mean cross-entropy in nats."""
    return float(-np.mean(np.log(np.maximum(
        p_next[np.arange(len(targets)), targets], 1e-300))))


def band_bayes_ce(ctx, tgt, cand_logT, cand_T, chunk=4000):
    """Posterior over candidate maps from context transitions, then mix."""
    n_cand = len(cand_T)
    out = np.empty(len(ctx))
    for s in range(0, len(ctx), chunk):
        c = ctx[s:s + chunk]
        # log-likelihood of each context under each candidate's order-1 model
        ll = cand_logT[:, c[:, :-1], c[:, 1:]].sum(axis=2)      # (n_cand, m)
        ll -= ll.max(axis=0, keepdims=True)
        wgt = np.exp(ll)
        wgt /= wgt.sum(axis=0, keepdims=True)
        # predictive: sum_c w_c * T_c[last_token, :]
        rows = cand_T[:, c[:, -1], :]                            # (n_cand,m,B)
        p = np.einsum("cm,cmb->mb", wgt, rows)
        out[s:s + chunk] = -np.log(np.maximum(
            p[np.arange(len(c)), tgt[s:s + chunk]], 1e-300))
    return float(out.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band_widths", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                             0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    ap.add_argument("--heldout_alphas", type=float, nargs="+",
                    default=[0.20, 0.22, 0.24, 0.26, 0.28, 0.30])
    ap.add_argument("--R_eval", type=float, nargs="+",
                    default=[0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
    ap.add_argument("--n_cand_alpha", type=int, default=17)
    ap.add_argument("--n_cand_R", type=int, default=17)
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--context_len", type=int, default=50)
    ap.add_argument("--traj_len", type=int, default=150)
    ap.add_argument("--n_traj", type=int, default=6)
    ap.add_argument("--out", default="figures_asym2/band_bayes_baseline.npz")
    a = ap.parse_args()

    B = a.n_bins
    eps_grid = np.array([3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])

    # ---- held-out evaluation contexts, shared by every predictor ----------
    print(f"building held-out contexts: alpha in {a.heldout_alphas}, "
          f"R in {a.R_eval}")
    cells = []
    for al in a.heldout_alphas:
        for R in a.R_eval:
            ctx, tgt = make_contexts(R, al, B, a.context_len, a.n_traj,
                                     a.traj_len, seed=int(1000 * al + R * 7))
            cells.append({"alpha": al, "R": R, "ctx": ctx, "tgt": tgt,
                          "counts": transition_matrix(R, al, B)})
    n_ctx = sum(len(c["ctx"]) for c in cells)
    print(f"  {len(cells)} cells, {n_ctx} contexts total")

    # ---- oracle and context-only, independent of band width --------------
    def oracle_ce(eps):
        return float(np.mean([
            ce_from_probs(normalize(c["counts"], eps)[c["ctx"][:, -1]], c["tgt"])
            for c in cells]))
    eps_o = eps_grid[np.argmin([oracle_ce(e) for e in eps_grid])]
    oracle = oracle_ce(eps_o)

    def context_only_ce(eps):
        tot = []
        for c in cells:
            ctx, tgt = c["ctx"], c["tgt"]
            idx = ctx[:, :-1] * B + ctx[:, 1:]
            ce = np.empty(len(ctx))
            for i in range(len(ctx)):
                C = np.bincount(idx[i], minlength=B * B).reshape(B, B)
                P = normalize(C, eps)
                ce[i] = -np.log(max(P[ctx[i, -1], tgt[i]], 1e-300))
            tot.append(ce.mean())
        return float(np.mean(tot))
    eps_c = eps_grid[np.argmin([context_only_ce(e) for e in eps_grid])]
    context_only = context_only_ce(eps_c)

    print(f"  oracle Markov-1   CE = {oracle:.4f}  (eps={eps_o:g})")
    print(f"  context-only      CE = {context_only:.4f}  (eps={eps_c:g})")

    # ---- band-restricted Bayes, one value per band width -----------------
    print("\n%7s %10s %14s" % ("w", "band", "band_bayes CE"))
    results, tuned = [], {}
    for w in a.band_widths:
        alphas = np.linspace(1.0 - w, 1.0, a.n_cand_alpha) if w > 0 else np.array([1.0])
        Rs = np.linspace(0.125, 1.0, a.n_cand_R)
        counts = np.stack([transition_matrix(R, al, B)
                           for al in alphas for R in Rs])
        # Tune the smoothing once (on the first band) and reuse it: the
        # optimum is a property of the model class and the held-out data, not
        # of the band, and re-tuning per band costs a factor of len(eps_grid).
        nonlocal_eps = tuned.get("eps")
        trial = eps_grid if nonlocal_eps is None else [nonlocal_eps]
        best, best_eps = np.inf, None
        for eps in trial:
            T = normalize(counts, eps)
            ce = float(np.mean([band_bayes_ce(c["ctx"], c["tgt"], np.log(T), T)
                                for c in cells]))
            if ce < best:
                best, best_eps = ce, eps
        tuned.setdefault("eps", best_eps)
        results.append(best)
        print("%7g %10s %14.4f   (eps=%g)"
              % (w, f"[{1-w:.2f},1]", best, best_eps))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, band_widths=np.array(a.band_widths),
             band_bayes=np.array(results), oracle=oracle,
             context_only=context_only,
             heldout_alphas=np.array(a.heldout_alphas),
             R_eval=np.array(a.R_eval))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
