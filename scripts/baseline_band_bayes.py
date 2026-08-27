"""Reference predictors for the held-out region, analogous to Figure 3 of
Goddard et al. (arXiv:2506.05574).

That figure plots far-OOD test loss against task diversity and compares the
transformer against the *optimal in-task-distribution Bayesian solution* — the
best predictor that knows only the pretraining task distribution. The
transformer beating it is the evidence that out-of-distribution generalization
is happening even below the transition.

Three families of reference, all model-free, all scored on the same held-out
tasks, each at Markov orders k = 1..4:

  band_bayes    Bayes over order-k Markov models restricted to the TRAINING
                BAND. It infers which map it is looking at from the context,
                but its prior lives entirely inside alpha in [1-w, 1], so on a
                held-out alpha it must explain the data with a map it wrongly
                believes is in-band. Analogue of the paper's "Optimal Bayes*".
  context_only  order-k model estimated from the context window itself. No task
                prior at all. Degrades with k as the k-grams in 50 tokens go
                unique — the sparsity cost of memory without a prior.
  oracle        order-k model of the TRUE map, fit on a long orbit. Best
                achievable in the class; a floor, not a competitor.

Why orders 1-4 are cheap despite 64^4 = 1.7e7 possible 4-grams: the maps are
deterministic and one-dimensional, so a long orbit visits only ~944 distinct
4-grams. Models are stored as sorted integer keys plus count rows and queried
with searchsorted, which keeps the whole sweep to a few minutes.

Candidates are fitted ONCE over a global grid spanning the widest band and then
subset per band width, so widening the band costs nothing extra.
"""
import argparse
from pathlib import Path

import numpy as np

from src.maps import iterate_asym, tokenize_trajectory


def encode(tok, k):
    """Base-n_bins encoding of every length-k window, plus the following token."""
    n = len(tok) - k
    key = np.zeros(n, dtype=np.int64)
    for i in range(k):
        key = key * 64 + tok[i:i + n]
    return key, tok[k:k + n]


class SparseKGram:
    """Order-k counts held as sorted keys + rows, for fast batched lookup."""

    def __init__(self, tok, k, n_bins=64):
        self.k, self.B = k, n_bins
        key, nxt = encode(tok, k)
        order = np.argsort(key, kind="stable")
        key, nxt = key[order], nxt[order]
        self.keys, start = np.unique(key, return_index=True)
        self.rows = np.zeros((len(self.keys), n_bins), dtype=np.float32)
        idx = np.searchsorted(self.keys, key)
        np.add.at(self.rows, (idx, nxt), 1.0)
        # Laplace on the unigram: it is the backoff of last resort, and a
        # 50-token context visits only a fraction of the 64 bins, so an
        # unsmoothed unigram would assign the rest probability zero and send
        # the held-out cross-entropy to infinity.
        u = np.bincount(tok, minlength=n_bins).astype(np.float64) + 1.0
        self.unigram = u / u.sum()

    def logp(self, query_keys, eps):
        """(m, B) log-probabilities; unseen contexts back off to the unigram."""
        i = np.searchsorted(self.keys, query_keys)
        i_clipped = np.clip(i, 0, len(self.keys) - 1)
        hit = self.keys[i_clipped] == query_keys
        rows = np.where(hit[:, None], self.rows[i_clipped], 0.0)
        p = rows + eps * self.B * self.unigram[None, :]
        return np.log(p / p.sum(axis=1, keepdims=True))

    def prob(self, query_keys, eps):
        return np.exp(self.logp(query_keys, eps))


def orbit_tokens(R, alpha, n_bins, n_steps, burn_in=2000, seed=5):
    x = np.random.default_rng(seed).uniform(0.2, 0.8)
    return tokenize_trajectory(
        iterate_asym(x, R, alpha, burn_in + n_steps)[burn_in:], n_bins)


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


def context_keys(ctx, k):
    """Keys of every in-context k-gram, and of the final predictive k-gram."""
    m, L = ctx.shape
    n = L - k
    hist = np.zeros((m, n), dtype=np.int64)
    for i in range(k):
        hist = hist * 64 + ctx[:, i:i + n]
    nxt = ctx[:, k:k + n]
    query = np.zeros(m, dtype=np.int64)
    for i in range(k):
        query = query * 64 + ctx[:, L - k + i]
    return hist, nxt, query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--band_widths", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                             0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    ap.add_argument("--heldout_alphas", type=float, nargs="+",
                    default=[0.20, 0.22, 0.24, 0.26, 0.28, 0.30])
    ap.add_argument("--R_eval", type=float, nargs="+",
                    default=[0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
    ap.add_argument("--cand_alpha_step", type=float, default=0.0125)
    ap.add_argument("--n_cand_R", type=int, default=15)
    ap.add_argument("--cand_steps", type=int, default=60_000)
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--context_len", type=int, default=50)
    ap.add_argument("--traj_len", type=int, default=150)
    ap.add_argument("--n_traj", type=int, default=6)
    ap.add_argument("--eps_grid", type=float, nargs="+",
                    default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
                    help="smoothing is tuned over this grid so each reference "
                         "is given its best shot rather than being handicapped")
    ap.add_argument("--out", default="figures_asym2/band_bayes_baseline.npz")
    a = ap.parse_args()
    B = a.n_bins

    # ---- held-out evaluation contexts, shared by every predictor ----------
    cells = []
    for al in a.heldout_alphas:
        for R in a.R_eval:
            ctx, tgt = make_contexts(R, al, B, a.context_len, a.n_traj,
                                     a.traj_len, seed=int(1000 * al + R * 7))
            cells.append({"alpha": al, "R": R, "ctx": ctx, "tgt": tgt})
    n_ctx = sum(len(c["ctx"]) for c in cells)
    print(f"held-out: {len(cells)} cells, {n_ctx} contexts "
          f"(alpha {a.heldout_alphas[0]}-{a.heldout_alphas[-1]})")

    # ---- global candidate pool, fitted once ------------------------------
    lo = 1.0 - max(a.band_widths)
    cand_alpha = np.arange(lo, 1.0 + 1e-9, a.cand_alpha_step)
    cand_R = np.linspace(0.125, 1.0, a.n_cand_R)
    pool = [(R, al) for al in cand_alpha for R in cand_R]
    print(f"fitting {len(pool)} candidates on alpha in [{lo:.2f}, 1.0] "
          f"x {a.n_cand_R} R values, {a.cand_steps} steps each ...", flush=True)
    cand_tok = [orbit_tokens(R, al, B, a.cand_steps) for R, al in pool]
    cand_a = np.array([al for _, al in pool])

    results = {}
    for k in a.orders:
        print(f"\n--- order {k} ---", flush=True)
        cand_models = [SparseKGram(t, k, B) for t in cand_tok]

        # oracle: fit the true map of each held-out cell
        true_models = [SparseKGram(
            orbit_tokens(c["R"], c["alpha"], B, a.cand_steps), k, B)
            for c in cells]

        def oracle_at(e):
            out = []
            for c, m in zip(cells, true_models):
                _, _, q = context_keys(c["ctx"], k)
                p = m.prob(q, e)
                out.append(-np.log(np.maximum(
                    p[np.arange(len(c["tgt"])), c["tgt"]], 1e-300)).mean())
            return float(np.mean(out))
        oracle = min(oracle_at(e) for e in a.eps_grid)

        # context-only: fit on the context window itself
        ctx_models = [[SparseKGram(c["ctx"][i], k, B)
                       for i in range(len(c["ctx"]))] for c in cells]

        def context_at(e):
            out = []
            for c, models in zip(cells, ctx_models):
                ctx, tgt = c["ctx"], c["tgt"]
                ce = np.empty(len(ctx))
                for i, m in enumerate(models):
                    _, _, q = context_keys(ctx[i][None, :], k)
                    ce[i] = -np.log(max(m.prob(q, e)[0, tgt[i]], 1e-300))
                out.append(ce.mean())
            return float(np.mean(out))
        context_only = min(context_at(e) for e in a.eps_grid)
        print(f"  oracle       {oracle:.4f}")
        print(f"  context-only {context_only:.4f}")

        # band-restricted Bayes, one value per band width
        per_w, tuned_eps = [], None
        for w in a.band_widths:
            sel = np.where(cand_a >= 1.0 - w - 1e-9)[0]
            trial = a.eps_grid if tuned_eps is None else [tuned_eps]
            best = (np.inf, None)
            for eps in trial:
              ce_cells = []
              for c in cells:
                hist, nxt, query = context_keys(c["ctx"], k)
                m_ctx = len(c["ctx"])
                ll = np.empty((len(sel), m_ctx))
                pred = np.empty((len(sel), m_ctx, B), dtype=np.float32)
                for j, ci in enumerate(sel):
                    mdl = cand_models[ci]
                    lp = mdl.logp(hist.ravel(), eps).reshape(m_ctx, -1, B)
                    ll[j] = np.take_along_axis(
                        lp, nxt[:, :, None], axis=2)[:, :, 0].sum(axis=1)
                    pred[j] = mdl.prob(query, eps)
                ll -= ll.max(axis=0, keepdims=True)
                wgt = np.exp(ll); wgt /= wgt.sum(axis=0, keepdims=True)
                p = np.einsum("jm,jmb->mb", wgt, pred)
                ce_cells.append(-np.log(np.maximum(
                    p[np.arange(m_ctx), c["tgt"]], 1e-300)).mean())
              val = float(np.mean(ce_cells))
              if val < best[0]:
                  best = (val, eps)
            tuned_eps = tuned_eps or best[1]
            per_w.append(best[0])
            print(f"  w={w:<5g} band=[{1-w:.2f},1] n_cand={len(sel):<4d} "
                  f"band_bayes {best[0]:.4f}  (eps={best[1]:g})", flush=True)
        results[k] = {"band_bayes": np.array(per_w), "oracle": oracle,
                      "context_only": context_only}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, band_widths=np.array(a.band_widths),
             orders=np.array(a.orders),
             band_bayes=np.stack([results[k]["band_bayes"] for k in a.orders]),
             oracle=np.array([results[k]["oracle"] for k in a.orders]),
             context_only=np.array([results[k]["context_only"]
                                    for k in a.orders]),
             heldout_alphas=np.array(a.heldout_alphas),
             R_eval=np.array(a.R_eval), eps=eps)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
