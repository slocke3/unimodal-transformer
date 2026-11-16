"""Reference predictors on the task-diversity sweep, scored on the transformer's
own evaluation contexts (reviewer FWzs).

For every (m, seed) of the runs behind taskdiv_gap and every input resolution
N, this regenerates -- bit for bit, without torch -- the training windows that
train_subset.py built and the new-task / seen-task evaluation windows that
evaluate_per_r scored, then fits two kinds of predictor to the same training
data:

  k-gram (k = 1, 2, 4, 6, 10)
                         Count which output bin (64 bins) follows each history
                         of the last k input tokens, and the mean raw next state
                         for the square loss. A history never seen in training
                         backs off to the longest seen suffix (down to order 0,
                         the marginal). CE uses additive smoothing alpha on the
                         chosen order, picked per (N, k) from {1e-5,..,1} on the
                         run's own validation split (seen tasks). Knows nothing
                         about the logistic family: a local-transition strategy.

  least squares r-hat    Knows the family. Replaces each context token by its
                         bin midpoint z, fits z_{t+1} = r z_t (1 - z_t) by least
                         squares over the 49 transitions in the 50-token
                         context, clips r-hat to [0.5, 4] and predicts
                         r-hat z_L (1 - z_L). Square loss only. Uses no training
                         data: an explicit system-identification strategy whose
                         error comes from quantization alone.

Output: runs_rebuttal_baselines/m{m}_seed{s}.npz with, per N and predictor,
per-r CE and MSE on the new-task grid and on the seen tasks. Torch-free:

  python scripts/rebuttal/baselines_taskdiv.py                 # all m, seeds 0-3
  python scripts/rebuttal/baselines_taskdiv.py --ms 1000 --seeds 0

Existing results are preserved; only missing predictors are computed. For example,
--ks 10 adds the ten-token baseline to an existing sweep.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.maps import tokenize_trajectory  # noqa: E402

MS = [100, 250, 500, 1000, 2000, 4000, 8000]
NS = [8, 16, 32, 64, 128, 256]
KS = [1, 2, 4, 6, 10]
ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
L, TRAJ_LEN, N_OUT = 50, 150, 64
N_TRAIN, VAL_FRAC, MAX_VAL = 32_000, 0.15, 600
R_GRID = np.linspace(0.5, 4.0, 300)
N_EVAL_PER_R, N_SEEN = 30, 200


def trajectories(x0, r):
    """(n, TRAJ_LEN+1) array; identical to src.maps.iterate_map row by row."""
    out = np.empty((len(x0), TRAJ_LEN + 1))
    x = np.array(x0, dtype=float)
    out[:, 0] = x
    for i in range(TRAJ_LEN):
        x = r * x * (1.0 - x)
        out[:, i + 1] = x
    return out


def split(m, seed):
    """train_r, training-trajectory r's, validation r's -- as train_subset.py."""
    train_r = np.sort(np.random.default_rng(seed).uniform(0.5, 4.0, size=m))
    n_per_r, n_extra = divmod(N_TRAIN, m)
    rng = np.random.default_rng(seed)
    counts = np.full(m, n_per_r)
    if n_extra:
        counts[rng.permutation(m)[:n_extra]] += 1
    all_r = np.repeat(train_r, counts)
    rng.shuffle(all_r)
    n_val = min(int(VAL_FRAC * len(all_r)), MAX_VAL)
    return train_r, all_r[n_val:], all_r[:n_val]


def dataset_trajs(r_values, seed):
    x0 = np.random.default_rng(seed).uniform(0.05, 0.95, size=len(r_values))
    return trajectories(x0, r_values)


def eval_trajs(r_list, seed):
    """evaluate_per_r: per r, N_EVAL_PER_R fresh x0 from one sequential stream."""
    r_rep = np.repeat(r_list, N_EVAL_PER_R)
    x0 = np.random.default_rng(seed).uniform(0.05, 0.95, size=len(r_rep))
    return trajectories(x0, r_rep)


def hist_keys(tok, j, n):
    """Encode histories exactly, using two words when int64 would overflow.

    tok: (n_traj, T) tokens. Returns (n_traj, n_win). Short histories use
    the original integer encoding; longer ones use structured (recent, older)
    integer keys. No hashes or truncation are used.
    """
    n_win = tok.shape[1] - (L + 1)
    shape = (tok.shape[0], n_win)
    if n**j - 1 <= np.iinfo(np.int64).max:
        key = np.zeros(shape, dtype=np.int64)
        for d in range(1, j + 1):          # token at target-d
            key = key * n + tok[:, L - d: L - d + n_win]
        return key
    split = 63 // (int(n) - 1).bit_length()
    if n**(j - split) - 1 > np.iinfo(np.uint64).max:
        raise ValueError("History exceeds the capacity of the exact two-word encoding")
    key = np.zeros(shape, dtype=[("recent", np.uint64), ("older", np.uint64)])
    for d in range(1, j + 1):
        field = "recent" if d <= split else "older"
        key[field] = key[field] * n + tok[:, L - d: L - d + n_win].astype(np.uint64)
    return key


class KGram:
    """Counts for orders 0..kmax on one training set."""

    def __init__(self, tok, tgt_bin, tgt_raw, n, kmax):
        self.n, self.models = n, {}
        tb, tr = tgt_bin.reshape(-1), tgt_raw.reshape(-1)
        for j in range(kmax + 1):
            h = hist_keys(tok, j, n).reshape(-1)
            uh, inv, tot = np.unique(h, return_inverse=True, return_counts=True)
            mean = np.bincount(inv, weights=tr) / tot
            # Dense history IDs keep joint history/target keys small even when
            # the original history requires more than one machine word.
            pair = inv * N_OUT + tb
            up, pc = np.unique(pair, return_counts=True)
            self.models[j] = (uh, tot, mean, up, pc)

    def predict(self, tok, tgt_bin, k, alpha):
        """Per-window p(target bin) and predicted mean, with backoff."""
        shape = (tok.shape[0], tok.shape[1] - (L + 1))
        p = np.full(shape, np.nan)
        mu = np.full(shape, np.nan)
        todo = np.ones(shape, dtype=bool)
        for j in range(k, -1, -1):
            uh, tot, mean, up, pc = self.models[j]
            h = hist_keys(tok, j, self.n)
            i = np.clip(np.searchsorted(uh, h), 0, len(uh) - 1)
            seen = (uh[i] == h) & todo
            pair = i * N_OUT + tgt_bin
            ip = np.clip(np.searchsorted(up, pair), 0, len(up) - 1)
            c = np.where(up[ip] == pair, pc[ip], 0)
            p[seen] = ((c + alpha) / (tot[i] + N_OUT * alpha))[seen]
            mu[seen] = mean[i][seen]
            todo &= ~seen
        return p, mu


def ls_rhat_pred(tok, n):
    """Least-squares r-hat on bin midpoints over each 50-token context."""
    z = (tok + 0.5) / n
    phi = z * (1 - z)
    a = np.cumsum(np.pad(z[:, 1:] * phi[:, :-1], ((0, 0), (1, 0))), axis=1)
    b = np.cumsum(np.pad(phi[:, :-1] ** 2, ((0, 0), (1, 0))), axis=1)
    n_win = tok.shape[1] - (L + 1)
    t = np.arange(n_win)
    # transitions (u, u+1) for u = t .. t+L-2 inside context t .. t+L-1
    num = a[:, t + L - 1] - a[:, t]
    den = b[:, t + L - 1] - b[:, t]
    rhat = np.clip(num / np.maximum(den, 1e-300), 0.5, 4.0)
    return rhat * phi[:, t + L - 1]


def per_r(values, n_r):
    """Mean over each r's N_EVAL_PER_R trajectories x windows."""
    return values.reshape(n_r, -1).mean(1)


def run(m, seed, out_dir, ks=KS, ns=NS):
    t0 = time.time()
    path = os.path.join(out_dir, f"m{m}_seed{seed}.npz")
    if os.path.exists(path):
        with np.load(path) as saved:
            out = {key: saved[key] for key in saved.files}
    else:
        out = {}
    pending = {}
    for n in ns:
        missing = []
        for k in ks:
            keys = [f"N{n}_k{k}_alpha"] + [
                f"N{n}_k{k}_{s}_{obj}"
                for s in ("new", "seen") for obj in ("ce", "mse")
            ]
            if not all(key in out for key in keys):
                missing.append(k)
        missing_lsr = [s for s in ("new", "seen") if f"N{n}_lsr_{s}_mse" not in out]
        if missing or missing_lsr:
            pending[n] = (missing, missing_lsr)
    if not pending:
        print(f"[baselines] m={m} seed={seed} already complete", flush=True)
        return

    train_r, tr_r, val_r = split(m, seed)
    seen_rng = np.random.default_rng(seed + 11)
    seen_r = (np.sort(seen_rng.choice(train_r, size=N_SEEN, replace=False))
              if m > N_SEEN else np.sort(train_r))
    raw = {"train": dataset_trajs(tr_r, seed), "val": dataset_trajs(val_r, seed + 1),
           "new": eval_trajs(R_GRID, seed + 7), "seen": eval_trajs(seen_r, seed + 13)}
    n_win = raw["train"].shape[1] - (L + 1)
    tgt_raw = {s: v[:, L:L + n_win] for s, v in raw.items()}
    tgt_bin = {s: tokenize_trajectory(v, N_OUT) for s, v in tgt_raw.items()}
    n_r = {"new": len(R_GRID), "seen": len(seen_r)}
    for key, value in {"r_grid": R_GRID, "seen_r": seen_r, "train_r": train_r}.items():
        if key in out and not np.array_equal(out[key], value):
            raise ValueError(f"Saved {key} does not match m={m}, seed={seed}")
        out[key] = value

    for n, (missing, missing_lsr) in pending.items():
        tok = {s: tokenize_trajectory(v, n) for s, v in raw.items()}
        if missing:
            km = KGram(tok["train"], tgt_bin["train"], tgt_raw["train"], n, max(missing))
        for k in missing:
            val_ce = []
            for al in ALPHAS:
                p, _ = km.predict(tok["val"], tgt_bin["val"], k, al)
                val_ce.append(-np.log(p).mean())
            alpha = ALPHAS[int(np.argmin(val_ce))]
            out[f"N{n}_k{k}_alpha"] = alpha
            for s in ("new", "seen"):
                p, mu = km.predict(tok[s], tgt_bin[s], k, alpha)
                out[f"N{n}_k{k}_{s}_ce"] = per_r(-np.log(p), n_r[s])
                out[f"N{n}_k{k}_{s}_mse"] = per_r((mu - tgt_raw[s]) ** 2, n_r[s])
        for s in missing_lsr:
            pred = ls_rhat_pred(tok[s], n)
            out[f"N{n}_lsr_{s}_mse"] = per_r((pred - tgt_raw[s]) ** 2, n_r[s])

    os.makedirs(out_dir, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **out)
    os.replace(path + ".tmp", path)
    print(f"[baselines] m={m} seed={seed} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=MS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ks", type=int, nargs="+", choices=range(1, 11), default=KS)
    ap.add_argument("--ns", type=int, nargs="+", choices=NS, default=NS)
    ap.add_argument("--out_dir", default="runs_rebuttal_baselines")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of independent (task-count, seed) fits to run concurrently")
    ap.add_argument("--check", action="store_true",
                    help="verify train_r against a saved run, then exit")
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("--workers must be positive")
    if a.check:
        for m in a.ms:
            for s in a.seeds:
                p = f"runs_divbins_320k/in64_bins_m{m}_seed{s}/eval_per_r.npz"
                if os.path.exists(p):
                    ok = np.allclose(np.load(p)["train_r"], split(m, s)[0])
                    print(f"m={m} seed={s}: train_r matches saved run: {ok}")
        return
    jobs = [(m, s) for m in dict.fromkeys(a.ms) for s in dict.fromkeys(a.seeds)]
    if a.workers == 1:
        for m, s in jobs:
            run(m, s, a.out_dir, ks=a.ks, ns=a.ns)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(run, m, s, a.out_dir, a.ks, a.ns) for m, s in jobs]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
