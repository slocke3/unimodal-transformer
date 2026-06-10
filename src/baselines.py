"""
Finite-memory / k-gram baselines = Ulam discretizations of the transfer operator.

A k-gram model estimates  P(x_{t+1} | x_{t-k+1..t})  by counting transitions in
tokenized trajectories. For order k=1 this is the 1-step transition matrix on the
N-bin partition -- the Ulam approximation of the Perron-Frobenius (transfer)
operator. These models are the "operator floor": the optimal finite-memory
symbolic predictor at a given order.

The point of the harness: comparing a transformer's cross-entropy against this
floor tells us whether it does anything *beyond* finite-order transition
counting -- especially on held-out parameters / out-of-family maps, where the
counting tables have no entries but a model that learned the dynamics could
still predict.

Pure numpy -- no torch -- so it runs anywhere and stays fast.
"""
import numpy as np

from .maps import iterate_general, tokenize_trajectory


def gen_token_seqs(map_fn, param, n_eval, n_bins, traj_len,
                   burn_in=0, rng=None, x0_range=(0.05, 0.95)):
    """Generate tokenized trajectories from a map (torch-free eval helper).

    map_fn(x, param) -> next x. Returns a list of int token arrays, one per
    sampled initial condition. Mirrors how the transformer's eval builds data,
    so the k-gram floor can be scored on identical contexts.
    """
    if rng is None:
        rng = np.random.default_rng()
    seqs = []
    for _ in range(n_eval):
        x0 = rng.uniform(*x0_range)
        traj = iterate_general(x0, map_fn, param, burn_in + traj_len)[burn_in:]
        seqs.append(tokenize_trajectory(traj, n_bins))
    return seqs


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_context_target_pairs(token_seqs, context_len):
    """Slide a window over tokenized trajectories.

    token_seqs : iterable of 1D int arrays (one per trajectory)
    Returns (contexts, targets) with shapes (M, context_len) and (M,).
    """
    contexts, targets = [], []
    for seq in token_seqs:
        seq = np.asarray(seq)
        for t in range(context_len, len(seq)):
            contexts.append(seq[t - context_len:t])
            targets.append(seq[t])
    if not contexts:
        return (np.empty((0, context_len), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)


# ---------------------------------------------------------------------------
# k-gram / order-k Markov model  (= order-k Ulam discretization)
# ---------------------------------------------------------------------------

class KGramModel:
    """Order-k Markov next-token model fit by counting.

    Parameters
    ----------
    n_bins : int          size of the token alphabet (partition resolution N)
    order  : int          memory length k (k=1 -> the Ulam transition matrix)
    alpha  : float         additive (Laplace) smoothing; 0.0 = raw empirical.
                           Keep a small alpha for held-out CE so unseen
                           transitions don't give infinite loss; use 0.0 when
                           checking exact operator properties (e.g. doubling).
    """

    def __init__(self, n_bins, order, alpha=1e-3):
        self.n_bins = int(n_bins)
        self.order = int(order)
        self.alpha = float(alpha)
        self.counts = {}                       # context tuple -> counts[n_bins]
        self.unigram = np.zeros(self.n_bins)   # order-0 backoff

    # -- fitting ----------------------------------------------------------
    def fit(self, token_seqs):
        for seq in token_seqs:
            seq = np.asarray(seq)
            for t in range(self.order, len(seq)):
                ctx = tuple(int(s) for s in seq[t - self.order:t])
                nxt = int(seq[t])
                if ctx not in self.counts:
                    self.counts[ctx] = np.zeros(self.n_bins)
                self.counts[ctx][nxt] += 1.0
                self.unigram[nxt] += 1.0
        return self

    def _normalize(self, count_vec):
        c = count_vec + self.alpha
        s = c.sum()
        return c / s if s > 0 else np.full(self.n_bins, 1.0 / self.n_bins)

    # -- prediction -------------------------------------------------------
    def predict_proba(self, context):
        ctx = tuple(int(s) for s in np.asarray(context)[-self.order:]) if self.order > 0 else ()
        if ctx in self.counts:
            return self._normalize(self.counts[ctx])
        # backoff: smoothed unigram, else uniform
        if self.unigram.sum() > 0:
            return self._normalize(self.unigram)
        return np.full(self.n_bins, 1.0 / self.n_bins)

    def predict_proba_batch(self, contexts):
        return np.stack([self.predict_proba(c) for c in contexts])

    def cross_entropy(self, contexts, targets):
        """Mean next-token cross-entropy in nats (the operator-floor loss)."""
        probs = self.predict_proba_batch(contexts)
        p_true = probs[np.arange(len(targets)), np.asarray(targets)]
        return float(-np.log(p_true).mean())

    def accuracy(self, contexts, targets):
        probs = self.predict_proba_batch(contexts)
        return float((probs.argmax(axis=1) == np.asarray(targets)).mean())

    # -- transfer-operator view (order-1) ---------------------------------
    def transition_matrix(self):
        """N x N row-stochastic Ulam matrix (order-1 only)."""
        if self.order != 1:
            raise ValueError("transition_matrix is defined only for order == 1")
        T = np.empty((self.n_bins, self.n_bins))
        for i in range(self.n_bins):
            ctx = (i,)
            if ctx in self.counts:
                T[i] = self._normalize(self.counts[ctx])
            else:
                T[i] = np.full(self.n_bins, 1.0 / self.n_bins)
        return T

    def spectrum(self):
        """Eigenvalues of the Ulam matrix, sorted by descending modulus."""
        eig = np.linalg.eigvals(self.transition_matrix())
        return eig[np.argsort(-np.abs(eig))]

    def stationary_distribution(self):
        """Invariant distribution (left eigenvector for eigenvalue 1)."""
        T = self.transition_matrix()
        w, v = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        pi = pi / pi.sum()
        return pi

    def entropy_rate(self):
        """Order-1 Markov entropy rate  h = -sum_i pi_i sum_j T_ij log T_ij.

        For the doubling map on a dyadic partition this equals ln 2 = lambda.
        """
        T = self.transition_matrix()
        pi = self.stationary_distribution()
        with np.errstate(divide="ignore", invalid="ignore"):
            logT = np.where(T > 0, np.log(T), 0.0)
        return float(-(pi[:, None] * T * logT).sum())


def operator_floor_curve(train_seqs, contexts, targets, n_bins, orders, alpha=1e-3):
    """Fit k-gram models at several orders and report held-out CE/accuracy.

    Returns a dict: order -> {"ce": float, "acc": float}. The CE values are the
    operator floor at each memory length; they decrease toward the true entropy
    rate as the partition/order captures the symbolic dynamics.
    """
    out = {}
    for k in orders:
        m = KGramModel(n_bins=n_bins, order=k, alpha=alpha).fit(train_seqs)
        out[k] = {"ce": m.cross_entropy(contexts, targets),
                  "acc": m.accuracy(contexts, targets)}
    return out
