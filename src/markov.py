"""Variable-order Markov sources, for testing CONCEPT shift.

Sweeping a map parameter changes which task you face but not the *form* of the
rule -- that is domain shift. Changing the ORDER of the dependence changes the
functional form itself, which is the concept shift we actually want to probe.

Design:

  task    an order-k conditional table P(x_t | x_{t-k..t-1}), rows drawn
          i.i.d. from Dirichlet(beta) over an n-symbol alphabet. A fresh table
          per sequence, so task diversity is effectively infinite and the model
          cannot memorise tasks -- in-context inference is the only option.
  arms    which orders appear in training. A FIXED-order arm makes order-k
          counting the optimal solution, so nothing rewards a general routine.
          A MIXED-order arm makes "match the longest suffix you can" optimal,
          because no single order is right, and that routine has no built-in
          ceiling -- which is the mechanism that could extrapolate to unseen k.
  test    orders inside and outside the training set.

The references are EXACT, not approximations, because Dirichlet-multinomial
conjugacy gives both the posterior predictive and the marginal likelihood in
closed form:

  full_bayes        marginalises over all orders up to k_max: the target.
  restricted_bayes  marginalises only over the TRAINING orders. On a sequence
                    of unseen order it must explain the data with a rule form
                    it wrongly believes is correct. This is the concept-shift
                    analogue of the band-restricted predictor, and the line the
                    model has to beat to have generalized.
"""
import numpy as np
from scipy.special import gammaln


def sample_chain(n, k, beta, rng):
    """Order-k conditional table, shape (n**k, n)."""
    return rng.dirichlet(np.full(n, beta), size=n ** k)


def generate(chain, n, k, length, rng):
    """Sequence of `length` symbols from an order-k table."""
    x = np.empty(length, dtype=np.int64)
    x[:k] = rng.integers(0, n, size=k)
    idx = 0
    for i in range(k):
        idx = idx * n + x[i]
    for t in range(k, length):
        x[t] = rng.choice(n, p=chain[idx])
        idx = (idx * n + x[t]) % (n ** k)
    return x


def _context_index(seq, k, n):
    """Index of the length-k context preceding each position t >= k."""
    L = len(seq)
    idx = np.zeros(L - k, dtype=np.int64)
    for i in range(k):
        idx = idx * n + seq[i:L - k + i]
    return idx


def counts(seq, n, k):
    """(n**k, n) transition counts within one sequence."""
    if k == 0:
        c = np.bincount(seq, minlength=n).astype(float)
        return c.reshape(1, n)
    idx = _context_index(seq, k, n)
    nxt = seq[k:]
    C = np.zeros((n ** k, n))
    np.add.at(C, (idx, nxt), 1.0)
    return C


def predictive(seq, n, k, beta):
    """Exact posterior predictive P(next | seq, order=k) under Dirichlet(beta)."""
    C = counts(seq, n, k)
    if k == 0:
        row = C[0]
    else:
        cur = 0
        for i in range(k):
            cur = cur * n + seq[len(seq) - k + i]
        row = C[cur]
    return (row + beta) / (row.sum() + n * beta)


def log_marginal(seq, n, k, beta):
    """log P(seq | order=k), conditioning on the first k symbols."""
    C = counts(seq, n, k)
    tot = C.sum(axis=1)
    return float(
        (gammaln(n * beta) - gammaln(tot + n * beta)).sum()
        + (gammaln(C + beta) - gammaln(beta)).sum()
    )


def bayes_mixture(seq, n, orders, beta, log_prior=None):
    """Predictive marginalising over a SET of orders, weighted by evidence.

    orders = all orders considered admissible -> full Bayes.
    orders = the training orders only        -> restricted Bayes.
    """
    orders = list(orders)
    lm = np.array([log_marginal(seq, n, k, beta) for k in orders])
    if log_prior is not None:
        lm = lm + np.asarray(log_prior, dtype=float)
    lm -= lm.max()
    w = np.exp(lm)
    w /= w.sum()
    p = np.zeros(n)
    for wi, k in zip(w, orders):
        p += wi * predictive(seq, n, k, beta)
    return p, w


def make_batch(n, orders, beta, length, n_seq, rng):
    """Sequences with their generating order, one fresh task each."""
    seqs = np.empty((n_seq, length), dtype=np.int64)
    ks = np.empty(n_seq, dtype=np.int64)
    for i in range(n_seq):
        k = int(rng.choice(orders))
        seqs[i] = generate(sample_chain(n, k, beta, rng), n, k, length, rng)
        ks[i] = k
    return seqs, ks
