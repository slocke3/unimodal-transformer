"""Correct binning reference for the implied return map.

The model observes only the BIN of x_n, so the best it can do is predict the
bin-conditional mean of the next state,

    m*(j) = E[ f(x) | x in bin j ]      (under the invariant measure)

The implied-map error is measured against f(bin centre), so an IDEAL predictor
does not score zero -- it scores RMS(m* - f(centre)), the systematic offset
between the bin-conditional mean and the centre value. That is the right
reference.

The earlier version normalised by RMS(f(x) - f(centre)) instead, which also
contains the within-bin variance of f:

    E[(f(x) - f(c))^2] = Var(f | bin) + (m* - f(c))^2

so it is strictly larger and a good model scores below 1.0 against it -- which
is exactly the impossible "0.68" that exposed the error.

Vectorised across the whole (parameter, R) grid: orbits for every cell are
advanced together, so the grid costs seconds rather than an hour of Python.
"""
import numpy as np


def _step(x, R, p, family):
    if family == "tilted":
        from .maps import tilt_norm
        C = R * np.array([tilt_norm(float(v)) for v in np.atleast_1d(p)])
        v = C * x * (1.0 - x) * (1.0 + p * (x - 0.5))
    else:
        xc = p / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            v = R * (x / xc) ** p * ((1.0 - x) / (1.0 - xc)) ** (2.0 - p)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, 0.0, 1.0)


def ideal_bin_reference(p_grid, R_grid, n_bins, family="asym",
                        n_steps=20000, burn_in=1000, seed=11):
    """RMS(m* - f(centre)) for every (p, R) cell, shape (len(p), len(R))."""
    P, RR = np.meshgrid(np.asarray(p_grid, float),
                        np.asarray(R_grid, float), indexing="ij")
    shape = P.shape
    p, R = P.ravel(), RR.ravel()
    n = p.size
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 0.8, n)

    if family == "tilted":
        from .maps import tilt_norm
        for v in np.unique(p):
            tilt_norm(float(v))              # warm the cache once per parameter

    for _ in range(burn_in):
        x = _step(x, R, p, family)

    centres = (np.arange(n_bins) + 0.5) / n_bins
    # accumulate, per cell and bin: count and sum of f(x)
    cnt = np.zeros((n, n_bins))
    ssum = np.zeros((n, n_bins))
    rows = np.arange(n)
    for _ in range(n_steps):
        b = np.clip((x * n_bins).astype(np.int64), 0, n_bins - 1)
        nxt = _step(x, R, p, family)
        np.add.at(cnt, (rows, b), 1.0)
        np.add.at(ssum, (rows, b), nxt)
        x = nxt

    seen = cnt > 0
    m_star = np.divide(ssum, np.maximum(cnt, 1.0))
    f_c = np.stack([_step(centres, R[i], p[i], family) for i in range(n)])
    w = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1.0)
    ref = np.sqrt((w * seen * (m_star - f_c) ** 2).sum(axis=1))
    return ref.reshape(shape)
