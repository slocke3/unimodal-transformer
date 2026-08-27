from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Core map: quadratic (logistic) family
# ---------------------------------------------------------------------------

def quadratic_map(x, r):
    return r * x * (1.0 - x)


def iterate_map(x0, r, n_steps):
    traj = np.empty(n_steps + 1)
    traj[0] = x0
    for i in range(n_steps):
        traj[i + 1] = r * traj[i] * (1.0 - traj[i])
    return traj


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_trajectory(traj, n_bins):
    """Map a trajectory in [0,1] to integer bin indices in {0, ..., n_bins-1}."""
    tokens = np.floor(traj * n_bins).astype(np.int64)
    return np.clip(tokens, 0, n_bins - 1)


def detokenize(tokens, n_bins):
    """Map bin indices back to bin centers."""
    return (tokens + 0.5) / n_bins


# ---------------------------------------------------------------------------
# Lyapunov exponent
# ---------------------------------------------------------------------------

def compute_lyapunov(r, n_steps=100_000, x0=None):
    if x0 is None:
        x0 = np.random.default_rng(seed=42).uniform(0.1, 0.9)
    x = x0
    for _ in range(1000):
        x = r * x * (1.0 - x)
    log_sum = 0.0
    count = 0
    for _ in range(n_steps):
        deriv = abs(r * (1.0 - 2.0 * x))
        if deriv > 1e-10:
            log_sum += np.log(deriv)
            count += 1
        x = r * x * (1.0 - x)
    return log_sum / count if count > 0 else 0.0


def compute_lyapunov_array(r_values, n_steps=100_000, verbose=True):
    lyapunovs = np.empty(len(r_values))
    for i, r in enumerate(r_values):
        lyapunovs[i] = compute_lyapunov(r, n_steps=n_steps)
        if verbose and (i + 1) % 500 == 0:
            print(f"  Lyapunov: {i+1}/{len(r_values)} done")
    return lyapunovs


# ---------------------------------------------------------------------------
# Other unimodal map families (for generalization experiments)
# ---------------------------------------------------------------------------

def tent_map(x, s):
    return np.where(x < 0.5, s * x, s * (1.0 - x))


def sine_map(x, r):
    return r * np.sin(np.pi * x)


def cubic_map(x, r):
    y = 2 * x - 1
    z = r * y * (1 - y**2)
    return (z + 1) / 2.0


def iterate_general(x0, map_fn, param, n_steps):
    traj = np.empty(n_steps + 1)
    traj[0] = x0
    for i in range(n_steps):
        traj[i + 1] = map_fn(traj[i], param)
        traj[i + 1] = np.clip(traj[i + 1], 0.0, 1.0)
    return traj


def tent_deriv(x, s):
    return s if x < 0.5 else -s


def sine_deriv(x, r):
    return r * np.pi * np.cos(np.pi * x)


def cubic_deriv(x, r):
    y = 2 * x - 1
    return r * (1 - 3 * y**2)


def compute_lyapunov_general(map_fn, deriv_fn, param, n_steps=100_000, x0=0.4):
    x = x0
    for _ in range(1000):
        x = map_fn(x, param)
        x = np.clip(x, 1e-10, 1 - 1e-10)
    log_sum = 0.0
    count = 0
    for _ in range(n_steps):
        d = abs(deriv_fn(x, param))
        if d > 1e-10:
            log_sum += np.log(d)
            count += 1
        x = map_fn(x, param)
        x = np.clip(x, 1e-10, 1 - 1e-10)
    return log_sum / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Doubling map x -> 2x mod 1  (rigorous control)
# ---------------------------------------------------------------------------
# Uniformly expanding, deriv = 2 everywhere, so lambda = ln 2. With a dyadic
# partition (n_bins a power of 2) it is an exact Markov map: the order-1
# transition matrix IS the Ulam discretization of its Perron-Frobenius
# operator, and the optimal-predictor cross-entropy equals ln 2 exactly. This
# makes it the clean control for validating the empirical k-gram baseline.
#
# NOTE: in float64 the orbit loses one bit of precision per step and collapses
# to 0 after ~52 iterations. Generate only SHORT trajectories (< ~30 steps)
# from many initial conditions when sampling it.

def doubling_map(x):
    return (2.0 * x) % 1.0


def doubling_deriv(x):
    return 2.0


# ---------------------------------------------------------------------------
# Family registry (used in generalization experiments)
# ---------------------------------------------------------------------------

FAMILIES = {
    "quadratic": {
        "map_fn":   lambda x, r: r * x * (1 - x),
        "deriv_fn": lambda x, r: r * (1 - 2 * x),
        "params":   np.linspace(0.5, 4.0, 200),
        "color":    "#1B2A4A",
    },
    "tent": {
        "map_fn":   tent_map,
        "deriv_fn": tent_deriv,
        "params":   np.linspace(0.2, 2.0, 200),
        "color":    "#1D9E75",
    },
    "sine": {
        "map_fn":   sine_map,
        "deriv_fn": sine_deriv,
        "params":   np.linspace(0.1, 1.0, 200),
        "color":    "#378ADD",
    },
    "cubic": {
        "map_fn":   cubic_map,
        "deriv_fn": cubic_deriv,
        "params":   np.linspace(0.1, 1.0, 200),
        "color":    "#D85A30",
    },
}


# ---------------------------------------------------------------------------
# Asymmetric unimodal family (peak position as a free parameter)
# ---------------------------------------------------------------------------
#
#   g_{alpha,beta}(x) = R (x/x_c)^alpha ((1-x)/(1-x_c))^beta,  x_c = alpha/(alpha+beta)
#
# We use the one-parameter slice beta = 2 - alpha, so alpha + beta = 2 and the
# critical point sits at x_c = alpha/2. Verified properties:
#
#   * R is the PEAK HEIGHT: max_x g = g(x_c) = R exactly. The logistic map peaks
#     at r/4, so R = r/4 is the correspondence, and at alpha = 1 the two agree
#     to machine zero.
#   * The critical point stays quadratic for every alpha (the drop from the peak
#     goes as u^2 with coefficient (alpha+beta)^3/(2*alpha*beta) = 4/(alpha*beta)),
#     so the family stays in the logistic universality class.
#   * Near the origin g ~ x^alpha, so g'(0) is infinite for alpha < 1, 4R at
#     alpha = 1, and ZERO for alpha > 1. For alpha > 1 the origin is therefore
#     superattracting and a growing fraction of orbits die there -- at alpha=1.4,
#     R=1 every orbit does. Sweeps should stay at alpha <= 1 unless that
#     degeneracy is the object of study.

def asym_map(x, R, alpha):
    """One step of the asymmetric family with beta = 2 - alpha."""
    beta = 2.0 - alpha
    x_c = alpha / 2.0
    x = np.clip(x, 0.0, 1.0)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return R * (x / x_c) ** alpha * ((1.0 - x) / (1.0 - x_c)) ** beta


def asym_derivative(x, R, alpha):
    """g'(x) = g(x) * (alpha/x - beta/(1-x))."""
    beta = 2.0 - alpha
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return asym_map(x, R, alpha) * (alpha / x - beta / (1.0 - x))


def iterate_asym(x0, R, alpha, n_steps):
    """Orbit of the asymmetric map, same signature style as iterate_map."""
    beta = 2.0 - alpha
    x_c = alpha / 2.0
    traj = np.empty(n_steps + 1)
    traj[0] = x0
    x = x0
    for i in range(n_steps):
        if x <= 0.0 or x >= 1.0:
            x = 0.0
        else:
            x = R * (x / x_c) ** alpha * ((1.0 - x) / (1.0 - x_c)) ** beta
        traj[i + 1] = x
    return traj


def compute_lyapunov_asym(R, alpha, n_steps=20_000, burn_in=1_000, x0=None):
    """Lyapunov exponent of the asymmetric map at (R, alpha)."""
    if x0 is None:
        x0 = np.random.default_rng(seed=42).uniform(0.2, 0.8)
    x = x0
    for _ in range(burn_in):
        x = asym_map(x, R, alpha)
    log_sum, count = 0.0, 0
    for _ in range(n_steps):
        d = abs(asym_derivative(x, R, alpha))
        if d > 1e-12:
            log_sum += np.log(d)
            count += 1
        x = asym_map(x, R, alpha)
    return log_sum / count if count > 0 else float("nan")


def r_to_R(r):
    """Logistic r -> peak height R. Both families peak at the same height."""
    return r / 4.0


def R_to_r(R):
    return 4.0 * R


# ---------------------------------------------------------------------------
# Tilted-logistic family (peak position, with both endpoints kept regular)
# ---------------------------------------------------------------------------
#
#   f_s(x) = C x (1-x) (1 + s (x - 1/2)),    C chosen so max f = R
#
# s = 0 is the logistic map exactly (C = 4R, i.e. r = 4R). Unlike the asymmetric
# family above, BOTH endpoint exponents stay at 1 for every s, so neither
# endpoint can become superattracting -- the degeneracy that confined that
# family to alpha <= 1 is absent here by construction. Verified usable over
# s in [-1.9, +1.4]: no orbit death, and f'(0) > 1 throughout (it crosses 1 near
# s = 1.45, which is the true boundary). The peak sweeps x_c = 0.338 -> 0.636.
#
# The point of this family is that the base map sits in the INTERIOR of the
# usable range, so a band |s| <= w centred on logistic has held-out regions on
# both sides of it.

def _tilt_peak(s):
    """argmax of x(1-x)(1 + s(x-1/2)) on (0,1), by solving the cubic's root."""
    # f = x(1-x)(1 + s(x-1/2));  f' = -3s x^2 + (3s - 2) x + (1 - s/2)
    a, b, c = -3.0 * s, 3.0 * s - 2.0, 1.0 - 0.5 * s
    if abs(a) < 1e-12:
        return 0.5
    disc = b * b - 4 * a * c
    if disc >= 0.0:
        roots = [r for r in ((-b + np.sqrt(disc)) / (2 * a),
                             (-b - np.sqrt(disc)) / (2 * a))
                 if 0.0 < r < 1.0]
        if roots:
            return max(roots, key=lambda x: x * (1 - x) * (1 + s * (x - 0.5)))
    # fallback: no interior stationary point found analytically
    g = np.linspace(1e-9, 1 - 1e-9, 200001)
    return float(g[np.argmax(g * (1 - g) * (1 + s * (g - 0.5)))])


@lru_cache(maxsize=8192)
def tilt_norm(s):
    """C such that max_x C x(1-x)(1 + s(x-1/2)) = 1.

    Cached: the cubic solve was otherwise repeated on every map evaluation,
    which made the per-cell binning floor ruinously slow.
    """
    xc = _tilt_peak(s)
    return 1.0 / (xc * (1.0 - xc) * (1.0 + s * (xc - 0.5)))


def tilted_map(x, R, s):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    v = R * tilt_norm(s) * x * (1.0 - x) * (1.0 + s * (x - 0.5))
    return v if v > 0.0 else 0.0


def iterate_tilted(x0, R, s, n_steps):
    C = R * tilt_norm(s)
    traj = np.empty(n_steps + 1)
    traj[0] = x0
    x = x0
    for i in range(n_steps):
        if x <= 0.0 or x >= 1.0:
            x = 0.0
        else:
            x = C * x * (1.0 - x) * (1.0 + s * (x - 0.5))
            if x < 0.0:
                x = 0.0
        traj[i + 1] = x
    return traj


# --- family dispatch -------------------------------------------------------
# Both parametric families take (R, p) with R the peak height, so downstream
# code can stay family-agnostic.

FAMILY_ITERATE = {"asym": iterate_asym, "tilted": iterate_tilted}
FAMILY_MAP = {"asym": asym_map, "tilted": tilted_map}


def iterate_family(x0, R, p, n_steps, family="asym"):
    return FAMILY_ITERATE[family](x0, R, p, n_steps)


def family_map(x, R, p, family="asym"):
    return FAMILY_MAP[family](x, R, p)


def asym_map_vec(x, R, alpha):
    """Vectorised asym_map."""
    beta = 2.0 - alpha
    x_c = alpha / 2.0
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    xm = x[m]
    out[m] = R * (xm / x_c) ** alpha * ((1.0 - xm) / (1.0 - x_c)) ** beta
    return out


def tilted_map_vec(x, R, s):
    """Vectorised tilted_map."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    xm = x[m]
    v = R * tilt_norm(float(s)) * xm * (1.0 - xm) * (1.0 + s * (xm - 0.5))
    out[m] = np.maximum(v, 0.0)
    return out


FAMILY_MAP_VEC = {"asym": asym_map_vec, "tilted": tilted_map_vec}


def family_map_vec(x, R, p, family="asym"):
    return FAMILY_MAP_VEC[family](x, R, p)
