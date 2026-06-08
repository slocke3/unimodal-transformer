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