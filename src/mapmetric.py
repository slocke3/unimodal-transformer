"""Arclength along a map family, so bands and probes can be set in map space.

Everywhere else in this repo a training band is an interval in the family's
parameter and a held-out probe sits a fixed parameter distance past its edge.
That is not the same as a fixed distance between maps. Two consequences showed
up in the asymmetric-family figures:

  * the clamping baseline -- RMS between the probe map and the band-edge map,
    which is what a model gets for free by ignoring the shift -- drifted 49%
    across the sweep, so the x axis mixed task diversity with probe difficulty;
  * the two families could not be put on one axis, since a unit of alpha and a
    unit of s buy quite different amounts of map.

The fix is to measure along the curve p -> g_p that the family traces in
L^2([0,1]). Its speed and signed arclength from the base map are

    v(p)     = || d g_p / d p ||_{L^2}
    sigma(p) = int_{p_base}^{p} v(u) du

with sigma reported in bin widths (n_bins * L^2 norm) so it reads as "how far
apart are these two maps, in units of the output discretisation". sigma is
strictly monotone, hence invertible, and bands become intervals in sigma.

Two properties make this cheap. First, every family here is f_{R,p} = R * g_p
with max g_p = 1, so R is a pure scale factor and drops out of the metric
exactly -- the table is built once at R = 1. Second, the curves are nearly
straight: arclength exceeds the chord ||g_p - g_base|| by only 1-2% for tilted,
so geodesic and chordal distance agree and either could serve.

The uniform-dx norm is the design coordinate rather than one weighted by the
invariant measure. A measure-weighted arclength flattens the asymmetric family
further (12% residual drift against 26%), but it depends on R and on sampled
orbits, and on the tilted family its speed collapses near s = 1.45 -- where the
origin stops repelling and lambda -> 0 -- which cuts the reachable range by
half. Uniform dx is deterministic, noise-free and analytically checkable.
"""
from functools import lru_cache

import numpy as np

from src.maps import asym_map_vec, tilted_map_vec, _tilt_peak, tilt_norm

# parameter ranges over which each family stays a well-behaved unimodal map,
# and the parameter value at which it IS the logistic map (the base, sigma = 0)
FAMILY_RANGE = {"asym": (0.18, 1.00), "tilted": (-1.95, 1.40)}
FAMILY_BASE = {"asym": 1.0, "tilted": 0.0}
_MAPVEC = {"asym": asym_map_vec, "tilted": tilted_map_vec}


def tilted_speed_exact(s, x, n_bins=64):
    """|d g_s / d s| for the tilted family, in bin widths, in closed form.

    g_s = h(x,s) / M(s) with h = x(1-x)(1 + s(x-1/2)) and M(s) = h(x_c(s), s).
    x_c is a stationary point of h, so the envelope theorem kills the chain-rule
    term and M'(s) = d_s h evaluated at x_c:

        d_s g_s = [ x(1-x)(x-1/2) - g_s(x) * x_c(1-x_c)(x_c-1/2) ] / M(s)

    Used to validate the finite-difference table; agreement is ~1e-11.
    """
    xc = _tilt_peak(float(s))
    M = 1.0 / tilt_norm(float(s))
    g = tilted_map_vec(x, 1.0, s)
    d = (x * (1 - x) * (x - 0.5) - g * xc * (1 - xc) * (xc - 0.5)) / M
    return n_bins * np.sqrt(np.trapz(d ** 2, x))


@lru_cache(maxsize=8)
def arclength_table(family, n_grid=1201, n_quad=6000, n_bins=64, h=1e-5):
    """(p_grid, sigma) with sigma in bin widths and sigma = 0 at the base map."""
    lo, hi = FAMILY_RANGE[family]
    mv = _MAPVEC[family]
    p = np.linspace(lo, hi, n_grid)
    x = np.linspace(1e-6, 1 - 1e-6, n_quad)
    v = np.array([n_bins * np.sqrt(np.trapz(
        ((mv(x, 1.0, pi + h) - mv(x, 1.0, pi - h)) / (2 * h)) ** 2, x))
        for pi in p])
    sigma = np.concatenate(([0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(p))))
    return p, sigma - np.interp(FAMILY_BASE[family], p, sigma)


def sigma_of(family, p):
    """Signed map-space distance from the base map, in bin widths."""
    g, s = arclength_table(family)
    return float(np.interp(p, g, s))


def param_of_sigma(family, sigma):
    """Inverse of sigma_of. Clipped to the family's usable range."""
    g, s = arclength_table(family)
    return float(np.interp(sigma, s, g))


def sigma_range(family):
    g, s = arclength_table(family)
    return float(s[0]), float(s[-1])


@lru_cache(maxsize=8)
def measure_arclength_table(family, n_grid=85, n_bins=64, h=1e-4, n_orbit=24,
                            traj_len=150, burn=50, n_R=8, seed=3):
    """Arclength in the L^2(mu_p) norm, mu_p being the orbit measure of g_p.

    The uniform-dx table weights every x equally, but a model only ever sees the
    x values its orbits visit, and in the asymmetric family that measure migrates
    as alpha falls -- toward the shifted peak, which is exactly where nearby maps
    disagree most. That migration was the larger half of the 49% drift in the
    clamping baseline (a factor 1.30, against 1.14 from the map geometry).

    Weighting the speed by the orbit measure absorbs it: drift across the sweep
    falls to 12%, against 26% for uniform dx. The cost is that this table depends
    on sampled orbits and on the R range, so it is used for reading probes off an
    existing evaluation grid, not for defining bands to train on.
    """
    from src.maps import iterate_family
    lo, hi = FAMILY_RANGE[family]
    mv = _MAPVEC[family]
    p = np.linspace(lo, hi, n_grid)
    Rs = np.linspace(0.25, 1.0, n_R)
    v = np.empty(n_grid)
    for i, pi in enumerate(p):
        acc = []
        for R in Rs:
            rng = np.random.default_rng(seed)
            x = np.concatenate([
                iterate_family(rng.uniform(0.05, 0.95), R, pi, traj_len,
                               family)[burn:] for _ in range(n_orbit)])
            d = (mv(x, R, pi + h) - mv(x, R, pi - h)) / (2 * h * R)
            acc.append(n_bins * np.sqrt(np.mean(d ** 2)))
        v[i] = np.mean(acc)
    sigma = np.concatenate(([0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(p))))
    return p, sigma - np.interp(FAMILY_BASE[family], p, sigma)


def sigma_mu_of(family, p):
    g, s = measure_arclength_table(family)
    return float(np.interp(p, g, s))


def param_of_sigma_mu(family, sigma):
    g, s = measure_arclength_table(family)
    return float(np.interp(sigma, s, g))
