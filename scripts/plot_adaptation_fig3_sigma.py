"""Figure-3 analogue for continuous input, in map-space coordinates.

The parameter-coordinate version has a clamping baseline that RISES with the
band width, which makes the x axis part diversity and part probe difficulty. The
probe there sits a fixed 0.2 of alpha past the band edge, and a fixed step in
alpha is not a fixed step between maps: at low alpha the family reshapes faster
AND the orbit measure migrates onto the shifted peak, where nearby maps disagree
most. The two effects multiply to a 49% drift.

Nothing has to be retrained to fix this. Every run stored implied_rms on the full
alpha grid, so the probe can be re-read at whatever alpha sits a constant
MAP-space distance past that run's band edge. Arclength is measured in the
orbit-measure norm (see src.mapmetric), which is the one that absorbs both
effects; distances are in bin widths.

Left panel reproduces the parameter-coordinate figure, right panel is the same
runs read in map space, so the flattening of the dashed line is visible directly.
"""
import glob
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.maps import asym_map_vec, iterate_asym
from src.mapmetric import sigma_mu_of, param_of_sigma_mu

NB, OFFSET = 64, 8.0          # probe offset past the band edge, in bin widths


def orbit(a, R, n=30, traj=150, burn=50, seed=3):
    rng = np.random.default_rng(seed)
    return np.concatenate([iterate_asym(rng.uniform(.05, .95), R, a, traj)[burn:]
                           for _ in range(n)])


def clamp(a_probe, a_edge, Rs):
    """RMS between the probe map and the band-edge map, on the probe's orbit."""
    return float(np.mean([np.sqrt(np.mean(
        (asym_map_vec(x := orbit(a_probe, R), R, a_probe)
         - asym_map_vec(x, R, a_edge)) ** 2)) for R in Rs]))


def series(mode, coord):
    """(x, model RMS, clamp RMS, probe alpha) for one output mode."""
    out = []
    for d in sorted(glob.glob(f"runs_cont/asym_{mode}_w*_seed0"),
                    key=lambda s: float(re.search(r"_w([\d.]+)_", s).group(1))):
        z = np.load(d + "/eval_cont.npz")
        keep = z["R_grid"] >= 0.25
        edge = float(z["band_lo"])
        curve = np.nanmean(z["implied_rms"][:, keep], axis=1)   # RMS vs alpha
        if coord == "param":
            x, probe = float(z["band_width"]), edge - 0.2
        else:
            x = -sigma_mu_of("asym", edge)
            probe = param_of_sigma_mu("asym", sigma_mu_of("asym", edge) - OFFSET)
        # implied_rms is smooth in alpha, so read the probe off by interpolation
        model = float(np.interp(probe, z["p_grid"], curve))
        out.append((x, model, clamp(probe, edge, z["R_grid"][keep][::6]), probe))
    return list(zip(*out))


plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True)
YMAX = 0.115          # shared, so the null flattening is visible side by side

for ax, coord, xlabel, title in (
        (axes[0], "param", "training-band half-width  $w$   (units of $\\alpha$)",
         "parameter coordinate"),
        (axes[1], "sigma", "training-band half-width   (bin widths of map)",
         "map-space coordinate")):
    sx, sm, sc, sp = series("scalar", coord)
    bx, bm, bc, _ = series("bins", coord)
    drift = 100 * (max(sc) / min(sc) - 1)
    ax.plot(sx, sc, "k--", lw=1.6,
            label=f"Clamp to nearest trained map*  ({drift:.0f}% drift)")
    ax.plot(bx, bm, "o-", color="k", mfc="white", ms=7, lw=1.4,
            label="binned output")
    ax.plot(sx, sm, "o-", color="k", mfc="k", ms=7, lw=1.4, label="scalar output")
    ax.set_xlim(0, max(sx) * 1.12); ax.set_ylim(0, YMAX)
    ax.set_xlabel(xlabel)
    if ax is axes[0]:
        ax.set_ylabel("implied-map RMS at the probe")
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print(f"{coord:6s} probe alphas {np.round(sp, 3)}  null {np.round(sc, 4)}"
          f"  drift {drift:.0f}%")

fig.text(0.5, -0.13,
         "Continuous input, asymmetric family, same six runs in both panels.  Left: the probe sits a "
         "fixed 0.2 of $\\alpha$ past the band edge, and the null rises.\nRight: it sits a fixed "
         f"{OFFSET:.0f} bin widths past it in map space, and the null is flat, so the fall is the "
         "model rather than the probe.\n*the best predictor using only maps seen in training.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_cont/adaptation_fig3_sigma.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_cont/adaptation_fig3_sigma.png")
