"""Adaptation to concept shift, in one number.

The probe figures went through three coordinate systems -- band width in alpha,
arclength in map space, and a probe placed where the clamping baseline hits a
target -- because a raw error at a raw probe is only meaningful once you know how
hard that probe was. All of that divides out of a ratio:

    adaptation = (clamp - model) / (clamp - in-band)

      0  the model does no better than applying the nearest map it trained on,
         which needs no model at all
      1  the model is as accurate on an unseen map as on the ones it trained on,
         i.e. the training distribution stopped mattering

Both reference points are measured on the same model and the same probe, so the
rising clamping baseline, its residual concavity, the different output floors of
the two arms (bins quantises, scalar does not) and the choice of probe distance
all cancel. The shaded envelope is the spread over six probe placements -- fixed
alpha at two distances, map-space arclength at two distances, and null-matched at
two targets -- and it is narrow, which is the evidence that the curve is a
property of the model rather than of the coordinate.

x is the fraction of the usable alpha range covered by the training band, which
is the analogue of the cap half-angle phi in Goddard et al. (arXiv:2506.05574).
"""
import glob
import re

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.maps import asym_map_vec, iterate_asym
from src.mapmetric import sigma_mu_of, param_of_sigma_mu

SPAN = 0.8                     # usable alpha range, [0.2, 1.0]
RS = np.linspace(0.125, 1.0, 60)
RS = RS[RS >= 0.25][::6]


def orbit(a, R, n=30, traj=150, burn=50, seed=3):
    rng = np.random.default_rng(seed)
    return np.concatenate([iterate_asym(rng.uniform(.05, .95), R, a, traj)[burn:]
                           for _ in range(n)])


def clamp(a_probe, a_edge):
    return float(np.mean([np.sqrt(np.mean(
        (asym_map_vec(x := orbit(a_probe, R), R, a_probe)
         - asym_map_vec(x, R, a_edge)) ** 2)) for R in RS]))


def runs(mode):
    return [np.load(d + "/eval_cont.npz") for d in
            sorted(glob.glob(f"runs_cont/asym_{mode}_w*_seed0"),
                   key=lambda s: float(re.search(r"_w([\d.]+)_", s).group(1)))]


def placements(z):
    """The six ways of choosing a probe alpha for one run."""
    e = float(z["band_lo"])
    s = sigma_mu_of("asym", e)
    out = [e - 0.20, e - 0.15,
           param_of_sigma_mu("asym", s - 6), param_of_sigma_mu("asym", s - 8)]
    for target in (0.070, 0.080):
        out.append(brentq(lambda t: clamp(t, e) - target, 0.20, e - 1e-3,
                          xtol=1e-4))
    return out


def adaptation(z, probe):
    keep = z["R_grid"] >= 0.25
    curve = np.nanmean(z["implied_rms"][:, keep], axis=1)
    in_band = float(np.nanmean(z["implied_rms"][z["in_band"]][:, keep]))
    model = float(np.interp(probe, z["p_grid"], curve))
    c = clamp(probe, float(z["band_lo"]))
    return 100 * (c - model) / (c - in_band)


plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(figsize=(6.6, 5.0))

ax.axhline(100, color="0.55", lw=1.2, ls=(0, (5, 3)))
ax.axhline(0, color="0.55", lw=1.2, ls=(0, (5, 3)))
ax.text(0.093, 101.5, "as accurate off-distribution as on it", ha="left",
        va="bottom", fontsize=9.5, color="0.4")
ax.text(0.093, -2.0, "no better than the nearest trained map", ha="left",
        va="top", fontsize=9.5, color="0.4")

for mode, mfc, label in (("scalar", "k", "scalar output"),
                         ("bins", "white", "binned output")):
    Z = runs(mode)
    x = np.array([float(z["band_width"]) for z in Z]) / SPAN
    a = np.array([[adaptation(z, p) for p in placements(z)] for z in Z])
    ax.fill_between(x, a.min(1), a.max(1), color="k", alpha=0.11, lw=0)
    ax.plot(x, np.median(a, 1), "o-", color="k", mfc=mfc, ms=7, lw=1.5,
            label=label)
    print(f"{mode:7s} " + "  ".join(f"{v:.0f}%" for v in np.median(a, 1)))

ax.set_xlim(0.08, 0.80); ax.set_ylim(-12, 116)
ax.set_xticks([0.125, 0.25, 0.375, 0.50, 0.625, 0.75])
ax.set_xticklabels(["12%", "25%", "38%", "50%", "62%", "75%"])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("fraction of the map family seen in training")
ax.set_ylabel("adaptation to the unseen map")
ax.legend(frameon=False, fontsize=11, loc="lower right",
          bbox_to_anchor=(1.0, 0.06))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.5, -0.09,
         "Continuous input, asymmetric family. Shading spans six ways of placing the probe,\n"
         "so the curves are not an artefact of how off-distribution distance is measured.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_cont/adaptation.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_cont/adaptation.png")
