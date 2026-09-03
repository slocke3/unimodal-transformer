"""Analogue of Figure 2 of Goddard et al. (arXiv:2506.05574).

There: test error in bands at angular distance delta from the pretraining pole,
plotted against the cap half-angle phi. Curves for large delta stay high while
the solution is specialized, then collapse onto the floor once phi passes the
transition, which is the signature of a general-purpose solution.

Here the task space is one-dimensional. The pole is the logistic map (alpha = 1
for asym, s = 0 for tilted), so the analogue of delta is the distance of the
evaluation parameter from that base map, and the analogue of phi is the
training-band half-width w. A curve at distance delta is in-band exactly when
w >= delta, so each curve is expected to fall once w reaches its own delta --
that much is coverage. The transition claim is stronger: curves at delta LARGER
than any w trained on should also collapse.

y is implied-map RMS, averaged over R >= 0.25, which is comparable across
output modes in a way cross-entropy is not.
"""
import re, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAM = {"asym": {"pole": 1.0, "sign": -1.0, "span": 0.8},
       "tilted": {"pole": 0.0, "sign": 1.0, "span": 1.2}}


def load(fam, om):
    runs = {}
    for d in sorted(glob.glob(f"runs_cont/{fam}_{om}_w*_seed0"),
                    key=lambda s: float(re.search(r"_w([\d.]+)_", s).group(1))):
        z = np.load(d + "/eval_cont.npz")
        keep = z["R_grid"] >= 0.25
        runs[float(z["band_width"])] = (
            z["p_grid"], np.nanmean(z["implied_rms"][:, keep], axis=1))
    return runs


def panel(ax, fam, om, deltas, label):
    runs = load(fam, om)
    if not runs:
        return None
    ws = sorted(runs)
    cmap = plt.get_cmap("copper_r")
    for i, dl in enumerate(deltas):
        p_target = FAM[fam]["pole"] - dl if fam == "asym" else dl
        y = []
        for w in ws:
            g, r = runs[w]
            y.append(r[int(np.argmin(np.abs(g - p_target)))])
        ax.plot(ws, y, "-o", ms=4.5, lw=1.3,
                color=cmap(i / max(1, len(deltas) - 1)),
                mfc="white" if om == "bins" else None,
                label=f"{dl:g}")
    ax.set_yscale("log")
    ax.set_ylabel("implied-map RMS")
    ax.text(0.97, 0.92, label, transform=ax.transAxes, ha="right", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ws


plt.rcParams.update({"font.size": 11, "xtick.direction": "in",
                     "ytick.direction": "in"})
deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.6), sharex=True)

ws = panel(axes[0], "asym", "scalar", deltas, "scalar output")
panel(axes[1], "asym", "bins", deltas, "binned output")
axes[1].set_xlabel("training-band half-width  $w$")

# a curve is in-band once w >= its own delta; mark where that starts for the
# largest delta that any run covers, so coverage and transition are separable
axes[0].legend(title=r"$\delta = 1-\alpha$", frameon=False, fontsize=8,
               title_fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.suptitle("Distance from the logistic base map vs training diversity",
             fontsize=12.5)
fig.text(0.5, -0.02,
         "Each curve is a fixed evaluation distance $\\delta$ from the base map.\n"
         "A curve becomes in-band when $w \\geq \\delta$; a transition would show "
         "as curves with $\\delta$ beyond every trained $w$ collapsing too.",
         ha="center", fontsize=9, color="0.3")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_cont/fig2_analogue.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_cont/fig2_analogue.png   band widths:", ws)
