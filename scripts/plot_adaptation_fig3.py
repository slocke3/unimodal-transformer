"""Adaptation, in the form of Figure 3 of Goddard et al. (arXiv:2506.05574).

One panel per input type. In each: test error against training-task diversity,
with a dashed line for the best predictor that only knows the training
distribution. There the dashed line is the optimal in-task-distribution
Bayesian solution; here it is the clamping baseline -- the error you get by
applying the nearest map you actually trained on, which needs no model.

The reading is the same as theirs: curves BELOW the dashed line are doing
something the training distribution does not license.
"""
import re, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.maps import iterate_asym, asym_map_vec

NB = 64


def clamp(R, pp, pe, x):
    return float(np.sqrt(np.mean((asym_map_vec(x, R, pp) - asym_map_vec(x, R, pe))**2)))


def cont(om):
    def cg(pp, pe, Rs, L=50, traj=150, n=30, seed=3):
        rng = np.random.default_rng(seed)
        return float(np.mean([clamp(R, pp, pe, np.concatenate(
            [iterate_asym(rng.uniform(.05, .95), R, pp, traj)[L-1:-1]
             for _ in range(n)])) for R in Rs]))
    o = []
    for d in sorted(glob.glob(f"runs_cont/asym_{om}_w*_seed0"),
                    key=lambda s: float(re.search(r"_w([\d.]+)_", s).group(1))):
        z = np.load(d + "/eval_cont.npz")
        keep = z["R_grid"] >= 0.25; hm = z["heldout_mask"]
        e, pr = float(z["band_lo"]), float(z["p_grid"][hm][0])
        o.append((float(z["band_width"]),
                  float(np.nanmean(z["implied_rms"][hm][:, keep])),
                  cg(pr, e, z["R_grid"][keep][::6])))
    return list(zip(*o))


def token():
    p = np.load("figures_asym2/return_map_probe.npz")
    runs = sorted({re.sub(r"_(alpha|d_true|d_train|floor|band_lo|x_.*|e_.*)$", "", k)
                   for k in p.files if k.endswith("_band_lo")})
    o = []
    for r in runs:
        lo = float(p[f"{r}_band_lo"])
        cand = [(abs((lo - a_) / 2 - 0.1), a_, d_)
                for a_, d_ in zip(p[f"{r}_alpha"], p[f"{r}_d_true"]) if a_ < lo - 1e-9]
        if not cand:
            continue
        _, a_, d_ = min(cand)
        x = p[f"{r}_x_{a_:g}"]
        o.append((round(1 - lo, 2), float(d_), clamp(1.0, float(a_), lo, x),
                  (lo - a_) / 2 * NB))
    o.sort()
    return list(zip(*o))

sw, sm, sc = cont("scalar")
bw, bm, bc = cont("bins")
tw, tm, tc, td = token()

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

ax = axes[0]
ax.plot(sw, sc, "k--", lw=1.6, label="Clamp to nearest trained map*")
ax.plot(bw, bm, "o-", color="k", mfc="white", ms=7, lw=1.4, label="binned output")
ax.plot(sw, sm, "o-", color="k", mfc="k", ms=7, lw=1.4, label="scalar output")
ax.set_ylim(0, 0.105); ax.set_xlim(0.02, 0.68)
ax.set_xlabel("training-band half-width  $w$")
ax.set_ylabel("implied-map RMS at the probe")
ax.set_title("continuous input", fontsize=13)
ax.legend(frameon=False, fontsize=10, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
ax.plot(tw, tc, "k--", lw=1.6, label="Clamp to nearest trained map*")
ax.plot(tw, tm, "o-", color="k", mfc="k", ms=7, lw=1.4, label="binned input+output")
for w, m, dd in zip(tw, tm, td):
    ax.annotate(f"{dd:.1f} bins", (w, m), fontsize=8, color="0.35",
                textcoords="offset points", xytext=(5, 6))
ax.set_ylim(0, 0.33); ax.set_xlim(0.02, 0.78)
ax.set_xlabel("training-band half-width  $w$")
ax.set_ylabel("implied-map RMS at the probe")
ax.set_title("token input", fontsize=13)
ax.legend(frameon=False, fontsize=10, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.5, -0.04, "Curves below the dashed line generalize beyond the "
         "training distribution. Continuous input does; token input does not.\n"
         "*the best predictor using only maps seen in training — the analogue "
         "of the optimal in-task-distribution Bayes solution.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_cont/adaptation_fig3.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_cont/adaptation_fig3.png")
