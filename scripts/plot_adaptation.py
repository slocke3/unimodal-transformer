"""Adaptation to the shift: model against the clamping baseline.

The question is not whether held-out error is large, but whether it beats what
a model would score by simply applying its nearest trained map. That clamping
baseline needs no model: RMS between the true map at the probe and the
band-edge map, on the probe's own orbit points.

  adaptation = (clamp - model) / (clamp - floor)

  0%    indistinguishable from applying the band-edge map
  100%  reached the best a model of this output type could do
  <0%   WORSE than clamping -- the model's substitute is worse than the
        nearest map it actually trained on

Peak displacement is carried on the plots because it is what makes the two
input types comparable: every continuous run probes at a fixed 6.4 bin widths
of peak movement, while the token probes sat on a fixed alpha grid and landed
at 1.6-8.0 bins.
"""
import re, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.maps import iterate_asym, asym_map_vec

NB = 64
C = {"token": "#1B2A4A", "scalar": "#D85A30", "bins": "#2E7D32"}


def clamp(R, pp, pe, x):
    return float(np.sqrt(np.mean((asym_map_vec(x, R, pp) - asym_map_vec(x, R, pe))**2)))


def token_rows():
    p = np.load("figures_asym2/return_map_probe.npz")
    runs = sorted({re.sub(r"_(alpha|d_true|d_train|floor|band_lo|x_.*|e_.*)$", "", k)
                   for k in p.files if k.endswith("_band_lo")})
    out = []
    for r in runs:
        lo = float(p[f"{r}_band_lo"])
        for a_, d_, f_ in zip(p[f"{r}_alpha"], p[f"{r}_d_true"], p[f"{r}_floor"]):
            if a_ >= lo - 1e-9:
                continue
            x = p[f"{r}_x_{a_:g}"]
            c = clamp(1.0, float(a_), lo, x)
            out.append({"w": round(1 - lo, 2), "dxc": (lo - a_) / 2 * NB,
                        "m": float(d_), "c": c, "f": float(f_),
                        "adapt": 100 * (c - d_) / max(c - f_, 1e-12)})
    return out


def cont_rows(om):
    def cg(pp, pe, Rs, L=50, traj=150, n=30, seed=3):
        rng = np.random.default_rng(seed)
        return float(np.mean([clamp(R, pp, pe, np.concatenate(
            [iterate_asym(rng.uniform(.05, .95), R, pp, traj)[L-1:-1]
             for _ in range(n)])) for R in Rs]))
    out = []
    for d in sorted(glob.glob(f"runs_cont/asym_{om}_w*_seed0"),
                    key=lambda s: float(re.search(r"_w([\d.]+)_", s).group(1))):
        z = np.load(d + "/eval_cont.npz")
        keep = z["R_grid"] >= 0.25; hm = z["heldout_mask"]
        e, pr = float(z["band_lo"]), float(z["p_grid"][hm][0])
        m = float(np.nanmean(z["implied_rms"][hm][:, keep]))
        f = float(np.nanmean(z["binning_floor"][hm][:, keep]))
        c = cg(pr, e, z["R_grid"][keep][::6])
        out.append({"w": float(z["band_width"]), "dxc": (e - pr) / 2 * NB,
                    "m": m, "c": c, "f": f,
                    "adapt": 100 * (c - m) / max(c - f, 1e-12)})
    return out


tok, sca, bns = token_rows(), cont_rows("scalar"), cont_rows("bins")
fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.0))

# -- A: adaptation vs band width ------------------------------------------
a = ax[0]
a.axhspan(-260, 0, color="#B71C1C", alpha=0.07, lw=0)
a.axhline(0, color="#111111", lw=1.6)
a.text(0.02, 4, "clamping to the band edge", fontsize=8, color="#111111")
for rows, k, lbl, mk in ((sca, "scalar", "continuous · scalar out", "o"),
                         (bns, "bins", "continuous · bins out", "s")):
    a.plot([r["w"] for r in rows], [r["adapt"] for r in rows], mk + "-",
           color=C[k], lw=2.2, ms=7, label=lbl)
tw = [r["w"] for r in tok]; ta = [r["adapt"] for r in tok]
a.plot(tw, ta, "^", color=C["token"], ms=9, label="token input (all probes)")
for r in tok:
    a.annotate(f"{r['dxc']:.1f}", (r["w"], r["adapt"]), fontsize=6.5,
               color=C["token"], textcoords="offset points", xytext=(6, -3))
a.set_xlabel("Training-band half-width $w$")
a.set_ylabel("Adaptation:  (clamp $-$ model) / (clamp $-$ floor)   [%]")
a.set_title("Does the model beat clamping?\n"
            "token labels = peak displacement in bin widths", fontsize=10.5)
a.grid(alpha=.25, lw=.4); a.legend(fontsize=8, loc="lower right")

# -- B: the three absolute levels -----------------------------------------
a = ax[1]
for rows, k, lbl in ((sca, "scalar", "scalar"), (bns, "bins", "bins")):
    w = [r["w"] for r in rows]
    a.plot(w, [r["c"] for r in rows], ":", color=C[k], lw=1.6)
    a.plot(w, [r["m"] for r in rows], "o-", color=C[k], lw=2.2, ms=6,
           label=f"continuous {lbl}: model")
    if k == "bins":
        a.plot(w, [r["f"] for r in rows], "--", color=C[k], lw=1.4,
               label="bins output floor")
a.plot([r["w"] for r in tok], [r["m"] for r in tok], "^", color=C["token"],
       ms=9, label="token: model")
a.plot([r["w"] for r in tok], [r["c"] for r in tok], "v", color=C["token"],
       ms=8, mfc="white", label="token: clamp")
a.set_yscale("log")
a.set_xlabel("Training-band half-width $w$")
a.set_ylabel("Implied-map RMS (absolute)")
a.set_title("Model vs clamp vs floor\n(dotted = clamping baseline)", fontsize=10.5)
a.grid(alpha=.25, which="both", lw=.4); a.legend(fontsize=7)

# -- C: the displacement confound -----------------------------------------
a = ax[2]
a.axhline(0, color="#111111", lw=1.6)
a.axhspan(-260, 0, color="#B71C1C", alpha=0.07, lw=0)
for rows, k, lbl, mk in ((sca, "scalar", "continuous · scalar", "o"),
                         (bns, "bins", "continuous · bins", "s")):
    a.plot([r["dxc"] for r in rows], [r["adapt"] for r in rows], mk,
           color=C[k], ms=8, label=lbl)
a.plot([r["dxc"] for r in tok], [r["adapt"] for r in tok], "^",
       color=C["token"], ms=9, label="token input")
a.axvline(6.4, color="gray", ls="--", lw=1.2)
a.text(6.55, -200, "continuous probes\nall sit here", fontsize=7.5, color="gray")
a.set_xlabel("Peak displacement at the probe (bin widths)")
a.set_ylabel("Adaptation [%]")
a.set_title("Same axis, honestly: the token probes are spread\n"
            "over displacement, the continuous ones are not", fontsize=10.5)
a.grid(alpha=.25, lw=.4); a.legend(fontsize=8)

fig.suptitle("Adaptation to the parameter shift: continuous input crosses the "
             "clamping line, token input never does", fontsize=12.5)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_cont/adaptation.{e}", dpi=160, bbox_inches="tight")
print("wrote figures_cont/adaptation.png")
