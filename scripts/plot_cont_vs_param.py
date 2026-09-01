"""Loss against the family parameter, one curve per training band.

The same view as the right panel of figures_asym2/asym_generalization.png --
sweep the evaluation parameter across the whole family, one curve per trained
band, with each model's band edges marked -- but for the continuous-input runs
and using implied-map RMS rather than cross-entropy.

RMS is the comparable quantity here: the bins and scalar arms have different
losses (cross-entropy vs squared error) and different floors (output
quantisation vs none), so their loss curves cannot be put on one axis, while
the distance from the model's implied E[x_{n+1}|x_n] to the true map can.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", default="runs_cont")
ap.add_argument("--out_dir", default="figures_cont")
ap.add_argument("--name", default="cont_vs_param")
a = ap.parse_args()

groups = defaultdict(dict)
for p in sorted(Path(a.runs_dir).glob("*/eval_cont.npz")):
    z = np.load(p)
    keep = z["R_grid"] >= 0.25
    groups[(str(z["family"]), str(z["output_mode"]))][float(z["band_width"])] = {
        "p": z["p_grid"],
        "rms": np.nanmean(z["implied_rms"][:, keep], axis=1),
        "floor": float(np.nanmean(z["binning_floor"][:, keep])),
        "lo": float(z["band_lo"]), "hi": float(z["band_hi"])}
if not groups:
    raise SystemExit(f"no runs under {a.runs_dir}")

keys = sorted(groups, key=lambda k: (k[0], k[1]))
fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.8),
                         squeeze=False, sharey=True)
cmap = plt.get_cmap("viridis")
for ax, key in zip(axes[0], keys):
    fam, om = key
    d = groups[key]; ws = sorted(d)
    for i, w in enumerate(ws):
        c = cmap(i / max(1, len(ws) - 1))
        ax.plot(d[w]["p"], d[w]["rms"], lw=1.7, color=c, label=f"w={w:g}")
        for e in {d[w]["lo"], d[w]["hi"]}:
            if d[w]["p"].min() < e < d[w]["p"].max():
                ax.axvline(e, color=c, lw=0.8, ls=":", alpha=0.65)
    fl = np.mean([d[w]["floor"] for w in ws])
    if fl > 0:
        ax.axhline(fl, color="#2E7D32", ls="--", lw=1.6)
        ax.text(d[ws[0]]["p"].min(), fl * 1.1, "output-quantisation floor",
                fontsize=7.5, color="#2E7D32")
    ax.set_yscale("log")
    ax.set_xlabel(r"Evaluation $\alpha$" if fam == "asym" else "Evaluation $s$")
    ax.set_title(f"{fam} — {om} output\n(dotted lines mark each band edge)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7, ncol=2)
axes[0][0].set_ylabel("Implied-map RMS  (mean over $R\\geq0.25$)")
fig.suptitle("Continuous input: identification error across the family, "
             "one curve per training band", fontsize=12)
fig.tight_layout()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for e in ("png", "pdf"):
    fig.savefig(out / f"{a.name}.{e}", dpi=160, bbox_inches="tight")
print(f"wrote {out}/{a.name}.png  ({len(keys)} panels: {keys})")
