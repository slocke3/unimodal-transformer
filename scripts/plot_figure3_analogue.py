"""Figure-3 analogue: held-out loss vs task diversity, against the best
predictor that knows only the pretraining distribution.

Mirrors Figure 3 of Goddard et al. (arXiv:2506.05574), which plots far-OOD test
loss against task diversity and shows the transformer beating the optimal
in-task-distribution Bayesian solution. Here the x-axis is the training band
half-width w (equivalently its coverage of the usable alpha span), and the loss
is measured on the permanently held-out region alpha in [0.20, 0.30] that no
model in the sweep ever trains on.

The gap between the transformer and band_bayes is the quantity of interest: it
is out-of-distribution generalization that the pretraining task distribution
does not license, and it can be non-zero well before the transition.
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", default="runs_asym2")
ap.add_argument("--baseline", default="figures_asym2/band_bayes_baseline.npz")
ap.add_argument("--out_dir", default="figures_asym2")
ap.add_argument("--alpha_span", type=float, default=0.8)
a = ap.parse_args()

B = np.load(a.baseline)
held_lo, held_hi = B["heldout_alphas"].min(), B["heldout_alphas"].max()
R_eval = B["R_eval"]

rows = []
for p in sorted(Path(a.runs_dir).glob("asym_w*/eval_asym.npz")):
    z = np.load(p)
    al, R, ce = z["alpha_grid"], z["R_grid"], z["ce_final"]
    ia = (al >= held_lo - 1e-9) & (al <= held_hi + 1e-9)
    iR = np.array([np.argmin(abs(R - r)) for r in R_eval])
    rows.append({"w": float(z["band_width"]),
                 "ce": float(ce[np.ix_(ia, iR)].mean())})
rows.sort(key=lambda r: r["w"])
if not rows:
    raise FileNotFoundError(f"no runs under {a.runs_dir}")

ws = np.array([r["w"] for r in rows]); tf = np.array([r["ce"] for r in rows])
bw, bb = B["band_widths"], B["band_bayes"]
keep = np.isin(bw, ws)

fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.plot(bw[keep] / a.alpha_span * 100, bb[keep], "--", color="#444444",
        lw=1.8, marker="s", ms=4, mfc="white",
        label="Band-restricted Bayes (knows only the\ntraining band) — the line to beat")
ax.axhline(B["context_only"], color="#7B1FA2", ls=":", lw=1.5,
           label="Context-only Markov-1 (no task prior)")
ax.axhline(B["oracle"], color="#2E7D32", ls="-.", lw=1.5,
           label="Oracle Markov-1 (knows the true map)")
ax.plot(ws / a.alpha_span * 100, tf, "-o", color="#D85A30", lw=2.0, ms=5,
        label="Transformer")
ax.axvspan(62, 72, color="gray", alpha=0.15, lw=0)
ax.text(67, ax.get_ylim()[1] * 0.96, "predicted\ntransition", fontsize=7,
        ha="center", va="top", color="#555555")
ax.set_xlabel(r"Training-band coverage of the usable $\alpha$ span (%)")
ax.set_ylabel("Cross-entropy on held-out "
              rf"$\alpha \in [{held_lo:.2f}, {held_hi:.2f}]$ (nats)")
ax.set_title("Out-of-distribution generalization vs task diversity\n"
             "gap below the dashed line is generalization the pretraining\n"
             "distribution does not license", fontsize=10.5)
ax.grid(alpha=0.25, lw=0.4); ax.legend(fontsize=7.5)
fig.tight_layout()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"figure3_analogue.{ext}", dpi=160, bbox_inches="tight")
print("%7s %10s %14s %10s" % ("w", "coverage", "transformer", "band_bayes"))
for w, t in zip(ws, tf):
    j = int(np.argmin(abs(bw - w)))
    print("%7g %9.0f%% %14.4f %10.4f%s"
          % (w, 100 * w / a.alpha_span, t, bb[j],
             "   <- beats it" if t < bb[j] else ""))
print(f"\nwrote {out}/figure3_analogue.png")
