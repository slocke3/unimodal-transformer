"""Figure-3 analogue: held-out loss vs task diversity, against the best
predictor that knows only the pretraining distribution.

Mirrors Figure 3 of Goddard et al. (arXiv:2506.05574). The x-axis is the
training band's coverage of the usable alpha span; the loss is measured on the
permanently held-out region alpha in [0.20, 0.30], which no model in the sweep
ever trains on, so the same tasks are scored at every coverage level.

The line to beat is the band-restricted Bayes predictor, minimised over Markov
orders 1-4 so it is the strongest in-task-distribution reference available.
Any gap below it is generalization the pretraining task distribution does not
license, and it can be non-zero well before the transition.
"""
import argparse
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
lo, hi = B["heldout_alphas"].min(), B["heldout_alphas"].max()
R_eval, orders = B["R_eval"], B["orders"]
bw, bb = B["band_widths"], B["band_bayes"]          # bb: (n_order, n_w)
oracle, ctx_only = B["oracle"], B["context_only"]   # per order

rows = []
for p in sorted(Path(a.runs_dir).glob("asym_w*/eval_asym.npz")):
    z = np.load(p)
    al, R, ce = z["alpha_grid"], z["R_grid"], z["ce_final"]
    ia = (al >= lo - 1e-9) & (al <= hi + 1e-9)
    if not ia.any():
        print(f"  skip {p.parent.name}: eval grid does not reach [{lo},{hi}]")
        continue
    iR = np.array([int(np.argmin(abs(R - r))) for r in R_eval])
    rows.append({"w": float(z["band_width"]),
                 "ce": float(ce[np.ix_(ia, iR)].mean())})
if not rows:
    raise SystemExit(f"no usable runs under {a.runs_dir}")
rows.sort(key=lambda r: r["w"])
ws = np.array([r["w"] for r in rows]); tf = np.array([r["ce"] for r in rows])

cov = lambda x: x / a.alpha_span * 100
best_bb = bb.min(axis=0)                      # strongest reference per w
bb_at = np.array([best_bb[int(np.argmin(abs(bw - w)))] for w in ws])

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))

ax = axes[0]
greys = plt.get_cmap("Greys")
for i, k in enumerate(orders):
    ax.plot(cov(bw), bb[i], "--", lw=1.1, alpha=0.75,
            color=greys(0.35 + 0.15 * i), label=f"band Bayes, order {k}")
ax.plot(cov(bw), best_bb, "--", color="#111111", lw=2.2, marker="s", ms=4,
        mfc="white", label="band Bayes, best order  (line to beat)")
ax.plot(cov(ws), tf, "-o", color="#D85A30", lw=2.2, ms=5.5,
        label="Transformer", zorder=5)
ax.axhline(ctx_only.min(), color="#7B1FA2", ls=":", lw=1.5,
           label=f"context-only Markov (best order)")
ax.axhline(oracle.min(), color="#2E7D32", ls="-.", lw=1.5,
           label="oracle Markov (knows the true map)")
ax.axvspan(62, 72, color="steelblue", alpha=0.12, lw=0)
ax.set_xlabel(r"Training-band coverage of usable $\alpha$ span (%)")
ax.set_ylabel(rf"CE on held-out $\alpha \in [{lo:.2f}, {hi:.2f}]$ (nats)")
ax.set_title("Held-out loss vs task diversity", fontsize=11)
ax.grid(alpha=0.25, lw=0.4); ax.legend(fontsize=6.8, loc="best")

ax = axes[1]
gap = bb_at - tf
ax.axhline(0, color="#111111", lw=1.4, ls="--")
ax.plot(cov(ws), gap, "-o", color="#D85A30", lw=2.2, ms=5.5)
ax.fill_between(cov(ws), 0, gap, where=gap > 0, color="#D85A30", alpha=0.18)
ax.axvspan(62, 72, color="steelblue", alpha=0.12, lw=0)
ax.set_xlabel(r"Training-band coverage of usable $\alpha$ span (%)")
ax.set_ylabel("Advantage over band-restricted Bayes (nats)")
ax.set_title("Generalization the pretraining distribution\ndoes not license"
             " (shaded band: predicted transition)", fontsize=11)
ax.grid(alpha=0.25, lw=0.4)

fig.suptitle("Out-of-task-distribution generalization on the asymmetric map "
             "family", fontsize=12)
fig.tight_layout()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"figure3_analogue.{ext}", dpi=160, bbox_inches="tight")

print("%7s %10s %13s %14s %10s" % ("w", "coverage", "transformer",
                                   "band Bayes", "advantage"))
for w, t, b in zip(ws, tf, bb_at):
    print("%7g %9.0f%% %13.4f %14.4f %10.4f%s"
          % (w, cov(w), t, b, b - t, "  <-- beats it" if t < b else ""))
print(f"\nwrote {out}/figure3_analogue.png")
