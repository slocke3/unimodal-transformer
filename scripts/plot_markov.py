"""Concept-shift results: does mixed-order training extrapolate to unseen orders?

For each arm, next-token NLL against the test order, alongside two EXACT
references (Dirichlet-multinomial conjugacy, not approximations):

  full Bayes        marginalises over all admissible orders -- the target
  restricted Bayes  marginalises only over the arm's TRAINING orders, so on an
                    unseen order it must explain the data with a rule form it
                    wrongly believes is right. This is the concept-shift
                    analogue of a mis-specified prior, and the line to beat.

Beating restricted Bayes on an order never trained on is generalization the
training distribution does not license -- and it is non-compositional: the same
suffix-matching routine applied at greater depth, not primitives recombined.
"""
import json, argparse
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", default="runs_markov")
ap.add_argument("--out_dir", default="figures_markov")
a = ap.parse_args()

arms = defaultdict(list)
for p in sorted(Path(a.runs_dir).glob("*/results.json")):
    d = json.load(open(p))
    arms[p.parent.name.rsplit("_seed", 1)[0]].append(d)
if not arms:
    raise SystemExit(f"no results under {a.runs_dir}")

order = ["fixed_k1", "fixed_k2", "fixed_k3", "mixed_12", "mixed_123",
         "mixed_1234", "mixed_13"]
names = [k for k in order if k in arms] + [k for k in arms if k not in order]

print("%12s %6s %9s %10s %12s %11s %s"
      % ("arm", "test k", "in train", "model", "restricted", "full", "verdict"))
summary = {}
for name in names:
    runs = arms[name]
    ks = sorted(int(k) for k in runs[0]["results"])
    tr = runs[0]["args"]["train_orders"]
    rows = {}
    for k in ks:
        m = np.mean([r["results"][str(k)]["model"] for r in runs])
        sd = np.std([r["results"][str(k)]["model"] for r in runs])
        rr = np.mean([r["results"][str(k)]["restricted"] for r in runs])
        ff = np.mean([r["results"][str(k)]["full"] for r in runs])
        rows[k] = (m, sd, rr, ff)
        tag = ""
        if k not in tr:
            tag = "BEATS restricted" if m < rr else "clamps"
        print("%12s %6d %9s %10.4f %12.4f %11.4f %s"
              % (name, k, "yes" if k in tr else "NO", m, rr, ff, tag))
    summary[name] = {"ks": ks, "rows": rows, "train": tr, "n_seed": len(runs)}
    print()

n = len(names)
ncol = min(4, n); nrow = -(-n // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.9 * nrow),
                         squeeze=False, sharey=True)
for ax, name in zip(axes.ravel(), names):
    s = summary[name]; ks = s["ks"]
    m = np.array([s["rows"][k][0] for k in ks])
    sd = np.array([s["rows"][k][1] for k in ks])
    rr = np.array([s["rows"][k][2] for k in ks])
    ff = np.array([s["rows"][k][3] for k in ks])
    ax.plot(ks, ff, "-", color="#2E7D32", lw=2.0, label="full Bayes (target)")
    ax.plot(ks, rr, "--s", color="#111111", lw=1.8, ms=5, mfc="white",
            label="restricted Bayes (line to beat)")
    ax.errorbar(ks, m, yerr=sd, fmt="-o", color="#D85A30", lw=2.2, ms=6,
                capsize=3, label="transformer", zorder=5)
    for k in s["train"]:
        ax.axvspan(k - .35, k + .35, color="#1B2A4A", alpha=0.10, lw=0)
    ax.set_title("%s   train={%s}  (%d seeds)"
                 % (name, ",".join(map(str, s["train"])), s["n_seed"]),
                 fontsize=10)
    ax.set_xlabel("test order $k$"); ax.grid(alpha=.25, lw=.4)
    ax.set_xticks(ks)
for ax in axes.ravel()[n:]:
    ax.set_visible(False)
for row in axes:
    row[0].set_ylabel("next-token NLL (nats)")
axes[0][0].legend(fontsize=7.5)
fig.suptitle("Concept shift: generalizing to unseen ORDERS of dependence.  "
             "Shaded = orders seen in training.\nBeating the dashed line on an "
             "unshaded order is generalization the training distribution does "
             "not license.", fontsize=11.5)
fig.tight_layout()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for e in ("png", "pdf"):
    fig.savefig(out / f"markov_concept_shift.{e}", dpi=160, bbox_inches="tight")
print(f"wrote {out}/markov_concept_shift.png")
