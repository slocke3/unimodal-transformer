"""Continuous-input sweeps: does feeding x directly change the picture?

Every earlier map sweep fed bin tokens, so identifying the map meant
reconstructing a function from a quantised, unevenly sampled view of it. These
runs feed x directly, in two output modes, so the input change is separated
from the output one:

  bins    same output as before (logits over n_bins) -- only the INPUT differs
  scalar  squared error on a real prediction -- continuous end to end

The headline is implied-map RMS in ABSOLUTE units, because the reference
differs between modes: scalar has no floor at all (x_n is known exactly, so
nothing forces an error), while bins retains only the output-quantisation
residual RMS(centre(bin(f)) - f). Both are far below the old input-binning
floor, so cross-entropy here is NOT comparable to the token runs; implied-map
RMS against its own floor is the honest bridge.

Held-out probes sit at a fixed DISTANCE past each band edge, so a widening band
never brings its own test closer.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", default="runs_cont")
ap.add_argument("--out_dir", default="figures_cont")
a = ap.parse_args()

SPAN = {"asym": 0.8, "tilted": 2.4}
runs = defaultdict(dict)
for p in sorted(Path(a.runs_dir).glob("*/eval_cont.npz")):
    z = np.load(p)
    fam, om, w = str(z["family"]), str(z["output_mode"]), float(z["band_width"])
    keep = z["R_grid"] >= 0.25
    ib, hm = z["in_band"], z["heldout_mask"]
    runs[(fam, om)][w] = {
        "in": float(np.nanmean(z["implied_rms"][ib][:, keep])),
        "held": float(np.nanmean(z["implied_rms"][hm][:, keep])) if hm.any() else np.nan,
        "floor": float(np.nanmean(z["binning_floor"][ib][:, keep])),
        "loss_in": float(np.nanmean(z["loss_grid"][ib][:, keep])),
        "loss_held": float(np.nanmean(z["loss_grid"][hm][:, keep])) if hm.any() else np.nan,
        "p": z["p_grid"], "rms_by_p": np.nanmean(z["implied_rms"][:, keep], axis=1),
        "lo": float(z["band_lo"]), "hi": float(z["band_hi"])}
if not runs:
    raise SystemExit(f"no runs under {a.runs_dir}")

# --- convergence: is 30k steps enough? -------------------------------------
# These runs use 30k steps with all-position training, against 160k with
# last-position training in the token sweeps. More targets per step lowers
# gradient variance but is NOT a substitute for optimizer steps, so the tail of
# the training loss is the thing to check before comparing across sweeps.
print("Convergence check -- fractional drop over the last 20% of training:")
for p_ in sorted(Path(a.runs_dir).glob("*/params.json")):
    h = json.load(open(p_)).get("history") or []
    if len(h) < 4:
        continue
    tail = [d["loss"] for d in h[-4:]]
    drop = (tail[0] - tail[-1]) / max(abs(tail[0]), 1e-12)
    flag = "STILL IMPROVING" if drop > 0.05 else "flat"
    print("  %-34s final %.5f  drop %+6.1f%%  %s"
          % (p_.parent.name, tail[-1], 100 * drop, flag))
print()

keys = sorted(runs, key=lambda k: (k[0], k[1]))
print("%8s %8s %7s %9s %11s %11s %9s" % ("family","output","w","coverage",
      "in-band RMS","held-out RMS","floor"))
for fam, om in keys:
    for w in sorted(runs[(fam, om)]):
        d = runs[(fam, om)][w]
        cov = 100 * (w if fam == "asym" else 2 * w) / SPAN[fam]
        print("%8s %8s %7g %8.0f%% %11.5f %11.5f %9.5f"
              % (fam, om, w, cov, d["in"], d["held"], d["floor"]))
    print()

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
for col, fam in enumerate(["asym", "tilted"]):
    ax = axes[0][col]
    for om, c in (("bins", "#D85A30"), ("scalar", "#1B2A4A")):
        if (fam, om) not in runs: continue
        d = runs[(fam, om)]
        ws = sorted(d)
        cov = [100 * (w if fam == "asym" else 2 * w) / SPAN[fam] for w in ws]
        ax.plot(cov, [d[w]["held"] for w in ws], "-o", color=c, lw=2.1, ms=6,
                label=f"{om}: held out")
        ax.plot(cov, [d[w]["in"] for w in ws], "--s", color=c, lw=1.5, ms=5,
                mfc="white", alpha=.8, label=f"{om}: in band")
        if om == "bins":
            ax.axhline(np.mean([d[w]["floor"] for w in ws]), color="#2E7D32",
                       ls=":", lw=1.5, label="output-quantisation floor (bins)")
    ax.axhline(0, color="#2E7D32", ls="-", lw=1.0, alpha=.5)
    ax.set_yscale("log"); ax.grid(alpha=.25, which="both", lw=.4)
    ax.set_xlabel("Training-band coverage (%)")
    ax.set_ylabel("Implied-map RMS (absolute)")
    ax.set_title(f"{fam}: identification error vs coverage", fontsize=11)
    ax.legend(fontsize=7)

    ax = axes[1][col]
    cmap = plt.get_cmap("viridis")
    for om, ls in (("bins", "-"), ("scalar", "--")):
        if (fam, om) not in runs: continue
        d = runs[(fam, om)]; ws = sorted(d)
        for i, w in enumerate(ws):
            ax.plot(d[w]["p"], d[w]["rms_by_p"], ls, lw=1.4,
                    color=cmap(i / max(1, len(ws) - 1)),
                    label=f"{om} w={w:g}" if col == 0 else None)
    ax.set_yscale("log"); ax.grid(alpha=.25, which="both", lw=.4)
    ax.set_xlabel(r"$\alpha$" if fam == "asym" else "$s$")
    ax.set_ylabel("Implied-map RMS")
    ax.set_title(f"{fam}: swept over all positions\n(solid=bins, dashed=scalar)",
                 fontsize=10.5)
    if col == 0: ax.legend(fontsize=6, ncol=2)
fig.suptitle("Continuous input embeddings: x fed directly instead of binned.  "
             "Top: fixed-distance held-out probes.  Bottom: all positions.",
             fontsize=12)
fig.tight_layout()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for e in ("png", "pdf"):
    fig.savefig(out / f"cont_sweep.{e}", dpi=160, bbox_inches="tight")
print(f"wrote {out}/cont_sweep.png")
