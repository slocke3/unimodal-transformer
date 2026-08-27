"""Tilted-logistic sweep: does identification extend past the training band?

Two things this design fixes relative to the asymmetric-family sweep.

1. The band is SYMMETRIC about the logistic map (`s = 0`), because both
   endpoints of `f_s(x) = C x(1-x)(1 + s(x-1/2))` stay regular for every `s` —
   no superattracting origin forcing a one-sided range. So every model has
   held-out regions on both sides of its training band.

2. The held-out probes sit at a FIXED DISTANCE `d` beyond each band edge, at
   `s = +/-(w + d)`, rather than at fixed absolute positions. A widening band
   therefore does not bring its own test closer, which is the confound that
   made the asymmetric sweep's falling loss uninterpretable.

The headline quantity is error relative to the BINNING FLOOR. The floor is the
RMS error that bin quantisation alone forces, so it is an absolute reference:
a model that clamps to the edge of its training band cannot reach it on a
held-out task, while genuine in-context identification can. That sidesteps the
"is it closer to the truth or to the band edge" question entirely, which is
where the previous analysis went wrong.

Reads eval_asym.npz only; torch-free.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_TRIVIAL = 0.25          # below this the map is subcritical; orbits die


def load(root):
    runs = defaultdict(list)
    for p in sorted(Path(root).glob("asym_w*/eval_asym.npz")):
        z = np.load(p)
        keep = z["R_grid"] >= R_TRIVIAL
        ib, hm = z["in_band_alpha"], z["heldout_mask"]
        ratio = z["implied_rms"] / z["binning_floor"]
        runs[float(z["band_width"])].append({
            "seed": int(json.load(open(p.parent / "params.json"))["seed"]),
            "band_lo": float(z["band_lo"]), "band_hi": float(z["band_hi"]),
            "s": z["alpha_grid"], "in_band": ib, "held": hm,
            "ce_in": float(z["ce_final"][ib][:, keep].mean()),
            "ce_held": float(z["ce_final"][hm][:, keep].mean()) if hm.any() else np.nan,
            "r_in": float(np.nanmean(ratio[ib][:, keep])),
            "r_held": float(np.nanmean(ratio[hm][:, keep])) if hm.any() else np.nan,
            "ratio_by_s": np.nanmean(ratio[:, keep], axis=1),
        })
    if not runs:
        raise FileNotFoundError(f"no runs under {root}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_tilt")
    ap.add_argument("--out_dir", default="figures_tilt")
    ap.add_argument("--s_span", type=float, default=2.4)
    a = ap.parse_args()

    runs = load(a.runs_dir)
    ws = sorted(runs)
    cov = np.array([200.0 * w / a.s_span for w in ws])

    def agg(key):
        m = np.array([np.mean([r[key] for r in runs[w]]) for w in ws])
        sd = np.array([np.std([r[key] for r in runs[w]]) for w in ws])
        return m, sd

    ce_in, ce_in_sd = agg("ce_in")
    ce_ho, ce_ho_sd = agg("ce_held")
    r_in, r_in_sd = agg("r_in")
    r_ho, r_ho_sd = agg("r_held")

    print("%6s %9s %7s | %9s %9s | %9s %9s"
          % ("w", "coverage", "seeds", "CE in", "CE held", "in/floor",
             "held/floor"))
    for i, w in enumerate(ws):
        print("%6g %8.0f%% %7d | %9.4f %9.4f | %9.2f %9.2f"
              % (w, cov[i], len(runs[w]), ce_in[i], ce_ho[i], r_in[i], r_ho[i]))

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8))

    ax = axes[0]
    ax.errorbar(cov, r_ho, yerr=r_ho_sd, fmt="-o", color="#D85A30", lw=2.1,
                ms=6, capsize=3, label="held out (fixed distance past the edge)")
    ax.errorbar(cov, r_in, yerr=r_in_sd, fmt="-s", color="#1B2A4A", lw=1.8,
                ms=5, capsize=3, mfc="white", label="in band")
    ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.8)
    ax.text(cov[0], 1.06, "binning floor — optimal", fontsize=8, color="#2E7D32")
    ax.set_yscale("log")
    ax.set_xlabel(r"Training-band coverage of $s\in[-1.2,1.2]$ (%)")
    ax.set_ylabel("Implied-map RMS / binning floor")
    ax.set_title("Does identification reach the floor off-band?\n"
                 "clamping to the band edge cannot", fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(cov, ce_ho, yerr=ce_ho_sd, fmt="-o", color="#D85A30", lw=2.1,
                ms=6, capsize=3, label="held out")
    ax.errorbar(cov, ce_in, yerr=ce_in_sd, fmt="-s", color="#1B2A4A", lw=1.8,
                ms=5, capsize=3, mfc="white", label="in band")
    ax.axhline(np.log(64), color="gray", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Training-band coverage of $s\in[-1.2,1.2]$ (%)")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.set_title("Loss at a fixed distance past the band edge", fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=8)

    ax = axes[2]
    cmap = plt.get_cmap("viridis")
    for i, w in enumerate(ws):
        r0 = runs[w][0]
        ax.plot(r0["s"], r0["ratio_by_s"], lw=1.5,
                color=cmap(i / max(1, len(ws) - 1)), label=f"w={w:g}")
        ax.axvspan(r0["band_lo"], r0["band_hi"], color=cmap(i / max(1, len(ws) - 1)),
                   alpha=0.04, lw=0)
    ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("$s$   ($s=0$ is the logistic map)")
    ax.set_ylabel("Implied-map RMS / binning floor")
    ax.set_title("Identification quality across the whole family\n"
                 "(seed 0; flat at 1.0 would be full generalization)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=6.5, ncol=2)

    fig.suptitle("Tilted-logistic family: band symmetric about logistic, "
                 "held-out probes at fixed distance past each edge", fontsize=12)
    fig.tight_layout()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tilted_sweep.{ext}", dpi=160, bbox_inches="tight")
    np.savez(out / "tilted_sweep.npz", band_widths=np.array(ws), coverage=cov,
             ce_in=ce_in, ce_held=ce_ho, ratio_in=r_in, ratio_held=r_ho,
             ratio_in_sd=r_in_sd, ratio_held_sd=r_ho_sd)
    print(f"\nwrote {out}/tilted_sweep.png")


if __name__ == "__main__":
    main()
