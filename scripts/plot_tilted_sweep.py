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

The headline quantity is error relative to the IDEAL BINNED PREDICTOR. The
model sees only the bin of x_n, so the best it can do is the bin-conditional
mean m*(j) = E[f(x) | x in bin j]; measured against f(bin centre), that ideal
predictor scores RMS(m* - f(centre)), which is the reference used here (see
src/binfloor.py). An earlier version normalised by the RMS spread of f across a
bin, which also contains the within-bin variance and is therefore too large --
good models scored 0.68 against it, i.e. "better than optimal", which is what
exposed the error. The reference is absolute:
a model that clamps to the edge of its training band cannot reach it on a
held-out task, while genuine in-context identification can. That sidesteps the
"is it closer to the truth or to the band edge" question entirely, which is
where the previous analysis went wrong.

Both readings are shown, because they answer different questions and the
difference is itself informative:

  fixed distance   probes at s = +/-(w + d). The band edge is always d away, so
                   a widening band does not make its own test easier. This is
                   the confound-free reading of "does it generalize".
  position swept   loss at EVERY out-of-band s, the classic view. Richer, but
                   as w grows the surviving out-of-band positions are the ones
                   closest to the edge, so improvement here mixes "the model got
                   better" with "the remaining test got easier". Reading the two
                   side by side makes that visible rather than hidden.

Reads eval_asym.npz only; torch-free.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R_TRIVIAL = 0.25          # below this the map is subcritical; orbits die
_REF_CACHE = {}


def ideal_ref(p_grid, R_grid, n_bins, family):
    """RMS(m* - f(centre)) per cell; computed once and reused across runs."""
    key = (len(p_grid), float(p_grid[0]), float(p_grid[-1]),
           len(R_grid), float(R_grid[0]), float(R_grid[-1]), n_bins, family)
    if key not in _REF_CACHE:
        sys.path.insert(0, ".")
        from src.binfloor import ideal_bin_reference
        print("computing the ideal-binned-predictor reference ...", flush=True)
        _REF_CACHE[key] = ideal_bin_reference(p_grid, R_grid, n_bins, family)
    return _REF_CACHE[key]


def load(root):
    runs = defaultdict(list)
    for p in sorted(Path(root).glob("asym_w*/eval_asym.npz")):
        z = np.load(p)
        keep = z["R_grid"] >= R_TRIVIAL
        ib, hm = z["in_band_alpha"], z["heldout_mask"]
        fam = str(z["family"]) if "family" in z.files else "asym"
        ref = ideal_ref(z["alpha_grid"], z["R_grid"],
                        int(json.load(open(p.parent / "params.json"))["n_bins"]),
                        fam)
        ratio = z["implied_rms"] / np.maximum(ref, 1e-12)
        runs[float(z["band_width"])].append({
            "seed": int(json.load(open(p.parent / "params.json"))["seed"]),
            "band_lo": float(z["band_lo"]), "band_hi": float(z["band_hi"]),
            "s": z["alpha_grid"], "in_band": ib, "held": hm,
            "ce_in": float(z["ce_final"][ib][:, keep].mean()),
            "ce_held": float(z["ce_final"][hm][:, keep].mean()) if hm.any() else np.nan,
            "r_in": float(np.nanmean(ratio[ib][:, keep])),
            "r_held": float(np.nanmean(ratio[hm][:, keep])) if hm.any() else np.nan,
            "ratio_by_s": np.nanmean(ratio[:, keep], axis=1),
            "ce_by_s": np.nanmean(z["ce_final"][:, keep], axis=1),
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

    fig, axes2 = plt.subplots(2, 3, figsize=(16.4, 9.4))
    axes = axes2[0]

    ax = axes[0]
    ax.errorbar(cov, r_ho, yerr=r_ho_sd, fmt="-o", color="#D85A30", lw=2.1,
                ms=6, capsize=3, label="held out (fixed distance past the edge)")
    ax.errorbar(cov, r_in, yerr=r_in_sd, fmt="-s", color="#1B2A4A", lw=1.8,
                ms=5, capsize=3, mfc="white", label="in band")
    ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.8)
    ax.text(cov[0], 1.06, "ideal binned predictor = 1.0", fontsize=8, color="#2E7D32")
    ax.set_yscale("log")
    ax.set_xlabel(r"Training-band coverage of $s\in[-1.2,1.2]$ (%)")
    ax.set_ylabel("Implied-map RMS / ideal binned predictor")
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
    ax.set_ylabel("Implied-map RMS / ideal binned predictor")
    ax.set_title("Identification quality across the whole family\n"
                 "(seed 0; flat at 1.0 would be full generalization)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=6.5, ncol=2)

    # ---------- row 2: the position-swept ("old style") reading -----------
    cmap = plt.get_cmap("viridis")
    s_grid = runs[ws[0]][0]["s"]

    def mean_over_seeds(w, key):
        return np.mean([r[key] for r in runs[w]], axis=0)

    ax = axes2[1][0]
    for i, w in enumerate(ws):
        c = cmap(i / max(1, len(ws) - 1))
        ax.plot(s_grid, mean_over_seeds(w, "ce_by_s"), lw=1.6, color=c,
                label=f"w={w:g}")
        for edge in (-w, w):
            ax.axvline(edge, color=c, lw=0.7, ls=":", alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlabel("$s$   (dotted lines mark each model's band edges)")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.set_title("Position-swept: loss at every $s$\n"
                 "(the classic view)", fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes2[1][1]
    for i, w in enumerate(ws):
        c = cmap(i / max(1, len(ws) - 1))
        ax.plot(s_grid, mean_over_seeds(w, "ratio_by_s"), lw=1.6, color=c,
                label=f"w={w:g}")
    ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("$s$")
    ax.set_ylabel("Implied-map RMS / ideal binned predictor")
    ax.set_title("Position-swept: identification quality at every $s$\n"
                 "(1.0 = optimal for a binned model)", fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=6.5, ncol=2)

    # fixed OOD POSITIONS: the reading that conflates distance with generalization
    ax = axes2[1][2]
    targets = [t for t in (0.7, 0.9, 1.1) if t <= s_grid.max() + 1e-9]
    for k, t in enumerate(targets):
        j = int(np.argmin(np.abs(s_grid - t)))
        vals, covs = [], []
        for i, w in enumerate(ws):
            if t <= w + 1e-9:            # in band for this model: not a test
                continue
            vals.append(mean_over_seeds(w, "ratio_by_s")[j])
            covs.append(cov[i])
        ax.plot(covs, vals, "-o", ms=5, lw=1.8,
                color=plt.get_cmap("plasma")(k / max(1, len(targets) - 1)),
                label=f"fixed position $s$={s_grid[j]:.2f}")
    ax.errorbar(cov, r_ho, yerr=r_ho_sd, fmt="--s", color="#444444", lw=1.6,
                ms=5, capsize=3, mfc="white",
                label="fixed DISTANCE (confound-free)")
    ax.axhline(1.0, color="#2E7D32", ls="--", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel(r"Training-band coverage (%)")
    ax.set_ylabel("Implied-map RMS / ideal binned predictor")
    ax.set_title("Fixed position vs fixed distance\n"
                 "curves ending early = that $s$ fell inside the band",
                 fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7)

    fig.suptitle("Tilted-logistic family: band symmetric about logistic. "
                 "Top row: probes at fixed DISTANCE past the edge.  "
                 "Bottom row: swept over all out-of-band POSITIONS.",
                 fontsize=12)
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
