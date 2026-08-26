"""Read the representation-only effect off the R=1 curve of the asym sweep.

At R = 1 the critical orbit is x_c -> 1 -> 0, and the origin is repelling for
alpha < 1 (g ~ x^alpha there, so g'(0) = infinity). Every map on the curve
{(R=1, alpha) : alpha <= 1} therefore has the SAME kneading sequence as the
logistic map at r=4, and so the same symbolic dynamics -- the same set of
admissible token sequences under the kneading partition.

What does change along the curve is the invariant measure, and with it where
the orbit sits relative to the FIXED uniform partition of [0,1]. Topological
entropy is pinned at ln 2 while the SRB Lyapunov exponent slides (0.693 at
alpha=1 down to 0.646 at alpha=0.6), which is exactly the signature of the
measure moving under a fixed topology.

So cross-entropy variation along this curve cannot be a failure to infer the
dynamics: the dynamics are conjugate. It is the uniform binning moving under
the model. That makes the R=1 column a free control for separating

    "failed to generalize dynamically"   from   "the binning moved"

and this script extracts it, quantifies how much of the total out-of-band
degradation it accounts for, and checks the mechanism by correlating the CE
rise against the drift of the invariant density away from alpha=1.

Torch-free: reads eval_asym.npz plus numpy-only dynamics from src.maps.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.maps import iterate_asym, compute_lyapunov_asym

R_TRIVIAL = 0.25


def invariant_hist(R, alpha, n_bins, n_steps=200_000, burn_in=2_000, seed=3):
    """Histogram of the invariant measure on the uniform partition."""
    x = np.random.default_rng(seed).uniform(0.2, 0.8)
    traj = iterate_asym(x, R, alpha, burn_in + n_steps)[burn_in:]
    counts = np.bincount(
        np.clip(np.floor(traj * n_bins).astype(np.int64), 0, n_bins - 1),
        minlength=n_bins)
    return counts / counts.sum()


def overlap(p, q):
    """Histogram intersection: 1.0 = identical distributions."""
    return float(np.minimum(p, q).sum())


def transition_hist(R, alpha, n_bins, n_steps=200_000, burn_in=2_000, seed=3):
    """Joint distribution of (current bin, next bin) on the uniform partition.

    For a next-token task this is the statistic that matters: two maps can share
    a marginal occupancy while sending mass to completely different successors.
    The marginal overlap alone understates how far the representation has moved.
    """
    x = np.random.default_rng(seed).uniform(0.2, 0.8)
    traj = iterate_asym(x, R, alpha, burn_in + n_steps)[burn_in:]
    tok = np.clip(np.floor(traj * n_bins).astype(np.int64), 0, n_bins - 1)
    joint = np.bincount(tok[:-1] * n_bins + tok[1:],
                        minlength=n_bins * n_bins).astype(float)
    return joint / joint.sum()


def load_runs(root):
    runs = []
    for path in sorted(Path(root).glob("asym_w*/eval_asym.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        with np.load(path) as z:
            runs.append({
                "name": path.parent.name, "w": float(z["band_width"]),
                "alpha_lo": float(z["alpha_lo"]),
                "alpha": z["alpha_grid"].copy(), "R": z["R_grid"].copy(),
                "ce": z["ce_final"].copy(), "in_band": z["in_band_alpha"].copy(),
                "n_bins": int(params["n_bins"]),
            })
    runs.sort(key=lambda r: r["w"])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_asym")
    ap.add_argument("--out_dir", default="figures_asym")
    a = ap.parse_args()

    runs = load_runs(a.runs_dir)
    if not runs:
        raise FileNotFoundError(f"no asym runs under {a.runs_dir}")

    alpha = runs[0]["alpha"]
    n_bins = runs[0]["n_bins"]
    jR = int(np.argmin(np.abs(runs[0]["R"] - 1.0)))
    R_star = runs[0]["R"][jR]
    print(f"conjugacy curve at R = {R_star:.4f} ({len(runs)} runs, "
          f"{len(alpha)} alpha values)")

    # -- the family-level facts, computed once (torch-free) ------------------
    print("\nalpha    lambda_SRB    marginal-ovl   transition-ovl"
          "   (both vs alpha=1)")
    lam = np.array([compute_lyapunov_asym(R_star, av) for av in alpha])
    ref = invariant_hist(R_star, alpha[-1], n_bins)
    ov = np.array([overlap(invariant_hist(R_star, av, n_bins), ref)
                   for av in alpha])
    ref_t = transition_hist(R_star, alpha[-1], n_bins)
    ov_t = np.array([overlap(transition_hist(R_star, av, n_bins), ref_t)
                     for av in alpha])
    for av, lv, o, ot in list(zip(alpha, lam, ov, ov_t))[::max(1, len(alpha) // 8)]:
        print(f"{av:6.3f} {lv:12.4f} {o:20.3f} {ot:14.3f}")
    print(f"  topological entropy is ln 2 = {np.log(2):.4f} for every alpha "
          f"on this curve")

    # -- how much of the out-of-band loss is representation? -----------------
    print("\n%7s %11s %12s %12s %10s" % ("w", "OOB total", "OOB @R=1",
                                         "in-band @R=1", "repr share"))
    rows = []
    for run in runs:
        ib = run["in_band"]
        keep_R = run["R"] >= R_TRIVIAL
        oob_total = (run["ce"][~ib][:, keep_R].mean() if (~ib).any() else np.nan)
        col = run["ce"][:, jR]
        oob_curve = col[~ib].mean() if (~ib).any() else np.nan
        inb_curve = col[ib].mean() if ib.any() else np.nan
        # excess loss on new alpha, along the curve (pure representation)
        # versus over the whole plane (representation + dynamics)
        excess_curve = oob_curve - inb_curve
        inb_total = run["ce"][ib][:, keep_R].mean() if ib.any() else np.nan
        excess_total = oob_total - inb_total
        share = (excess_curve / excess_total
                 if np.isfinite(excess_total) and excess_total > 0 else np.nan)
        rows.append({"w": run["w"], "col": col, "ib": ib,
                     "excess_curve": excess_curve,
                     "excess_total": excess_total, "share": share})
        print("%7g %11.4f %12.4f %12.4f %9.1f%%"
              % (run["w"], oob_total, oob_curve, inb_curve, 100 * share))

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))
    cmap = plt.get_cmap("viridis")

    ax = axes[0]
    for i, row in enumerate(rows):
        c = cmap(i / max(1, len(rows) - 1))
        ax.plot(alpha, row["col"], "-o", ms=3, lw=1.5, color=c,
                label=f"w={row['w']:g}")
        edge = runs[i]["alpha_lo"]
        if edge < 1.0:
            ax.axvline(edge, color=c, lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel(r"$\alpha$  (peak position $x_c=\alpha/2$)")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.set_title(f"CE along the conjugacy curve $R={R_star:.2f}$\n"
                 "all maps share the logistic $r{=}4$ kneading sequence",
                 fontsize=10.5)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for i, row in enumerate(rows):
        ax.plot(ov_t, row["col"], "o", ms=4, alpha=0.8,
                color=cmap(i / max(1, len(rows) - 1)), label=f"w={row['w']:g}")
    ax.set_xlabel(r"Transition-distribution overlap with $\alpha=1$")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.set_title("Is the CE rise explained by transition mass moving\n"
                 "under the fixed uniform partition?", fontsize=10.5)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    ws = [r["w"] for r in rows]
    ax.plot(ws, [100 * r["share"] for r in rows], "o-", color="#D85A30", lw=1.8)
    ax.axhline(100, color="gray", ls=":", lw=1.0)
    ax.set_xlabel(r"Training band half-width $w$")
    ax.set_ylabel("Representation share of out-of-band excess (%)")
    ax.set_title("How much of the out-of-band loss is the binning\n"
                 "moving rather than a dynamical failure?", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.4)

    fig.suptitle("Representation-only control: the $R=1$ curve is a family of "
                 "topologically conjugate maps, so CE variation along it "
                 "cannot be a dynamical failure", fontsize=11.5)
    fig.tight_layout()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "conjugacy_curve.npz", alpha=alpha, R_star=R_star,
             lam=lam, density_overlap=ov, transition_overlap=ov_t,
             ce_curve=np.array([r["col"] for r in rows]),
             band_widths=np.array(ws),
             repr_share=np.array([r["share"] for r in rows]))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"conjugacy_curve.{ext}", dpi=160, bbox_inches="tight")
    print(f"\nwrote {out}/conjugacy_curve.png")


if __name__ == "__main__":
    main()
