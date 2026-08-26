"""Why marginal occupancy overlap is the wrong statistic for a next-token task.

Along the conjugacy curve R = 1, every alpha <= 1 gives a map with the same
kneading sequence as logistic r = 4, so the symbolic dynamics are shared and
only the invariant measure moves. Two ways of measuring "how far has the
representation moved" disagree sharply there:

  marginal   overlap of the 1-point occupancy histograms on the uniform partition
  transition overlap of the order-1 joint distribution of (bin, next bin)

The marginal barely moves -- every map on the curve is full-chaos and piles its
density up at the endpoints -- while the transition distribution moves almost
entirely. Since the model is trained to predict the next token, the transition
statistic is the one that governs the loss, and the marginal badly understates
the shift. This figure makes that concrete.

Model-free and torch-free: this is a property of the map family, so it can be
produced without any trained checkpoint.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.maps import iterate_asym


def orbit_tokens(R, alpha, n_bins, n_steps=400_000, burn_in=2_000, seed=3):
    x = np.random.default_rng(seed).uniform(0.2, 0.8)
    traj = iterate_asym(x, R, alpha, burn_in + n_steps)[burn_in:]
    return np.clip(np.floor(traj * n_bins).astype(np.int64), 0, n_bins - 1)


def marginal(tok, n_bins):
    c = np.bincount(tok, minlength=n_bins).astype(float)
    return c / c.sum()


def transition(tok, n_bins):
    j = np.bincount(tok[:-1] * n_bins + tok[1:],
                    minlength=n_bins * n_bins).astype(float)
    return (j / j.sum()).reshape(n_bins, n_bins)


def overlap(p, q):
    return float(np.minimum(p, q).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=1.0,
                    help="peak height; 1.0 is the conjugacy curve")
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--n_alpha", type=int, default=26)
    ap.add_argument("--alpha_lo", type=float, default=0.5)
    ap.add_argument("--show_alphas", type=float, nargs="+",
                    default=[1.0, 0.8, 0.6, 0.5])
    ap.add_argument("--out_dir", default="figures_asym")
    a = ap.parse_args()

    alphas = np.linspace(a.alpha_lo, 1.0, a.n_alpha)
    n_bins = a.n_bins

    toks = {al: orbit_tokens(a.R, al, n_bins) for al in alphas}
    margs = {al: marginal(toks[al], n_bins) for al in alphas}
    trans = {al: transition(toks[al], n_bins) for al in alphas}

    ref_m, ref_t = margs[alphas[-1]], trans[alphas[-1]]
    ov_m = np.array([overlap(margs[al], ref_m) for al in alphas])
    ov_t = np.array([overlap(trans[al], ref_t) for al in alphas])

    print(f"R = {a.R}, {n_bins} bins")
    print(f"{'alpha':>7} {'marginal ovl':>14} {'transition ovl':>16}")
    for al, m, t in zip(alphas, ov_m, ov_t):
        print(f"{al:>7.3f} {m:>14.3f} {t:>16.3f}")

    show = [al for al in a.show_alphas
            if np.isclose(alphas, al, atol=1e-9).any()]
    show = show or list(alphas[::max(1, len(alphas) // 4)])
    n_show = len(show)

    fig = plt.figure(figsize=(4.1 * max(n_show, 2), 8.4))
    gs = fig.add_gridspec(2, n_show, height_ratios=[1.0, 1.05], hspace=0.34,
                          wspace=0.28)

    # -- top left: the marginals, which barely move ------------------------
    ax = fig.add_subplot(gs[0, :max(1, n_show // 2)])
    centers = (np.arange(n_bins) + 0.5) / n_bins
    cmap = plt.get_cmap("plasma")
    for k, al in enumerate(show):
        ax.plot(centers, margs[al], lw=1.6,
                color=cmap(k / max(1, len(show) - 1)),
                label=rf"$\alpha$={al:g}  ($x_c$={al/2:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel("Invariant density (bin probability)")
    ax.set_title("1-point occupancy: nearly identical across $\\alpha$",
                 fontsize=10.5)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7.5)

    # -- top right: the two overlap curves ---------------------------------
    ax = fig.add_subplot(gs[0, max(1, n_show // 2):])
    ax.plot(alphas, ov_m, "o-", color="#1B2A4A", lw=1.9, ms=4,
            label="Marginal occupancy overlap")
    ax.plot(alphas, ov_t, "s-", color="#D85A30", lw=1.9, ms=4,
            label="Transition (order-1 joint) overlap")
    ax.fill_between(alphas, ov_t, ov_m, color="gray", alpha=0.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\alpha$   (peak position $x_c=\alpha/2$)")
    ax.set_ylabel(r"Overlap with $\alpha=1$")
    ax.set_title("The two statistics disagree sharply\n"
                 "(shaded: what the marginal hides)", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8)
    for al in show:
        if al != 1.0:
            i = int(np.argmin(np.abs(alphas - al)))
            ax.annotate(f"{ov_t[i]:.2f}", (alphas[i], ov_t[i]),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color="#D85A30")

    # -- bottom: the transition matrices, which move completely ------------
    vmax = max(trans[al].max() for al in show)
    vmin = vmax * 1e-5
    for k, al in enumerate(show):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(trans[al], origin="lower", cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       extent=[0, 1, 0, 1], aspect="auto")
        ax.axvline(al / 2, color="cyan", lw=0.9, ls="--")
        i = int(np.argmin(np.abs(alphas - al)))
        ax.set_title(rf"$\alpha$={al:g}   overlap={ov_t[i]:.2f}", fontsize=10)
        ax.set_xlabel("current bin ($x$)")
        if k == 0:
            ax.set_ylabel("next bin ($x'$)")
    fig.colorbar(im, ax=fig.axes[-n_show:], label="transition probability",
                 fraction=0.02, pad=0.01)

    fig.suptitle("Along the conjugacy curve $R=1$ the dynamics are fixed "
                 "(shared kneading sequence) and only the representation moves\n"
                 "Dashed line: the critical point $x_c=\\alpha/2$",
                 fontsize=11.5)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "family_overlap.npz", alphas=alphas, R=a.R,
             marginal_overlap=ov_m, transition_overlap=ov_t,
             marginals=np.array([margs[al] for al in alphas]))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"family_overlap.{ext}", dpi=160, bbox_inches="tight")
    print(f"\nwrote {out}/family_overlap.png")


if __name__ == "__main__":
    main()
