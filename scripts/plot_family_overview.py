"""Descriptive overview of the logistic family: the testbed figure.

  (a) cobweb diagram      -- the map, the diagonal, and one orbit in phase space
  (b) bifurcation diagram -- long-run states x_n against r
  (c) invariant measure   -- empirical histogram of visited states at r = 4,
                             against the analytic arcsine density
  (d) Lyapunov exponent   -- lambda(r), with lambda = 0 marked
      (b) and (d) share the r axis, so periodic windows line up with the dips
      of lambda below zero

Descriptive, not an experimental result: it justifies why varying r gives
controlled variation in dynamical behaviour and in state-visitation statistics.

Torch-free. Run from the repo root:
  python scripts/plot_family_overview.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.maps import iterate_map, compute_lyapunov_array  # noqa: E402


def orbit(r, n_keep, burn_in, x0=0.37):
    traj = np.asarray(iterate_map(x0, r, burn_in + n_keep))
    return traj[-n_keep:]


def cobweb_path(r, x0, n_iter):
    """Staircase polyline: (x,0) -> (x,f(x)) -> (f(x),f(x)) -> ..."""
    xs = np.empty(n_iter + 1)
    xs[0] = x0
    for i in range(n_iter):
        xs[i + 1] = r * xs[i] * (1.0 - xs[i])
    px = np.empty(2 * n_iter + 1)
    py = np.empty(2 * n_iter + 1)
    px[0], py[0] = x0, 0.0
    px[1::2], py[1::2] = xs[:-1], xs[1:]     # vertical legs
    px[2::2], py[2::2] = xs[1:], xs[1:]      # horizontal legs
    return px, py, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cob_r", type=float, default=3.16)
    ap.add_argument("--cob_x0", type=float, default=0.038)
    ap.add_argument("--cob_iters", type=int, default=500)
    ap.add_argument("--measure_r", type=float, default=4.0,
                    help="parameter for the invariant-measure panel (c)")
    ap.add_argument("--hist_bins", type=int, default=64,
                    help="histogram bins in (c); 64 matches the tokenizer")
    ap.add_argument("--r_lo", type=float, default=0.5)
    ap.add_argument("--r_hi", type=float, default=4.0)
    ap.add_argument("--r_marks", type=float, nargs="+", default=[3.16, 4.0])
    ap.add_argument("--marker_lines", action="store_true",
                    help="draw vertical guides at --r_marks in (b) and (d)")
    ap.add_argument("--n_r", type=int, default=1600)
    ap.add_argument("--n_plot", type=int, default=220)
    ap.add_argument("--burn_in", type=int, default=2000)
    ap.add_argument("--lam_steps", type=int, default=20000)
    ap.add_argument("--n_density", type=int, default=400_000)
    ap.add_argument("--out", default="figures_family/family_overview.pdf")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # (b) bifurcation
    r_grid = np.linspace(a.r_lo, a.r_hi, a.n_r)
    bif_r, bif_x = [], []
    for r in r_grid:
        x = orbit(r, a.n_plot, a.burn_in)
        bif_r.append(np.full(len(x), r)); bif_x.append(x)
    bif_r, bif_x = np.concatenate(bif_r), np.concatenate(bif_x)

    # (d) lambda(r)
    lam_r = np.linspace(a.r_lo, a.r_hi, max(400, a.n_r // 2))
    lam = compute_lyapunov_array(lam_r, n_steps=a.lam_steps, verbose=False)

    # (c) invariant measure
    d_meas = orbit(a.measure_r, a.n_density, a.burn_in)

    # (a) cobweb
    px, py, xs = cobweb_path(a.cob_r, a.cob_x0, a.cob_iters)
    lam_cob = compute_lyapunov_array(np.array([a.cob_r]), n_steps=200_000, verbose=False)[0]
    lam_meas = compute_lyapunov_array(np.array([a.measure_r]), n_steps=200_000, verbose=False)[0]
    print(f"cobweb  r={a.cob_r}  x0={a.cob_x0}  lambda={lam_cob:.4f}  "
          f"final states -> [{xs[-40:].min():.4f}, {xs[-40:].max():.4f}]")
    print(f"measure r={a.measure_r}  lambda={lam_meas:.4f}  "
          f"support [{d_meas.min():.4f}, {d_meas.max():.4f}]  ln2={np.log(2):.4f}")

    # ---------------- layout ----------------
    fig = plt.figure(figsize=(11.6, 7.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.22,
                  left=0.072, right=0.982, top=0.94, bottom=0.085)
    ax_cob = fig.add_subplot(gs[0, 0])
    ax_mea = fig.add_subplot(gs[1, 0])
    ax_bif = fig.add_subplot(gs[0, 1])
    ax_lam = fig.add_subplot(gs[1, 1], sharex=ax_bif)

    # (a) cobweb
    xx = np.linspace(0, 1, 600)
    ax_cob.plot(xx, a.cob_r * xx * (1 - xx), color="#1B2A4A", lw=1.9, zorder=3,
                label=rf"$f_r(x)$,  $r={a.cob_r:g}$")
    ax_cob.plot([0, 1], [0, 1], color="0.55", lw=1.0, ls="--", zorder=2,
                label=r"$y=x$")
    ax_cob.plot(px, py, color="#D62728", lw=0.7, alpha=0.75, zorder=4,
                label=rf"orbit, {a.cob_iters} iterations")
    ax_cob.plot([a.cob_x0], [0.0], "o", ms=6.5, mfc="w", mec="#D62728", mew=1.8,
                zorder=6, label=rf"$x_0={a.cob_x0:g}$")
    ax_cob.set_xlim(0, 1); ax_cob.set_ylim(0, 1)
    ax_cob.set_xlabel(r"$x_n$"); ax_cob.set_ylabel(r"$x_{n+1}$")
    ax_cob.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax_cob.set_title("(a)  Cobweb diagram", loc="left", fontsize=11)

    # (b) bifurcation
    ax_bif.plot(bif_r, bif_x, ",", color="#12203B", alpha=0.55, rasterized=True)
    ax_bif.set_ylabel(r"long-run states $x_n$")
    ax_bif.set_xlabel(r"parameter $r$")
    ax_bif.set_ylim(-0.02, 1.02)
    ax_bif.tick_params(labelbottom=True)
    ax_bif.set_title("(b)  Bifurcation diagram", loc="left", fontsize=11)

    # (c) invariant measure at measure_r
    h, edges = np.histogram(d_meas, bins=a.hist_bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_mea.bar(centers, h, width=(edges[1] - edges[0]) * 0.92,
               color="#C0392B", alpha=0.55, edgecolor="#8E2B20", lw=0.4,
               label=rf"empirical histogram ({a.hist_bins} bins)")
    z = np.linspace(1e-5, 1 - 1e-5, 4000)
    ax_mea.plot(z, 1.0 / (np.pi * np.sqrt(z * (1 - z))), "k--", lw=1.4,
                label=r"$\rho_4(x)=1/\pi\sqrt{x(1-x)}$")
    ax_mea.set_xlim(0, 1)
    ax_mea.set_ylim(0, float(np.max(h)) * 1.18)
    ax_mea.set_xlabel(r"state $x$")
    ax_mea.set_ylabel("density")
    ax_mea.legend(fontsize=8.5, loc="upper center", framealpha=0.92)
    ax_mea.set_title(rf"(c)  Invariant measure,  $r={a.measure_r:g}$",
                     loc="left", fontsize=11)

    # (d) lambda(r)
    ax_lam.axhline(0.0, color="0.45", lw=0.9, ls="--", zorder=1)
    ax_lam.plot(lam_r, lam, lw=0.9, color="#1B2A4A", zorder=3)
    ax_lam.fill_between(lam_r, 0, lam, where=(lam > 0), color="#C0392B",
                        alpha=0.22, lw=0, zorder=2)
    ax_lam.set_xlabel(r"parameter $r$"); ax_lam.set_ylabel(r"$\lambda(r)$")
    ax_lam.set_xlim(a.r_lo, a.r_hi); ax_lam.set_ylim(-3.0, 0.95)
    ax_lam.set_title(r"(d)  Lyapunov exponent  ($\lambda>0$: chaotic)",
                     loc="left", fontsize=11)

    if a.marker_lines:
        for r, c in zip(a.r_marks, ["#2F6FB2", "#E08A1E", "#C0392B"]):
            for ax in (ax_bif, ax_lam):
                ax.axvline(r, color=c, lw=1.1, alpha=0.85, zorder=4)

    for ax in (ax_cob, ax_bif, ax_mea, ax_lam):
        ax.grid(alpha=0.18, lw=0.5)

    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
