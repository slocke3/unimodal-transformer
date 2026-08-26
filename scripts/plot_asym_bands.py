"""Figures for the asymmetric-family band sweep.

Each run trains on alpha ~ U[1-w, 1] (w=0 is the logistic-only zero-shot arm)
and is evaluated over the whole (alpha, R) plane, where R is the peak height
(R = r/4) and x_c = alpha/2 is the peak position.

Three figures:

  asym_generalization   in-band vs out-of-band mean CE against band width w --
                        does training on a band of asymmetry buy generalization
                        to unseen asymmetry, and how wide must the band be?
  asym_ce_vs_R          CE against R at selected alpha, one panel per band width;
                        this is the per-r/per-R view, with the training band
                        shaded and the trivially-predictable low-R region marked
  asym_heatmaps         CE over the full (alpha, R) plane, one panel per run

A note on the low-R region: for R < 0.25 (logistic r < 1) the origin is
attracting for every alpha, so orbits die and the token sequence is a constant.
Those points are trivially predictable and are hatched in the heatmaps, since a
low CE there says nothing about generalization.

Torch-free: reads eval_asym.npz / params.json only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# Below this peak height the map is subcritical (logistic r < 1): every orbit
# falls into the origin, so the sequence is constant and trivially predictable.
R_TRIVIAL = 0.25
PROTOCOL = {"final": ("#D85A30", "-", "o", "Fixed steps (final)"),
            "bestval": ("#1B2A4A", "--", "s", "Early stopping (best val)")}


def load_runs(root):
    runs = []
    for path in sorted(Path(root).glob("asym_w*/eval_asym.npz")):
        with open(path.parent / "params.json") as handle:
            params = json.load(handle)
        with np.load(path) as z:
            run = {"name": path.parent.name, "params": params,
                   "w": float(z["band_width"]), "alpha_lo": float(z["alpha_lo"]),
                   "alpha": z["alpha_grid"].copy(), "R": z["R_grid"].copy(),
                   "ce_final": z["ce_final"].copy(),
                   "dead": z["dead_frac"].copy(),
                   "in_band": z["in_band_alpha"].copy()}
            run["ce_bestval"] = (z["ce_bestval"].copy()
                                 if "ce_bestval" in z.files else None)
        runs.append(run)
    runs.sort(key=lambda r: r["w"])
    return runs


def _means(run, protocol):
    """(in-band mean, out-of-band mean) over non-trivial R only."""
    ce = run[f"ce_{protocol}"]
    if ce is None:
        return np.nan, np.nan
    keep_R = run["R"] >= R_TRIVIAL
    ce = ce[:, keep_R]
    ib = run["in_band"]
    inb = ce[ib].mean() if ib.any() else np.nan
    out = ce[~ib].mean() if (~ib).any() else np.nan
    return inb, out


def fig_generalization(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ws = [r["w"] for r in runs]

    ax = axes[0]
    for protocol, (color, ls, marker, label) in PROTOCOL.items():
        if all(r.get(f"ce_{protocol}") is None for r in runs):
            continue
        inb = [_means(r, protocol)[0] for r in runs]
        out_ = [_means(r, protocol)[1] for r in runs]
        ax.plot(ws, inb, marker + "-", color=color, lw=1.8,
                label=f"{label} — in band")
        ax.plot(ws, out_, marker + "--", color=color, lw=1.4, mfc="white",
                alpha=0.8, label=f"{label} — out of band")
    ax.set_xlabel(r"Training band half-width $w$   ($\alpha \sim U[1-w,\ 1]$)")
    ax.set_ylabel("Mean cross-entropy (nats)")
    ax.set_title(r"Generalization across $\alpha$" + "\n"
                 r"($w=0$ is train-on-logistic, zero-shot)", fontsize=10.5)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7.5)

    # CE as a function of how far outside the band alpha sits
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, run in enumerate(runs):
        ce = run["ce_final"]
        keep_R = run["R"] >= R_TRIVIAL
        per_alpha = ce[:, keep_R].mean(axis=1)
        ax.plot(run["alpha"], per_alpha, "-", lw=1.5,
                color=cmap(i / max(1, len(runs) - 1)), label=f"w={run['w']:g}")
        ax.axvline(run["alpha_lo"], color=cmap(i / max(1, len(runs) - 1)),
                   lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel(r"Evaluation $\alpha$  (peak position $x_c=\alpha/2$)")
    ax.set_ylabel("Mean cross-entropy over $R$ (nats)")
    ax.set_title(r"Loss vs asymmetry; dotted lines mark each band edge",
                 fontsize=10.5)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Asymmetric unimodal family: does band training generalize "
                 "across peak position?", fontsize=11.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"asym_generalization.{ext}", dpi=160,
                    bbox_inches="tight")


def fig_ce_vs_R(runs, out, n_alpha_show=5):
    ncols = min(5, len(runs))
    nrows = -(-len(runs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for ax, run in zip(axes.ravel(), runs):
        idx = np.linspace(0, len(run["alpha"]) - 1, n_alpha_show).astype(int)
        cmap = plt.get_cmap("plasma")
        for k, i in enumerate(idx):
            a = run["alpha"][i]
            inside = "in" if run["in_band"][i] else "out"
            ax.plot(run["R"], run["ce_final"][i], lw=1.3,
                    color=cmap(k / max(1, len(idx) - 1)),
                    label=rf"$\alpha$={a:.2f} ({inside})")
        ax.axvspan(run["R"].min(), R_TRIVIAL, color="gray", alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_title(rf"$w$={run['w']:g}  ($\alpha \geq$ {run['alpha_lo']:.3f})",
                     fontsize=10)
        ax.grid(alpha=0.22, which="both", lw=0.4)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Peak height $R$   ($r = 4R$)")
    for row in axes:
        row[0].set_ylabel("Cross-entropy (nats)")
    fig.suptitle("Loss vs peak height $R$ at selected asymmetries; shaded band "
                 "is $R<0.25$ ($r<1$), where every orbit dies at the origin",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"asym_ce_vs_R.{ext}", dpi=160, bbox_inches="tight")


def fig_heatmaps(runs, out):
    ncols = min(5, len(runs))
    nrows = -(-len(runs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.3 * nrows),
                             squeeze=False)
    finite = np.concatenate([r["ce_final"][np.isfinite(r["ce_final"])].ravel()
                             for r in runs])
    vmin = max(np.percentile(finite, 1), 1e-4)
    vmax = np.percentile(finite, 99)
    for ax, run in zip(axes.ravel(), runs):
        im = ax.pcolormesh(run["R"], run["alpha"], run["ce_final"],
                           norm=LogNorm(vmin=vmin, vmax=vmax), cmap="magma",
                           shading="nearest")
        ax.axhline(run["alpha_lo"], color="white", lw=1.4, ls="--")
        # hatch the trivially-predictable region
        ax.contourf(run["R"], run["alpha"], (run["dead"] > 0.5).astype(float),
                    levels=[0.5, 1.5], colors="none", hatches=["///"])
        ax.set_title(rf"$w$={run['w']:g}", fontsize=10)
        ax.set_xlabel("$R$")
        ax.set_ylabel(r"$\alpha$")
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Cross-entropy (nats)",
                 fraction=0.02, pad=0.01)
    fig.suptitle("Cross-entropy over the $(\\alpha, R)$ plane. Dashed line: "
                 "lower edge of the training band. Hatching: orbits die at the "
                 "origin (trivially predictable).", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"asym_heatmaps.{ext}", dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_asym")
    ap.add_argument("--out_dir", default="figures_asym")
    a = ap.parse_args()

    runs = load_runs(a.runs_dir)
    if not runs:
        raise FileNotFoundError(f"no asym runs under {a.runs_dir}")

    print(f"{len(runs)} runs")
    print(f"{'w':>7} {'alpha_lo':>9} {'in-band':>9} {'out-band':>9} {'ratio':>7}")
    for run in runs:
        inb, outb = _means(run, "final")
        ratio = outb / inb if inb and np.isfinite(inb) else np.nan
        print(f"{run['w']:>7g} {run['alpha_lo']:>9.3f} {inb:>9.4f} "
              f"{outb:>9.4f} {ratio:>7.2f}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_generalization(runs, out)
    fig_ce_vs_R(runs, out)
    fig_heatmaps(runs, out)
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
