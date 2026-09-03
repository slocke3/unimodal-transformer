"""Torch-free analysis for the continuous-input / output-resolution sweep.

Reads runs_cont/*/eval_continuous.npz (+ params.json) and makes:
  1. mean reference-grid CE vs n_out (seed error bars) -- the bias/variance
     sweet spot; "continuous = infinite bins is better" made fair.
  2. mean implied-map RMS vs n_out (seed error bars) -- the resolution-free
     version of the same question.
  3. reference-grid CE vs r, one curve per n_out -- where in the family the
     output resolution actually matters (expect: the chaotic band).

Both metrics are split into the full range and the chaotic band, since the
whole effect lives in chaos. Run from the repo root:

  python scripts/plot_continuous.py --runs runs_cont --out figures_cont
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_runs(runs_dir):
    recs = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        npz_path = os.path.join(d, "eval_continuous.npz")
        par_path = os.path.join(d, "params.json")
        if not (os.path.exists(npz_path) and os.path.exists(par_path)):
            continue
        z = np.load(npz_path)
        p = json.load(open(par_path))
        recs.append(dict(
            n_out=int(z["n_out"]), seed=int(p.get("seed", 0)),
            r_grid=z["r_grid"], ce_ref=z["ce_ref_per_r"],
            ce_native=z["ce_native_per_r"], rms=z["rms_per_r"],
        ))
    if not recs:
        raise SystemExit(f"no eval_continuous.npz found under {runs_dir}")
    return recs


def group_by_nout(recs):
    g = defaultdict(list)
    for r in recs:
        g[r["n_out"]].append(r)
    return dict(sorted(g.items()))


def band_mean(rec, key, r_chaos, chaotic_only):
    v, rg = rec[key], rec["r_grid"]
    if chaotic_only:
        v = v[rg >= r_chaos]
    return np.nanmean(v)


def agg(grouped, n, key, r_chaos, chaotic_only):
    vals = [band_mean(rec, key, r_chaos, chaotic_only) for rec in grouped[n]]
    return np.mean(vals), np.std(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cont")
    ap.add_argument("--out", default="figures_cont")
    ap.add_argument("--r_chaos", type=float, default=3.5699,
                    help="onset of chaos; the chaotic-band average uses r >= this")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    grouped = group_by_nout(load_runs(a.runs))
    n_outs = np.array(list(grouped.keys()))

    # --- 1 & 2: mean CE_ref and RMS vs n_out -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, ylab, title in [
        (axes[0], "ce_ref", "Reference-grid CE (nats)", "Which output resolution predicts best?"),
        (axes[1], "rms", "Implied-map RMS", "Recovering the map (resolution-free)"),
    ]:
        for chaotic, label, fmt in [(False, "full range", "-o"),
                                    (True, "chaotic band", "--s")]:
            m = np.array([agg(grouped, n, key, a.r_chaos, chaotic)[0] for n in n_outs])
            e = np.array([agg(grouped, n, key, a.r_chaos, chaotic)[1] for n in n_outs])
            ax.errorbar(n_outs, m, yerr=e, fmt=fmt, capsize=3, lw=1.4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_outs)
        ax.set_xticklabels([str(int(n)) for n in n_outs])
        ax.set_xlabel(r"output bins  $N_{\mathrm{out}}$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
    fig.suptitle("Continuous input, swept output binning", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "continuous_resolution_sweep.png"),
                dpi=160, bbox_inches="tight")

    # --- 3: CE_ref vs r, one curve per n_out (seed-averaged) -------------
    fig2, ax = plt.subplots(figsize=(11, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_outs)))
    for c, n in zip(colors, n_outs):
        rg = grouped[n][0]["r_grid"]
        ce = np.mean([rec["ce_ref"] for rec in grouped[n]], axis=0)
        ax.plot(rg, ce, color=c, lw=1.3, label=rf"$N_{{\mathrm{{out}}}}$={n}")
    ax.axvline(a.r_chaos, color="gray", ls=":", lw=.8, alpha=.6)
    ax.set_xlabel(r"logistic parameter $r$")
    ax.set_ylabel("Reference-grid CE (nats)")
    ax.set_title("Where output resolution matters across the family", fontsize=11)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(a.out, "continuous_ce_vs_r.png"),
                 dpi=160, bbox_inches="tight")

    print(f"wrote figures to {a.out}/  (n_out = {list(map(int, n_outs))})")


if __name__ == "__main__":
    main()
