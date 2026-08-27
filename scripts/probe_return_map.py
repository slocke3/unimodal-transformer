"""Probe #1: does the model identify the map, or apply a memorized one?

The asymband sweep shows models failing hard just past their training band
(9-11 nats at 8% accuracy, on maps whose symbolic dynamics they have already
mastered). Two very different mechanisms produce that:

  memorized  the model applies a fixed token-transition table learned from the
             training band, and never attempts in-context identification. Its
             implied return map should trace the map it was TRAINED on,
             regardless of what the context actually shows.
  identified the model reads the map out of the context correctly but loses on
             the readout. Its implied return map should trace the TRUE g_alpha
             even where cross-entropy is terrible.

This is the analogue of the specialized/generalized diagnostic in Goddard et al.
(arXiv:2506.05574), where the generalized solution is recognisable by behaving
like OLS regardless of the task.

Method: feed the model contexts drawn from g_(R,alpha), read the implied
E[x_(n+1) | x_n] out of its predicted next-token distribution, and measure RMS
distance to (a) the true g_alpha and (b) the map at the band centre alpha=1.
Their ratio says which one it is tracking. The binning floor is reported as the
irreducible reference: no model can beat it, because the context only locates
x_n to within a bin.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.maps import asym_map


def binning_floor_asym(R, alpha, n_bins, n=40000, seed=0):
    """RMS spread of g(x) across a bin -- the best any binned model can do."""
    rng = np.random.default_rng(seed)
    xq = rng.uniform(0, 1, n)
    centers = (np.floor(xq * n_bins) + 0.5) / n_bins
    true = np.array([asym_map(float(x), R, alpha) for x in xq])
    approx = np.array([asym_map(float(c), R, alpha) for c in centers])
    return float(np.sqrt(np.mean((true - approx) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs_asym/asym_w0_seed0",
                             "runs_asym/asym_w0.4_seed0"],
                    help="run directories to probe (each needs best_final.pt)")
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--out_dir", default="figures_asym")
    a = ap.parse_args()

    import torch
    from src.model import DiscreteTrajectoryTransformer
    from src.system_id import implied_return_map_asym, rms_to_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for run in a.runs:
        with open(Path(run) / "params.json") as fh:
            P = json.load(fh)
        model = DiscreteTrajectoryTransformer(
            n_bins=P["n_bins"], context_len=P["context_len"],
            d_model=P["d_model"], n_heads=P["n_heads"],
            n_layers=P["n_layers"], dropout=P["dropout"]).to(device)
        ckpt = torch.load(Path(run) / "best_final.pt", map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        band_lo = P["alpha_lo"]
        print(f"\n=== {run}   band alpha in [{band_lo:.3f}, 1.000] ===")
        print("%7s %9s %11s %13s %11s %9s"
              % ("alpha", "in band", "RMS->true", "RMS->edge", "ratio",
                 "floor"))
        rows = []
        for al in a.alphas:
            x, e = implied_return_map_asym(
                model, device, a.R, al, P["n_bins"], P["context_len"],
                n_traj=a.n_traj)
            d_true = rms_to_map(x, e, a.R, al)
            # The model saw a BAND of maps. The meaningful null is the nearest
            # one it actually trained on -- its band edge -- not alpha=1, which
            # for a wide band is just a distant member of the training set.
            d_train = rms_to_map(x, e, a.R, band_lo)
            floor = binning_floor_asym(a.R, al, P["n_bins"])
            inb = al >= band_lo - 1e-9
            rows.append({"alpha": al, "x": x, "e": e, "d_true": d_true,
                         "d_train": d_train, "floor": floor, "in_band": inb})
            print("%7.2f %9s %11.4f %13.4f %11.2f %9.4f"
                  % (al, "yes" if inb else "no", d_true, d_train,
                     d_true / d_train if d_train > 0 else np.nan, floor))
        results[run] = {"rows": rows, "band_lo": band_lo, "P": P}

    # -- figure: implied map vs the two candidate truths -------------------
    n_run, n_al = len(results), len(a.alphas)
    fig, axes = plt.subplots(n_run, n_al, figsize=(3.0 * n_al, 3.2 * n_run),
                             squeeze=False, sharex=True, sharey=True)
    xs = np.linspace(0, 1, 400)
    handles = None
    for i, (run, res) in enumerate(results.items()):
        lo = res["band_lo"]
        # The model saw a whole BAND of maps, not one. Shade the envelope they
        # span: anything inside it is reproducible by some map the model was
        # trained on, so only orange lying on green OUTSIDE the envelope is
        # evidence of extrapolation.
        band_alphas = np.linspace(lo, 1.0, 41)
        fam = np.array([[asym_map(float(x), a.R, al) for x in xs]
                        for al in band_alphas])
        env_lo, env_hi = fam.min(axis=0), fam.max(axis=0)
        edge = np.array([asym_map(float(x), a.R, lo) for x in xs])
        for j, row in enumerate(res["rows"]):
            ax = axes[i][j]
            true_a = np.array([asym_map(float(x), a.R, row["alpha"])
                               for x in xs])
            h0 = ax.fill_between(xs, env_lo, env_hi, color="#1B2A4A",
                                 alpha=0.16, lw=0)
            h1, = ax.plot(xs, edge, lw=1.5, color="#1B2A4A", ls="--")
            h2, = ax.plot(xs, true_a, lw=1.9, color="#2E7D32")
            h3, = ax.plot(row["x"], row["e"], ".", ms=1.8, alpha=0.35,
                          color="#D85A30")
            handles = (h0, h1, h2, h3)
            tag = "IN BAND" if row["in_band"] else "held out"
            ax.set_title(rf"$\alpha$={row['alpha']:g}   ({tag})", fontsize=9.5,
                         color="#1B2A4A" if row["in_band"] else "#B71C1C")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
            if j == 0:
                ax.set_ylabel(f"trained on\n"
                              rf"$\alpha \in [{res['band_lo']:.2f}, 1]$"
                              f"\n(w={res['P']['band_width']:g})"
                              "\n\n$E[x_{n+1}\,|\,x_n]$", fontsize=9)
            if i == n_run - 1:
                ax.set_xlabel("$x_n$")

    fig.legend(handles,
               ["envelope of ALL maps in the training band",
                r"nearest map in the training band (its edge, $\alpha=\alpha_{lo}$)",
                r"TRUE map generating this panel's context",
                r"model's implied $E[x_{n+1}\,|\,x_n]$"],
               loc="lower center", ncol=4, fontsize=9.5, frameon=True,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Implied return map at $R=1$: is the model extrapolating, or "
                 "clamping to the nearest map it was trained on?\n"
                 "Orange inside the shaded envelope = reproducible by some "
                 "training map.  Only orange on GREEN and OUTSIDE the envelope "
                 "is extrapolation.", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.035, 1, 1])

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"return_map_probe.{ext}", dpi=160,
                    bbox_inches="tight")
    payload = {f"{os.path.basename(r)}_{k}": np.array(
                   [row[k] for row in res["rows"]])
               for r, res in results.items()
               for k in ("alpha", "d_true", "d_train", "floor")}
    # keep the raw clouds too: re-plotting then needs no GPU and no checkpoint
    for r, res in results.items():
        b = os.path.basename(r)
        payload[f"{b}_band_lo"] = np.array(res["band_lo"])
        for row in res["rows"]:
            payload[f"{b}_x_{row['alpha']:g}"] = row["x"]
            payload[f"{b}_e_{row['alpha']:g}"] = row["e"]
    np.savez(out / "return_map_probe.npz", **payload)
    print(f"\nwrote {out}/return_map_probe.png")


if __name__ == "__main__":
    main()
