"""Transformer vs reference predictors on the task-diversity sweep (FWzs).

Reads the transformer runs behind taskdiv_gap and the baselines written by
scripts/rebuttal/baselines_taskdiv.py (same training data, same evaluation
contexts). One column per input resolution, rows = objective:

  transformer, new tasks     solid black, seed range shaded
  transformer, seen tasks    dashed black
  k-gram, k = 1, 2, 4, 6, 10 new tasks; local transition counting, no notion
                             of the map family

The least-squares parameter-estimation results are retained in the numerical
summaries but omitted from the figures.

Lines are medians over seeds 0-3. Torch-free:
  python scripts/rebuttal/plot_baselines_taskdiv.py              # N = 16, 64, 256
  python scripts/rebuttal/plot_baselines_taskdiv.py --ns 8 16 32 64 128 256 \
      --out figures_rebuttal/baselines_taskdiv_all.pdf
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15.5, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
    "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
})

MS = [100, 250, 500, 1000, 2000, 4000, 8000]
TMPL = {8: "runs_divbins/in8_{o}_m{m}_seed{s}", 16: "runs_divbins/in16_{o}_m{m}_seed{s}",
        32: "runs_divbins/in32_{o}_m{m}_seed{s}", 64: "runs_divbins_320k/in64_{o}_m{m}_seed{s}",
        128: "runs_divbins_320k/in128_{o}_m{m}_seed{s}",
        256: "runs_divbins_320k/in256_{o}_m{m}_seed{s}"}
KS = (1, 2, 4, 6, 10)
K_COLORS = {1: "#9ecae1", 2: "#4292c6", 4: "#08519c", 6: "#7c3aed", 10: "#d97706"}
K_MARKERS = {1: "s", 2: "s", 4: "s", 6: "D", 10: "^"}


def collect(n, seeds, bl_dir):
    """Per seed x m arrays of the aggregated errors for every predictor."""
    keys = ["tf_ce_new", "tf_ce_seen", "tf_mse_new", "tf_mse_seen", "lsr_mse"]
    keys += [f"k{k}_{obj}" for k in KS for obj in ("ce", "mse")]
    d = {key: [] for key in keys}
    for s in seeds:
        rows = {k: [] for k in d}
        ok = True
        for m in MS:
            pc = TMPL[n].format(o="bins", m=m, s=s) + "/eval_per_r.npz"
            pm = TMPL[n].format(o="mse", m=m, s=s) + "/eval_per_r.npz"
            pb = os.path.join(bl_dir, f"m{m}_seed{s}.npz")
            if not all(os.path.exists(p) for p in (pc, pm, pb)):
                ok = False
                break
            c, q, b = np.load(pc), np.load(pm), np.load(pb)
            rows["tf_ce_new"].append(np.nanmean(c["ce_per_r"]))
            rows["tf_ce_seen"].append(np.nanmean(c["ce_at_train_r"]))
            rows["tf_mse_new"].append(np.nanmean(q["rms_per_r"] ** 2))
            rows["tf_mse_seen"].append(np.nanmean(q["rms_at_train_r"] ** 2))
            for k in KS:
                rows[f"k{k}_ce"].append(b[f"N{n}_k{k}_new_ce"].mean())
                rows[f"k{k}_mse"].append(b[f"N{n}_k{k}_new_mse"].mean())
            rows["lsr_mse"].append(b[f"N{n}_lsr_new_mse"].mean())
        if ok:
            for k in d:
                d[k].append(rows[k])
    return {k: np.array(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--bl_dir", default="runs_rebuttal_baselines")
    ap.add_argument("--out", default="figures_rebuttal/baselines_taskdiv.pdf")
    ap.add_argument("--ncols", type=int, default=None,
                    help="panels per row; resolutions wrap onto further rows, "
                         "all cross-entropy rows first, then all square-loss rows")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    nc = a.ncols or len(a.ns)
    nblk = -(-len(a.ns) // nc)          # rows per objective
    nrows = 2 * nblk
    fig, axes = plt.subplots(nrows, nc, figsize=(5.6 * nc + 1, 5.25 * nrows),
                             squeeze=False)
    for j in range(len(a.ns), nblk * nc):   # unused slots in a ragged last row
        for ro in range(2):
            axes[ro * nblk + j // nc, j % nc].axis("off")
    lines = []
    for j, n in enumerate(a.ns):
        d = collect(n, a.seeds, a.bl_dir)
        ns_ = len(d["tf_ce_new"])
        lines.append(f"== N={n} ({ns_} seeds)")
        for ro, (obj, ylab) in enumerate([("ce", "new-task cross-entropy (nats)"),
                                          ("mse", "new-task mean squared error")]):
            row, col = ro * nblk + j // nc, j % nc
            ax = axes[row, col]
            tf = d[f"tf_{obj}_new"]
            ax.fill_between(MS, tf.min(0), tf.max(0), color="0.15", alpha=0.15, lw=0)
            ax.plot(MS, np.median(tf, 0), "-o", color="0.1", lw=2.6, ms=6,
                    label="transformer, new tasks", zorder=5)
            ax.plot(MS, np.median(d[f"tf_{obj}_seen"], 0), "--", color="0.1", lw=1.6,
                    label="transformer, seen tasks", zorder=4)
            for k in KS:
                ax.plot(MS, np.median(d[f"k{k}_{obj}"], 0), linestyle="-",
                        marker=K_MARKERS[k], color=K_COLORS[k],
                        lw=2.0, ms=5, label=f"{k}-gram (last {k} token{'s' if k > 1 else ''})")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(alpha=0.18, lw=0.5, which="both")
            if row == 0 or nblk > 1:
                ax.set_title(f"{n} input bins")
            if row == nrows - 1:
                ax.set_xlabel(r"training tasks $m$")
            if col == 0:
                ax.set_ylabel(ylab)
            for i, m in enumerate(MS):
                if m in (100, 1000, 8000):
                    vals = [f"tf {np.median(tf, 0)[i]:.3g}"] + [
                        f"{k}g {np.median(d[f'k{k}_{obj}'], 0)[i]:.3g}" for k in KS]
                    if obj == "mse":
                        vals.append(f"lsr {np.median(d['lsr_mse'], 0)[i]:.3g}")
                    lines.append(f"  {obj:3s} m={m:<5d} " + "  ".join(vals))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06 if nrows == 2 else -0.03),
               framealpha=0.95)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    summary = os.path.splitext(a.out)[0] + "_summary.txt"
    with open(summary, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {a.out}, {png}, {summary}")


if __name__ == "__main__":
    main()
