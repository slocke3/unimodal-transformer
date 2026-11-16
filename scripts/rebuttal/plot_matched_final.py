"""Plot saved matched-final-token summaries without rerunning model inference.

Uses pooled-pair slope and correlation from matched_final.csv. The plotted
correlation is between true and predicted pair differences, not between the
individual predictions and targets. These are descriptive, single-seed results.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("runs_rebuttal_probe/matched_final.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures_rebuttal/matched_final_summary"))
    args = parser.parse_args()
    with args.input.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["regime"] == "all"]
    if not rows:
        raise ValueError("No pooled-pair results in input CSV.")
    if any(int(r["L"]) != 50 or int(r["n_bins_in"]) != 64 for r in rows):
        raise ValueError("This summary expects L=50 and 64 input bins.")
    if len({r["seed"] for r in rows}) != 1:
        raise ValueError("Multiple training seeds need a separate aggregation.")

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                         "axes.titlesize": 13, "axes.linewidth": 1.1,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    styles = [("bins", "CE model: mean of output bins", "#1f77b4", "o"),
              ("scalar", "Scalar MSE model", "#d55e00", "s")]
    tasks = sorted({int(r["m"]) for r in rows})
    for mode, label, color, marker in styles:
        selected = sorted((r for r in rows if r["output"] == mode),
                          key=lambda r: int(r["m"]))
        ms = [int(r["m"]) for r in selected]
        if ms != tasks:
            raise ValueError("Expected one observation per objective and task count.")
        for ax, metric in zip(axes, ["pair_slope", "pair_corr"]):
            ax.plot(ms, [float(r[metric]) for r in selected], color=color,
                    marker=marker, lw=2, ms=6, label=label)

    axes[0].set_title("(a) Size of the prediction difference")
    axes[0].set_ylabel("Slope: predicted vs. true difference")
    axes[0].set_ylim(-0.06, 1.10)
    axes[0].axhline(0, color="0.45", ls=":", lw=1.3)
    axes[0].text(0.04, 0.095, "Last-token-only predictor: slope 0",
                 transform=axes[0].transAxes, fontsize=10, color="0.35")
    axes[0].legend(loc="center right", frameon=True, fontsize=10)
    axes[1].set_title("(b) Agreement across matched pairs")
    axes[1].set_ylabel("Correlation: predicted vs. true difference")
    axes[1].set_ylim(0.85, 1.012)
    axes[1].set_yticks([0.85, 0.90, 0.95, 1.00])
    for ax in axes:
        ax.axhline(1, color="0.4", ls="--", lw=1.2, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(tasks, [f"{m:,}" for m in tasks])
        ax.set_xlabel("Number of training tasks $m$")
        ax.grid(alpha=0.2, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    counts = {int(r["n_pairs"]) for r in rows}
    if len(counts) != 1:
        raise ValueError("Expected a common number of pairs across models.")
    fig.suptitle("Different histories, identical final input token", fontsize=15)
    fig.text(0.5, 0.015,
             f"64 input bins; context length 50; {next(iter(counts)):,} matched pairs; "
             "one training seed per condition. Dashed lines mark 1.",
             ha="center", fontsize=10, color="0.3")
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = args.out.with_suffix("." + suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
