"""Combine the context-length sweep with the synonym-token control.

Panel (a) reproduces the trained-at-length task-diversity sweep using the
median and full range over three seeds.  Panel (b) reproduces the left panel
of ``plot_synonyms.py``: the crossover location under spatial refinement and
under vocabulary-only synonym splitting.
"""

import glob

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MS_CONTEXT = np.array(
    [100, 150, 250, 350, 500, 700, 1000, 1400, 2000, 4000, 8000]
)
CONTEXTS = [10, 20, 25, 30, 35, 50, 60, 75]
SEEDS = [0, 1, 2]
OLD_M = {100, 250, 500, 1000, 2000, 4000, 8000}

MS_SYNONYM = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
LOG_MS_SYNONYM = np.log10(MS_SYNONYM)
SYNONYMS = [(1, 64), (2, 128), (4, 256), (8, 512), (16, 1024)]
BIN_RUNS = [
    (8, "runs_divbins/in8_bins_m{m}_seed0"),
    (16, "runs_divbins/in16_bins_m{m}_seed0"),
    (32, "runs_divbins/in32_bins_m{m}_seed0"),
    (64, "runs_divbins_320k/in64_bins_m{m}_seed0"),
    (128, "runs_divbins_320k/in128_bins_m{m}_seed0"),
    (256, "runs_divbins_320k/in256_bins_m{m}_seed0"),
]


def context_result(context, m, key, seed):
    if context == 50 and m in OLD_M:
        run = f"runs_divbins_320k/in64_mse_m{m}_seed{seed}"
    else:
        run = f"runs_ctx/L{context}_m{m}_seed{seed}"
    path = run + "/eval_per_r.npz"
    if not glob.glob(path):
        return np.nan
    # Stored values are per-r RMS errors.  Square before averaging to recover
    # mean squared error over the evaluation grid.
    return float(np.nanmean(np.load(path)[key] ** 2))


def context_band(context, key):
    values = np.array(
        [
            [context_result(context, m, key, seed) for m in MS_CONTEXT]
            for seed in SEEDS
        ]
    )
    return np.nanmedian(values, axis=0), np.nanmin(values, axis=0), np.nanmax(
        values, axis=0
    )


def ce_curve(template):
    return np.array(
        [
            float(
                np.nanmean(
                    np.load(template.format(m=m) + "/eval_per_r.npz")["ce_per_r"]
                )
            )
            for m in MS_SYNONYM
        ]
    )


def crossover(curve, min_drop=2.0):
    """Log-scale midpoint between the m=100 value and large-m plateau."""
    plateau = curve[-2:].mean()
    if curve[0] / plateau < min_drop:
        return None
    midpoint = np.sqrt(curve[0] * plateau)
    for i in range(len(curve) - 1):
        if curve[i] >= midpoint >= curve[i + 1]:
            fraction = (np.log(curve[i]) - np.log(midpoint)) / (
                np.log(curve[i]) - np.log(curve[i + 1])
            )
            return 10 ** (
                LOG_MS_SYNONYM[i]
                + fraction * (LOG_MS_SYNONYM[i + 1] - LOG_MS_SYNONYM[i])
            )
    return None


synonym_curves = {
    j: ce_curve(
        "runs_divbins_320k/in64_bins_m{m}_seed0"
        if j == 1
        else f"runs_syn/syn{j}_m{{m}}_seed0"
    )
    for j, _ in SYNONYMS
}
bin_curves = {n_bins: ce_curve(template) for n_bins, template in BIN_RUNS}


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.linewidth": 1.05,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)

fig, (ax_context, ax_synonym) = plt.subplots(
    1,
    2,
    figsize=(12.4, 5.25),
    gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.32},
)

# (a) Task diversity by trained context length.
context_colors = [
    plt.get_cmap("viridis")(value)
    for value in np.linspace(0.08, 0.92, len(CONTEXTS))
]
for context, color in zip(CONTEXTS, context_colors):
    median, low, high = context_band(context, "rms_per_r")
    ax_context.fill_between(
        MS_CONTEXT, low, high, color=color, alpha=0.18, lw=0, zorder=1
    )
    ax_context.plot(
        MS_CONTEXT,
        median,
        "o-",
        color=color,
        ms=4.6,
        lw=1.65,
        zorder=3,
        label=str(context),
    )
    seen_median, _, _ = context_band(context, "rms_at_train_r")
    ax_context.plot(
        MS_CONTEXT,
        seen_median,
        "--",
        color=color,
        lw=1.1,
        alpha=0.78,
        zorder=2,
    )

ax_context.set_xscale("log")
ax_context.set_yscale("log")
ax_context.set_xlabel("number of training tasks (distinct $r$ values)")
ax_context.set_ylabel("mean squared error")
ax_context.set_title("Context length and task diversity", fontsize=12)
context_legend = ax_context.legend(
    frameon=False,
    fontsize=8.5,
    ncol=2,
    loc="center right",
    bbox_to_anchor=(1.0, 0.58),
    title="context length",
    title_fontsize=9,
)
ax_context.add_artist(context_legend)
style_handles = [
    plt.Line2D([], [], color="k", lw=1.65, marker="o", ms=4.6),
    plt.Line2D([], [], color="k", lw=1.1, ls="--"),
]
ax_context.legend(
    style_handles,
    ["new tasks", "seen tasks"],
    frameon=False,
    fontsize=8.5,
    loc="upper right",
)

# (b) Spatial resolution versus vocabulary-only inflation.
bin_x = [n_bins for n_bins, _ in BIN_RUNS if crossover(bin_curves[n_bins])]
bin_y = [
    crossover(bin_curves[n_bins])
    for n_bins, _ in BIN_RUNS
    if crossover(bin_curves[n_bins])
]
syn_x = [vocabulary for j, vocabulary in SYNONYMS if crossover(synonym_curves[j])]
syn_y = [
    crossover(synonym_curves[j])
    for j, _ in SYNONYMS
    if crossover(synonym_curves[j])
]
ax_synonym.plot(
    bin_x,
    bin_y,
    "o-",
    color=plt.get_cmap("viridis")(0.72),
    ms=4.6,
    lw=1.65,
    label="spatial refinement\n(vocabulary + information)",
)
ax_synonym.plot(
    syn_x,
    syn_y,
    "s--",
    color=plt.get_cmap("viridis")(0.18),
    ms=4.6,
    lw=1.65,
    label="synonym splitting\n(vocabulary only)",
)
ax_synonym.set_xscale("log", base=2)
ax_synonym.set_yscale("log")
ax_synonym.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
ax_synonym.set_xticklabels(
    ["8", "16", "32", "64", "128", "256", "512", "1024"], fontsize=8
)
ax_synonym.set_yticks([200, 400, 800])
ax_synonym.set_yticklabels(["200", "400", "800"])
ax_synonym.set_ylim(150, 1100)
ax_synonym.set_xlabel("input vocabulary size")
ax_synonym.set_ylabel("crossover task count $m^*$")
ax_synonym.set_title("Resolution versus vocabulary size", fontsize=12)
ax_synonym.legend(frameon=False, fontsize=8.5, loc="upper left")

for label, axis in zip(("(a)", "(b)"), (ax_context, ax_synonym)):
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    axis.grid(alpha=0.24, lw=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.075, right=0.99, bottom=0.14, top=0.91)
for extension in ("png", "pdf"):
    fig.savefig(
        f"figures_taskdiv/context_synonym_combined.{extension}",
        dpi=220,
        bbox_inches="tight",
    )

print("wrote figures_taskdiv/context_synonym_combined.{png,pdf}")
