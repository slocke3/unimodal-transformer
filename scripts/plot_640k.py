"""The input-binning sweep at 640k steps with three seeds.

Three things this settles that one seed at 320k could not.

The ordering is monotone. Averaged over seeds, m* runs 189, 512, 648, 783 for
32 to 256 bins; the inversion reported from a single seed at 320k, where 128 gave
960 against 714 for 256, was noise.

The spread is not scatter, it is a split. Away from the threshold the three seeds
agree to within 10%. At the threshold they differ by 20x to 62x, with individual
runs landing either near the generalising plateau or near the memorising branch
and nothing in between. That is the bimodality Nguyen & Reddy (arXiv:2412.00104)
predict from independent memorising and generalising sub-circuits racing, and it
is not what a capacity bound predicts, which would place the threshold at the
same m every time with small scatter either side.

8 and 16 bins never resolve a transition at any seed: their drop across the whole
task range is under 2x, the input-quantisation floor dominating throughout.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
LM = np.log10(MS)
BINS = [8, 16, 32, 64, 128, 256]
SEEDS = [0, 1, 2]


def mse(nb, s):
    return np.array([float(np.nanmean(np.load(
        f"runs_divbins_640k/in{nb}_mse_m{m}_seed{s}/eval_per_r.npz")["rms_per_r"] ** 2))
        for m in MS])


def cross(c, f):
    c = np.asarray(c); pl = c[-2:].mean()
    if c[0] / pl < 2:
        return np.nan
    t = pl * (c[0] / pl) ** f
    for i in range(len(c) - 1):
        if c[i] >= t >= c[i + 1]:
            g = (np.log(c[i]) - np.log(t)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LM[i] + g * (LM[i + 1] - LM[i]))
    return np.nan


Y = {nb: np.array([mse(nb, s) for s in SEEDS]) for nb in BINS}
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
cmap = plt.get_cmap("viridis")
cols = {nb: cmap(v) for nb, v in zip(BINS, np.linspace(0.12, 0.88, len(BINS)))}

# ---- 1. the sweep, median with the seed envelope ---------------------------
fig, ax = plt.subplots(figsize=(7.4, 5.6))
for nb in BINS:
    y = Y[nb]
    ax.fill_between(MS, y.min(0), y.max(0), color=cols[nb], alpha=0.22, lw=0)
    ax.plot(MS, np.median(y, 0), "o-", color=cols[nb], ms=5.5, lw=1.7, label=str(nb))
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5)
ax.set_xlabel("number of training tasks (distinct $r$ values)")
ax.set_ylabel("mean squared error, new tasks")
ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="lower left",
          title="input bins", title_fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, -0.05,
         "Square loss, 640k steps, three seeds; line is the median, band spans the seeds.\n"
         "The band is invisible away from the transition and enormous at it — individual runs land on one branch or the other.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/taskdiv_640k.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/taskdiv_640k.png")

# ---- 2. seed spread against task count: the bimodality signature -----------
fig, ax = plt.subplots(figsize=(7.4, 5.2))
for nb in BINS:
    y = Y[nb]
    if y[:, 0].mean() / y[:, -2:].mean() < 2:
        continue
    ax.plot(MS, y.max(0) / y.min(0), "o-", color=cols[nb], ms=6, lw=1.8, label=str(nb))
    ms_ = np.nanmean([cross(y[s], .5) for s in SEEDS])
    ax.axvline(ms_, color=cols[nb], lw=1, ls=":", alpha=0.7)
ax.axhline(1, color="0.6", lw=1, ls="--")
ax.text(105, 1.06, "seeds agree", fontsize=9.5, color="0.45")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5)
ax.set_xlabel("number of training tasks (distinct $r$ values)")
ax.set_ylabel("spread across seeds  (max / min MSE)")
ax.legend(frameon=False, fontsize=9.5, loc="upper left", title="input bins",
          title_fontsize=9.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, -0.05,
         "Run-to-run spread on new tasks, three seeds. Dotted verticals mark each arm's mean $m^*$.\n"
         "Outcomes are reproducible everywhere except at the threshold, where a run generalises or does not.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/bimodality.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/bimodality.png")

# ---- 3. m* against bins, with the seed range -------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.0))
xs, mid, lo, hi = [], [], [], []
for nb in BINS:
    v = [cross(Y[nb][s], .5) for s in SEEDS]
    if not np.isfinite(v).all():
        continue
    xs.append(nb); mid.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
ax.errorbar(xs, mid, yerr=[np.array(mid) - lo, np.array(hi) - np.array(mid)],
            fmt="o-", color="k", ms=8, lw=1.8, capsize=5, mfc="white")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_yticks([200, 400, 800]); ax.set_yticklabels(["200", "400", "800"])
ax.axvspan(6, 24, color="0.9", zorder=0)
ax.text(11, 240, "no transition\nto locate", ha="center", fontsize=9.5, color="0.4")
ax.set_xlim(6, 340)
ax.set_xlabel("input bins"); ax.set_ylabel("transition task count  $m^*$")
ax.grid(alpha=0.25, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.5, -0.05,
         "Square loss, 640k steps. Point is the mean over three seeds, bars span them.\n"
         "Monotone once seeds are averaged; the inversion seen at one seed and 320k was noise.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/transition_640k.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/transition_640k.png")
