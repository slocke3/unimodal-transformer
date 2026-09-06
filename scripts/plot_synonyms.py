"""Does vocabulary size matter once it stops destroying information?

The tokenisation sweep varied the number of bins, which changes the vocabulary
but also throws information away: bin(x) does not determine x. A real tokenizer
is lossless, so the two cannot be equated, and the coarse arms there may simply
have been handicapped.

Synonym splitting separates them. Each of 64 bins is split into j interchangeable
tokens, one drawn at random per occurrence, so the vocabulary grows by j while the
bin stays recoverable as token // j. Information, sequence length, example count
and the 64-bin output space are all untouched; the only change is that each
embedding row now receives a fraction 1/j of the gradient updates -- exactly the
mechanism the tokenisation literature blames for large vocabularies.

Left panel is the answer: against vocabulary size, the binning sweep climbs and
the synonym sweep is flat. Right panel is what vocabulary inflation does cost.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
LM = np.log10(MS)
SYN = [(1, 64), (2, 128), (4, 256), (8, 512), (16, 1024)]
BIN = [("in8", 8, "runs_divbins/in8_bins_m{m}_seed0"),
       ("in16", 16, "runs_divbins/in16_bins_m{m}_seed0"),
       ("in32", 32, "runs_divbins/in32_bins_m{m}_seed0"),
       ("in64", 64, "runs_divbins_320k/in64_bins_m{m}_seed0"),
       ("in128", 128, "runs_divbins_320k/in128_bins_m{m}_seed0"),
       ("in256", 256, "runs_divbins_320k/in256_bins_m{m}_seed0")]


def curve(tmpl, key="ce_per_r"):
    return np.array([float(np.nanmean(np.load(tmpl.format(m=m) + "/eval_per_r.npz")[key]))
                     for m in MS])


def mstar(c, min_drop=2.0):
    pl = c[-2:].mean()
    if c[0] / pl < min_drop:
        return None
    h = np.sqrt(c[0] * pl)
    for i in range(len(c) - 1):
        if c[i] >= h >= c[i + 1]:
            f = (np.log(c[i]) - np.log(h)) / (np.log(c[i]) - np.log(c[i + 1]))
            return 10 ** (LM[i] + f * (LM[i + 1] - LM[i]))
    return None


syn = {j: curve("runs_divbins_320k/in64_bins_m{m}_seed0" if j == 1
                else f"runs_syn/syn{j}_m{{m}}_seed0") for j, _ in SYN}
binn = {nb: curve(t) for _, nb, t in BIN}

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1,
                     "xtick.direction": "in", "ytick.direction": "in"})
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

ax = axes[0]
xb = [nb for _, nb, _ in BIN if mstar(binn[nb])]
yb = [mstar(binn[nb]) for _, nb, _ in BIN if mstar(binn[nb])]
ax.plot(xb, yb, "o-", color="#b03a2e", ms=8, lw=1.8,
        label="more bins\n(vocabulary AND information change)")
xs = [v for j, v in SYN if mstar(syn[j])]
ys = [mstar(syn[j]) for j, v in SYN if mstar(syn[j])]
ax.plot(xs, ys, "s--", color="#1f3352", mfc="white", ms=8, lw=1.8,
        label="synonym splitting\n(vocabulary only)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1024"], fontsize=9)
ax.set_yticks([200, 400, 800]); ax.set_yticklabels(["200", "400", "800"])
ax.set_ylim(150, 1100)
ax.set_xlabel("input vocabulary size")
ax.set_ylabel("transition task count  $m^*$")
ax.set_title("Where the transition sits", fontsize=13)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")

ax = axes[1]
ax.plot([v for _, v in SYN], [syn[j][-2:].mean() for j, _ in SYN], "s-",
        color="#1f3352", mfc="white", ms=8, lw=1.8, label="plateau (new tasks)")
ax.plot([v for _, v in SYN],
        [curve("runs_divbins_320k/in64_bins_m{m}_seed0" if j == 1
               else f"runs_syn/syn{j}_m{{m}}_seed0", "ce_at_train_r")[-2:].mean()
         for j, _ in SYN], "^-", color="#7a8b99", ms=8, lw=1.8,
        label="plateau (seen tasks)")
ax.set_xscale("log", base=2)
ax.set_xticks([64, 128, 256, 512, 1024])
ax.set_xticklabels(["64", "128", "256", "512", "1024"])
ax.set_xlabel("input vocabulary size  (synonym splitting)")
ax.set_ylabel("mean cross-entropy at large $m$ (nats)")
ax.set_title("What vocabulary inflation does cost", fontsize=13)
ax.legend(frameon=False, fontsize=10, loc="upper left")

for a in axes:
    a.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.text(0.5, -0.06,
         "Splitting each of 64 bins into $j$ interchangeable tokens grows the vocabulary $j$-fold while leaving the information, "
         "the sequence length\nand the 64-bin output unchanged. The transition does not move; only the achievable floor drifts. "
         "All arms at 320k steps.",
         ha="center", fontsize=9.5, color="0.25")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"figures_taskdiv/synonyms.{e}", dpi=170, bbox_inches="tight")
print("wrote figures_taskdiv/synonyms.png")
print("\nseen-task plateau CE by j:")
for j, v in SYN:
    c = curve("runs_divbins_320k/in64_bins_m{m}_seed0" if j == 1
              else f"runs_syn/syn{j}_m{{m}}_seed0", "ce_at_train_r")
    print(f"  j={j:2d} vocab {v:5d}   seen {c[-2:].mean():.4f}   "
          f"new {syn[j][-2:].mean():.4f}")
