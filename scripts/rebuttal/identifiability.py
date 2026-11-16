"""How much does a K-token context say about r and about the next state?
(reviewers FWzs on context length, pqmj on connecting results to the dynamics)

Exact set-membership computation for the logistic family observed through N
uniform bins. Given the last K tokens c_1..c_K of a context, a parameter r is
consistent if some state in bin c_1 produces bins c_2..c_K under f_r. For fixed
r the consistent current states form ONE interval S_K(r): S_1 = bin(c_1) and
S_j = f_r(S_{j-1}) intersected with bin(c_j), and the continuous image of an
interval is an interval. So consistency is exact on a fine r-grid (no
sampling), and the next state is known to lie in f_r(S_K(r)) for consistent r.

Reported per context, as a function of K, resolution N and regime. (An earlier
version also fit a per-context shrink rate against the Lyapunov exponent; the
per-context rates did not track lambda, so that panel was removed.) Both are
support statements (what the tokens rule in or out), not posterior averages, so
they need no prior over r or over where on its orbit the state is:
  r-measure     total length of the consistent r set (grid count x spacing). In
                the chaotic band this set breaks into many pieces, so its total
                length, not its max - min, is what shrinks.
  next-state    width of the union over consistent r of f_r(S_K(r)): the range
  range         of next states the context still allows. Dashed reference: the
                range allowed when r is known and only the current bin is,
                |f_r(bin)|.

Contexts are training-like windows: r ~ U(0.5, 4), x0 ~ U(0.05, 0.95), a random
window start in [0, 99]; the K tokens are the last K of a 50-token context.
The r-grid spacing is 2e-5; contexts whose consistent set is narrower than that
come back empty and are counted, not dropped silently.

Torch-free:  python scripts/rebuttal/identifiability.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.maps import tokenize_trajectory  # noqa: E402

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15.5, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
    "axes.linewidth": 1.3,
})

L, TRAJ_LEN = 50, 150
REGIMES = [("fixed point, r < 3", 0.5, 3.0), ("periodic, 3 < r < 3.57", 3.0, 3.5699),
           ("chaotic band, r > 3.57", 3.5699, 4.0)]


def image(lo, hi, r):
    """f_r([lo, hi]) for f_r(x) = r x (1 - x), elementwise."""
    flo, fhi = r * lo * (1 - lo), r * hi * (1 - hi)
    mx = np.where((lo < 0.5) & (hi > 0.5), r / 4, np.maximum(flo, fhi))
    return np.minimum(flo, fhi), mx


def sample_contexts(n, seed):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 4.0, n)
    x = rng.uniform(0.05, 0.95, n)
    traj = np.empty((n, TRAJ_LEN + 1))
    traj[:, 0] = x
    for i in range(TRAJ_LEN):
        x = r * x * (1 - x)
        traj[:, i + 1] = x
    t = rng.integers(0, TRAJ_LEN + 1 - (L + 1), n)
    win = np.take_along_axis(traj, t[:, None] + np.arange(L + 1)[None, :], axis=1)
    return win[:, :L], win[:, L], r


def filter_contexts(tok, n, rgrid, chunk=12):
    """For tokens (n_ctx, K): measure of consistent r set, next-state range."""
    dr = rgrid[1] - rgrid[0]
    out_w, out_pred = [], []
    for i in range(0, len(tok), chunk):
        c = tok[i:i + chunk]                                     # (b, K)
        R = rgrid[None, :]
        lo = np.repeat((c[:, :1] / n), len(rgrid), 1)
        hi = np.repeat(((c[:, :1] + 1) / n), len(rgrid), 1)
        ok = np.ones_like(lo, dtype=bool)
        for j in range(1, c.shape[1]):
            mn, mx = image(lo, hi, R)
            lo = np.maximum(mn, c[:, j:j + 1] / n)
            hi = np.minimum(mx, (c[:, j:j + 1] + 1) / n)
            ok &= lo <= hi
        mn, mx = image(lo, hi, R)
        cnt = ok.sum(1)
        with np.errstate(all="ignore"):
            rng_next = (np.nanmax(np.where(ok, mx, np.nan), 1)
                        - np.nanmin(np.where(ok, mn, np.nan), 1))
        out_w.append(np.where(cnt > 0, cnt * dr, np.nan))
        out_pred.append(rng_next)
    return np.concatenate(out_w), np.concatenate(out_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10])
    ap.add_argument("--n_ctx", type=int, default=1500)
    ap.add_argument("--dr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures_rebuttal/identifiability.pdf")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rgrid = np.arange(0.5, 4.0 + a.dr / 2, a.dr)
    ctx, y, r = sample_contexts(a.n_ctx, a.seed)

    res = {}
    lines = [f"{a.n_ctx} contexts, r-grid spacing {a.dr:g}"]
    for n in a.ns:
        tok = tokenize_trajectory(ctx, n)
        # reference: r known, only the current bin
        lo, hi = tok[:, -1] / n, (tok[:, -1] + 1) / n
        mn, mx = image(lo, hi, r)
        res[(n, "floor")] = mx - mn
        for k in a.ks:
            w, rng_next = filter_contexts(tok[:, -k:], n, rgrid)
            res[(n, k)] = (w, rng_next)
            lines.append(f"N={n:3d} K={k:2d}  median r-measure "
                         + "  ".join(f"{nm.split(',')[0]}: {np.nanmedian(w[(r >= lo_) & (r < hi_)]):.2e}"
                                     for nm, lo_, hi_ in REGIMES)
                         + f" | median next-state range {np.nanmedian(rng_next):.2e}"
                         + f" (r known: {np.median(mx - mn):.2e}; one output bin of 64: {1/64:.2e})"
                         + f" | empty {np.isnan(w).mean():.1%}")
            print(lines[-1], flush=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(a.ns)))
    reg_ls = ["-", "--", ":"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    ax = axes[0]                     # r-width vs K, by regime, N = middle
    n_mid = a.ns[len(a.ns) // 2]
    for (nm, lo_, hi_), ls in zip(REGIMES, reg_ls):
        sel = (r >= lo_) & (r < hi_)
        ax.plot(a.ks, [np.nanmedian(res[(n_mid, k)][0][sel]) for k in a.ks], ls,
                marker="o", color="0.15", lw=2.2, label=nm)
    ax.axhline(a.dr, color="0.6", lw=1, ls="-.")
    ax.text(a.ks[-1], a.dr * 1.4, "grid spacing", ha="right", color="0.45", fontsize=12)
    ax.set_yscale("log"); ax.set_xlabel("context tokens $K$")
    ax.set_ylabel("total length of consistent $r$ set (median)")
    ax.set_title(f"(a) how fast the context pins down $r$ ({n_mid} bins)")
    ax.legend(); ax.grid(alpha=0.2, which="both")

    ax = axes[1]                     # next-state range vs K, per N, with r-known reference
    for n, col in zip(a.ns, colors):
        ax.plot(a.ks, [np.nanmedian(res[(n, k)][1]) for k in a.ks], "-o", color=col, lw=2.2,
                label=f"{n} bins")
        ax.axhline(np.median(res[(n, "floor")]), color=col, lw=1.2, ls="--")
    ax.axhline(1 / 64, color="0.5", lw=1.0, ls=":")
    ax.text(a.ks[-1], 1 / 64 * 1.15, "one output bin", ha="right", color="0.45", fontsize=12)
    ax.set_yscale("log"); ax.set_xlabel("context tokens $K$")
    ax.set_ylabel("range of next states consistent with context")
    ax.set_title("(b) next-state uncertainty (dashed: $r$ known)")
    ax.legend(); ax.grid(alpha=0.2, which="both")

    np.savez(os.path.splitext(a.out)[0] + "_data.npz", r=r, ks=np.array(a.ks),
             ns=np.array(a.ns), **{f"N{n}_K{k}_rmeasure": res[(n, k)][0] for n in a.ns for k in a.ks},
             **{f"N{n}_K{k}_nextrange": res[(n, k)][1] for n in a.ns for k in a.ks},
             **{f"N{n}_rknown_range": res[(n, "floor")] for n in a.ns})
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    png = os.path.splitext(a.out)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    with open(os.path.splitext(a.out)[0] + "_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(lines[-1])
    print(f"wrote {a.out}, {png}")


if __name__ == "__main__":
    main()
