"""
In-context system identification: does the transformer infer *which* r it is
observing, and does that inference need context? This isolates whether the L=2
deficit in the N x L sweep is a system-ID failure (too little context to
identify r) rather than only a partition/Markov effect.

Probe: the implied one-step return map. Feed a model (trained across the family)
contexts drawn from a FIXED r and read off its implied expected next state
E[x_{n+1} | x_n] = sum_j p(j) * center(j). The true dynamics is deterministic,
so the correct answer is exactly the parabola f_r(x) = r x (1-x).

  * If the model has identified r, its implied map sits on that parabola.
  * If it has too little context, it hedges over its posterior on r, giving a
    *squashed* parabola (right shape, too low): a posterior average of f_r over r.

Three readouts across models trained at different L (the sweep checkpoints):
  1. return-map grid (rows=r, cols=L): watch the squashed parabola rise;
  2. deviation-vs-L: RMS distance to the true parabola (report vs binning floor);
  3. bias(r_hat - r)-vs-L: fit the implied map to the best r_hat -- bias shrinking
     = genuine system-ID; only scatter shrinking = mere noise reduction.

Binning note: N is FIXED across L, so the irreducible binning floor ~O(1/N) is a
constant offset that cancels out of the L-comparison. Bias is robust to it
(quantization is ~symmetric noise). torch is imported lazily so the numpy paths
(context generation, true map, best-fit r_hat, binning floor) test without it;
model=None returns the true parabola as the reference.
"""
import numpy as np

from .maps import iterate_map, tokenize_trajectory, detokenize, quadratic_map


def _model_next_dist(model, contexts, device):
    import torch
    with torch.no_grad():
        ctx = torch.as_tensor(np.asarray(contexts), dtype=torch.long, device=device)
        p = torch.softmax(model(ctx), dim=-1)
    return p.detach().cpu().numpy()


def implied_return_map(model, device, r, n_bins, context_len,
                       n_traj=40, traj_len=200, burn_in=50, seed=7):
    """Model's implied E[x_{n+1} | x_n] on contexts drawn from a fixed r.

    Returns (x_last, e_next), one entry per context window. model=None returns
    the true parabola f_r(x_last).
    """
    rng = np.random.default_rng(seed)
    contexts, last_x = [], []
    for _ in range(n_traj):
        x0 = rng.uniform(0.05, 0.95)
        traj = iterate_map(x0, r, burn_in + traj_len)[burn_in:]
        tok = tokenize_trajectory(traj, n_bins)
        for t in range(context_len, len(tok)):
            contexts.append(tok[t - context_len:t])
            last_x.append(detokenize(tok[t - 1], n_bins))
    contexts = np.asarray(contexts)
    last_x = np.asarray(last_x, dtype=float)
    if model is None:
        e_next = quadratic_map(last_x, r)
    else:
        centers = (np.arange(n_bins) + 0.5) / n_bins
        e_next = _model_next_dist(model, contexts, device) @ centers
    return last_x, e_next


def _true(x, r):
    return quadratic_map(np.asarray(x, dtype=float), r)


def deviation(last_x, e_next, r):
    """RMS distance of the implied map from the true parabola f_r."""
    return float(np.sqrt(np.mean((np.asarray(e_next) - _true(last_x, r)) ** 2)))


def best_fit_r(last_x, e_next, r_grid=None):
    """r_hat minimizing || e_next - f_r(x) ||^2 over a grid (bias = r_hat - r_true).

    f_r(x)=r*x(1-x) is linear in r, so the least-squares r_hat is closed-form:
    r_hat = <e_next, g> / <g, g> with g(x)=x(1-x).
    """
    x = np.asarray(last_x, dtype=float)
    g = x * (1.0 - x)
    denom = float(g @ g)
    if denom == 0.0:
        return np.nan
    return float((np.asarray(e_next) @ g) / denom)


def binning_floor(r, n_bins, n=20000, seed=0):
    """Irreducible RMS deviation from f_r for a *perfect* r-knowing predictor:
    input known only to bin-width and output quantized to bin centers."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    xq = detokenize(tokenize_trajectory(x, n_bins), n_bins)          # binned x_n
    true_next = _true(xq, r)
    yq = detokenize(tokenize_trajectory(np.clip(true_next, 0, 1), n_bins), n_bins)
    return float(np.sqrt(np.mean((yq - _true(xq, r)) ** 2)))


def run_system_id(models_by_L, device, n_bins, r_values,
                  n_traj=40, traj_len=200, burn_in=50, seed=7):
    """For each (L, r): implied map, deviation, and best-fit r_hat/bias.

    models_by_L : dict L -> model (each trained at context length L).
    Returns dict with 'maps', 'deviation', 'r_hat', 'bias', 'floor', 'Ls', 'rs'.
    """
    out = {"maps": {}, "deviation": {}, "r_hat": {}, "bias": {},
           "floor": {r: binning_floor(r, n_bins) for r in r_values},
           "Ls": sorted(models_by_L), "rs": list(r_values)}
    for L in sorted(models_by_L):
        out["maps"][L], out["deviation"][L] = {}, {}
        out["r_hat"][L], out["bias"][L] = {}, {}
        for r in r_values:
            xl, en = implied_return_map(models_by_L[L], device, r, n_bins, L,
                                        n_traj=n_traj, traj_len=traj_len,
                                        burn_in=burn_in, seed=seed)
            rhat = best_fit_r(xl, en)
            out["maps"][L][r] = (xl, en)
            out["deviation"][L][r] = deviation(xl, en, r)
            out["r_hat"][L][r] = rhat
            out["bias"][L][r] = rhat - r
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_return_map_grid(result, save_path=None):
    """Rows = r, cols = L. Scatter of implied map + true parabola (orange)."""
    import matplotlib.pyplot as plt
    Ls, rs = result["Ls"], result["rs"]
    fig, axes = plt.subplots(len(rs), len(Ls), figsize=(2.6 * len(Ls), 2.6 * len(rs)),
                             squeeze=False, sharex=True, sharey=True)
    xs = np.linspace(0, 1, 200)
    for i, r in enumerate(rs):
        for j, L in enumerate(Ls):
            ax = axes[i][j]
            xl, en = result["maps"][L][r]
            ax.scatter(xl, en, s=3, alpha=0.25, color="#1B2A4A", linewidths=0)
            ax.plot(xs, quadratic_map(xs, r), color="#D85A30", lw=1.3)
            ax.set_title(f"L={L}, r={r:g}\nRMS={result['deviation'][L][r]:.3f}, "
                         f"$\\hat r$={result['r_hat'][L][r]:.2f}", fontsize=7.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("$x_n$", fontsize=9)
    for row in axes:
        row[0].set_ylabel("$E[x_{n+1}\\mid x_n]$", fontsize=9)
    fig.suptitle("Implied return map vs context length (orange = true $f_r$)", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_deviation_and_bias(result, save_path=None, figsize=(11, 4.2)):
    """Left: deviation-vs-L (dashed = binning floor). Right: bias(r_hat-r)-vs-L."""
    import matplotlib.pyplot as plt
    Ls, rs = result["Ls"], result["rs"]
    fig, (axd, axb) = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap("plasma")
    col = {r: cmap(k / max(1, len(rs) - 1)) for k, r in enumerate(rs)}

    for r in rs:
        devs = [result["deviation"][L][r] for L in Ls]
        axd.plot(Ls, devs, "o-", color=col[r], lw=1.6, label=f"r = {r:g}")
        axd.axhline(result["floor"][r], color=col[r], ls=":", lw=1, alpha=0.7)
        axb.plot(Ls, [result["bias"][L][r] for L in Ls], "o-", color=col[r],
                 lw=1.6, label=f"r = {r:g}")

    for ax in (axd, axb):
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
        ax.set_xlabel("context length $L$")
        ax.legend(fontsize=8, title="parameter")
    axd.set_ylabel("RMS deviation from $f_r$")
    axd.set_title("Deviation vs $L$ (dotted = binning floor)")
    axb.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    axb.set_ylabel(r"bias $\hat r - r$")
    axb.set_title("System-ID bias vs $L$ (shrinking $=$ real ID)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
