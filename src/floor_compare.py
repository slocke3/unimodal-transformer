"""
Figure 1: transformer cross-entropy vs the operator floor (k-gram / Ulam), as a
function of the Lyapunov exponent, across generalization settings.

Both the transformer and the k-gram floor are trained on the QUADRATIC family.
We then evaluate both -- on identical test contexts -- in two panels:

  Panel 1  in-distribution : quadratic, r inside the training range
  Panel 2  out-of-family   : tent / sine / cubic (zero-shot transfer)

The story is the gap between the transformer's CE and (a) the lambda reference
line CE = lambda (the optimal predictor's loss in the chaotic regime) and (b)
the quadratic-fit operator floor -- and whether that gap survives going OOD.

torch is imported lazily (inside the one function that needs it) so this module
and its numpy pipeline import fine even where torch is unavailable; pass
model=None to compute the floor / lambda panels without a transformer.
"""
import numpy as np

from .maps import (
    compute_lyapunov, compute_lyapunov_general, FAMILIES,
)
from .baselines import KGramModel, gen_token_seqs, make_context_target_pairs

QUADRATIC = lambda x, r: r * x * (1.0 - x)


# ---------------------------------------------------------------------------
# Fit the quadratic operator floor (same training distribution as the model)
# ---------------------------------------------------------------------------

def fit_quadratic_floor(orders=(1, 2, 5), n_traj=2000, r_range=(0.5, 4.0),
                        n_bins=64, traj_len=150, burn_in=0, seed=0, alpha=1e-3):
    """Fit order-k k-gram models on quadratic trajectories drawn like training."""
    rng = np.random.default_rng(seed)
    rs = rng.uniform(r_range[0], r_range[1], size=n_traj)
    seqs = []
    for r in rs:
        seqs.extend(gen_token_seqs(QUADRATIC, r, 1, n_bins, traj_len, burn_in, rng))
    return {k: KGramModel(n_bins, k, alpha).fit(seqs) for k in orders}


# ---------------------------------------------------------------------------
# Transformer CE on a set of contexts (lazy torch)
# ---------------------------------------------------------------------------

def _transformer_ce(model, contexts, targets, device):
    import torch
    import torch.nn as nn
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        ctx = torch.as_tensor(np.asarray(contexts), dtype=torch.long, device=device)
        tgt = torch.as_tensor(np.asarray(targets), dtype=torch.long, device=device)
        return float(crit(model(ctx), tgt).item())


# ---------------------------------------------------------------------------
# Core: per-parameter transformer + floor CE on identical contexts
# ---------------------------------------------------------------------------

def compute_panel(params, map_fn, lambdas, floor_models, n_bins, context_len,
                  model=None, device="cpu", n_eval=20, traj_len=150,
                  burn_in=0, seed=99):
    """For each parameter: build eval pairs once, score transformer + each floor
    order on them, and record lambda. Returns a dict of arrays.

    model=None -> transformer CE is NaN (floor/lambda still computed).
    """
    rng = np.random.default_rng(seed)
    n = len(params)
    out = {
        "param": np.asarray(params, dtype=float),
        "lambda": np.asarray(lambdas, dtype=float),
        "ce_tf": np.full(n, np.nan),
        "ce_floor": {k: np.empty(n) for k in floor_models},
    }
    for i, p in enumerate(params):
        seqs = gen_token_seqs(map_fn, p, n_eval, n_bins, traj_len, burn_in, rng)
        ctx, tgt = make_context_target_pairs(seqs, context_len)
        if model is not None:
            out["ce_tf"][i] = _transformer_ce(model, ctx, tgt, device)
        for k, m in floor_models.items():
            out["ce_floor"][k][i] = m.cross_entropy(ctx, tgt)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n}")
    return out


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def panel_in_distribution(floor_models, n_bins, context_len, model=None, device="cpu",
                          r_grid=None, lambdas=None, n_eval=20, traj_len=150,
                          burn_in=0, lam_steps=20000, seed=99):
    """Panel 1: quadratic, r inside the training range."""
    if r_grid is None:
        r_grid = np.linspace(0.5, 4.0, 150)
    if lambdas is None:
        print("  computing lambda(r) grid...")
        lambdas = np.array([compute_lyapunov(r, n_steps=lam_steps) for r in r_grid])
    print("  in-distribution panel...")
    return compute_panel(r_grid, QUADRATIC, lambdas, floor_models, n_bins,
                         context_len, model, device, n_eval, traj_len, burn_in, seed)


def panel_family(family_name, floor_models, n_bins, context_len, model=None, device="cpu",
                 n_params=100, n_eval=20, traj_len=150, burn_in=0, lam_steps=20000, seed=99):
    """Panel 2 component: one out-of-family map (tent / sine / cubic)."""
    fam = FAMILIES[family_name]
    params = np.asarray(fam["params"])
    if n_params and n_params < len(params):
        params = params[np.linspace(0, len(params) - 1, n_params).astype(int)]
    print(f"  {family_name}: lambda...")
    lambdas = np.array([
        compute_lyapunov_general(fam["map_fn"], fam["deriv_fn"], p, n_steps=lam_steps)
        for p in params
    ])
    print(f"  {family_name}: CE...")
    data = compute_panel(params, fam["map_fn"], lambdas, floor_models, n_bins,
                        context_len, model, device, n_eval, traj_len, burn_in, seed)
    data["color"] = fam["color"]
    data["name"] = family_name
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fit_chaotic(ax, lam, ce, color="k", min_points=10):
    """Linear fit of CE vs lambda over the chaotic regime (lambda > 0).

    Mirrors evaluation.plot_ce_vs_lyapunov: dashed line + slope / R^2 label.
    """
    from scipy import stats
    lam = np.asarray(lam)
    ce = np.asarray(ce)
    mask = (lam > 0) & np.isfinite(ce)
    if mask.sum() < min_points:
        return
    slope, intercept, r_val, _, _ = stats.linregress(lam[mask], ce[mask])
    xs = np.linspace(0.0, lam[mask].max(), 50)
    ax.plot(xs, slope * xs + intercept, "--", color=color, lw=1.6, zorder=4,
            label=fr"Fit (chaotic): slope={slope:.2f}, $R^2$={r_val**2:.2f}")


def _draw_floor_and_ref(ax, lam, ce_floor, floor_orders):
    """Quadratic-fit floor line(s) (sorted by lambda) + CE=lambda reference."""
    order = np.argsort(lam)
    for k in floor_orders:
        ax.plot(lam[order], ce_floor[k][order], lw=1.4, alpha=0.9,
                label=f"{k}-gram floor")
    lo, hi = max(0.0, np.nanmin(lam)), np.nanmax(lam)
    xs = np.linspace(0.0, hi, 50)
    ax.plot(xs, xs, "k--", lw=1.0, alpha=0.6, label=r"$CE=\lambda$ (optimal)")
    ax.axvline(0, color="gray", lw=0.6, ls=":", alpha=0.5)


def plot_figure1(in_dist, families, floor_orders=(1, 5), n_bins=64,
                 save_path=None, figsize=(13, 5)):
    """Two panels: in-distribution (left), out-of-family (right).

    y-axis is clipped to ~ln(n_bins) (max sensible per-symbol entropy) so the
    out-of-family floor blow-up -- where the quadratic count-table assigns ~0
    probability to unseen transitions -- doesn't crush the readable range.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    y_top = float(np.log(n_bins)) + 0.4

    # Panel 1 -- in-distribution
    ax = axes[0]
    ax.scatter(in_dist["lambda"], in_dist["ce_tf"], s=10, alpha=0.7,
               color="#1B2A4A", label="Transformer", zorder=3)
    _fit_chaotic(ax, in_dist["lambda"], in_dist["ce_tf"], color="#1B2A4A")
    _draw_floor_and_ref(ax, in_dist["lambda"], in_dist["ce_floor"], floor_orders)
    ax.set_title("In-distribution (quadratic, trained $r$)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel 2 -- out-of-family (pool families)
    ax = axes[1]
    all_lam = np.concatenate([d["lambda"] for d in families])
    pooled_floor = {k: np.concatenate([d["ce_floor"][k] for d in families]) for k in floor_orders}
    for d in families:
        ax.scatter(d["lambda"], d["ce_tf"], s=10, alpha=0.6,
                   color=d["color"], label=f"{d['name']} (transformer)", zorder=3)
    all_tf = np.concatenate([d["ce_tf"] for d in families])
    _fit_chaotic(ax, all_lam, all_tf, color="#333333")
    _draw_floor_and_ref(ax, all_lam, pooled_floor, floor_orders)
    ax.set_title("Out-of-family (zero-shot: tent / sine / cubic)")
    ax.set_xlabel(r"$\lambda$")
    ax.legend(fontsize=8, loc="upper left")
    if np.nanmax(pooled_floor[max(floor_orders)]) > y_top:
        ax.text(0.97, 0.02, "quadratic count-table off-scale\n(counting fails to transfer)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#777")

    for ax in axes:
        ax.set_ylim(-0.1, y_top)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# One-call convenience (for a Colab cell)
# ---------------------------------------------------------------------------

def run_figure1(model, device, n_bins, context_len, burn_in=0,
                floor_orders=(1, 2, 5), families=("tent", "sine", "cubic"),
                n_eval=20, traj_len=150, in_dist_lambdas=None, r_grid=None,
                n_params=100, lam_steps=20000, save_path=None, seed=99):
    """Fit floor, compute both panels, plot. Returns (fig, in_dist, family_data)."""
    print("Fitting quadratic operator floor...")
    floor = fit_quadratic_floor(orders=floor_orders, n_bins=n_bins,
                                traj_len=traj_len, burn_in=burn_in, seed=seed)
    in_dist = panel_in_distribution(
        floor, n_bins, context_len, model, device, r_grid=r_grid,
        lambdas=in_dist_lambdas, n_eval=n_eval, traj_len=traj_len,
        burn_in=burn_in, lam_steps=lam_steps, seed=seed)
    fam_data = [
        panel_family(name, floor, n_bins, context_len, model, device,
                     n_params=n_params, n_eval=n_eval, traj_len=traj_len,
                     burn_in=burn_in, lam_steps=lam_steps, seed=seed)
        for name in families
    ]
    fig = plot_figure1(in_dist, fam_data, floor_orders=(min(floor_orders), max(floor_orders)),
                       n_bins=n_bins, save_path=save_path)
    return fig, in_dist, fam_data
