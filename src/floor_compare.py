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

def compute_panel(params, map_fn, lambdas, n_bins, context_len, orders=(1,),
                  model=None, device="cpu", n_eval=20, n_fit=40,
                  traj_len=150, burn_in=0, alpha=1e-3, seed=99):
    """Per-parameter ORACLE floor + transformer CE on identical test contexts.

    At each parameter we fit a fresh order-k k-gram on that system's own training
    trajectories (the achievable finite-order floor / Ulam transfer operator at
    that point), then score it -- and the transformer -- on separate held-out
    test trajectories from the same system. So the floor is the best a counting
    model could do *if it knew the system*, and the transformer (quadratic-
    trained, inferring the system in-context) is compared against it.

    model=None -> transformer CE is NaN (floor/lambda still computed).
    """
    n = len(params)
    out = {
        "param": np.asarray(params, dtype=float),
        "lambda": np.asarray(lambdas, dtype=float),
        "ce_tf": np.full(n, np.nan),
        "ce_floor": {k: np.empty(n) for k in orders},
    }
    for i, p in enumerate(params):
        rng_fit = np.random.default_rng(seed * 100003 + i)
        rng_eval = np.random.default_rng(seed * 100003 + i + 50000)
        train = gen_token_seqs(map_fn, p, n_fit, n_bins, traj_len, burn_in, rng_fit)
        oracle = {k: KGramModel(n_bins, k, alpha).fit(train) for k in orders}
        test = gen_token_seqs(map_fn, p, n_eval, n_bins, traj_len, burn_in, rng_eval)
        ctx, tgt = make_context_target_pairs(test, context_len)
        if model is not None:
            out["ce_tf"][i] = _transformer_ce(model, ctx, tgt, device)
        for k in orders:
            out["ce_floor"][k][i] = oracle[k].cross_entropy(ctx, tgt)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n}")
    return out


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def panel_in_distribution(n_bins, context_len, orders=(1,), model=None, device="cpu",
                          r_grid=None, lambdas=None, n_eval=20, n_fit=40,
                          traj_len=150, burn_in=0, lam_steps=20000, seed=99):
    """Panel 1: quadratic, r inside the training range."""
    if r_grid is None:
        r_grid = np.linspace(0.5, 4.0, 150)
    if lambdas is None:
        print("  computing lambda(r) grid...")
        lambdas = np.array([compute_lyapunov(r, n_steps=lam_steps) for r in r_grid])
    print("  in-distribution panel...")
    return compute_panel(r_grid, QUADRATIC, lambdas, n_bins, context_len, orders,
                         model, device, n_eval, n_fit, traj_len, burn_in, seed=seed)


def panel_family(family_name, n_bins, context_len, orders=(1,), model=None, device="cpu",
                 n_params=100, n_eval=20, n_fit=40, traj_len=150, burn_in=0,
                 lam_steps=20000, seed=99):
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
    data = compute_panel(params, fam["map_fn"], lambdas, n_bins, context_len, orders,
                        model, device, n_eval, n_fit, traj_len, burn_in, seed=seed)
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


def _binned_median(lam, ce, n_lam_bins=28):
    """Median CE within equal-width lambda bins -- a smooth floor curve that
    avoids connecting the (multivalued in lambda) per-parameter floor points."""
    lam, ce = np.asarray(lam), np.asarray(ce)
    m = np.isfinite(lam) & np.isfinite(ce)
    lam, ce = lam[m], ce[m]
    if len(lam) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(lam.min(), lam.max(), n_lam_bins + 1)
    idx = np.clip(np.digitize(lam, edges) - 1, 0, n_lam_bins - 1)
    centers, meds = [], []
    for b in range(n_lam_bins):
        sel = idx == b
        if sel.any():
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            meds.append(np.median(ce[sel]))
    return np.array(centers), np.array(meds)


_FLOOR_STYLE = {1: dict(color="#1D9E75", lw=2.0),
                2: dict(color="#D85A30", lw=2.0),
                5: dict(color="#9467bd", lw=2.0)}


def _floor_label(k):
    return "Transfer-operator floor (1-gram)" if k == 1 else f"{k}-gram floor"


def _draw_floor(ax, lam, ce_floor, orders):
    for k in orders:
        c, med = _binned_median(lam, ce_floor[k])
        if len(c):
            ax.plot(c, med, alpha=0.9, label=_floor_label(k),
                    **_FLOOR_STYLE.get(k, dict(lw=2.0)))


def _draw_ref(ax, lam_max):
    xs = np.linspace(0.0, max(lam_max, 0.01), 50)
    ax.plot(xs, xs, "k--", lw=1.0, alpha=0.6, label=r"$CE=\lambda$ (optimal)")
    ax.axvline(0, color="gray", lw=0.6, ls=":", alpha=0.5)


def plot_figure1(in_dist, families, floor_orders=(1,), save_path=None, figsize=(13, 5)):
    """Two panels: in-distribution (left), out-of-family (right).

    Floor = per-parameter oracle (drawn as a lambda-binned median curve).
    In-distribution keeps the chaotic CE-vs-lambda fit (slope / R^2); the
    out-of-family panel shows points + floor + the lambda reference (per-family
    regressions are unstable over each family's narrow chaotic range).
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Panel 1 -- in-distribution
    ax = axes[0]
    ax.scatter(in_dist["lambda"], in_dist["ce_tf"], s=12, alpha=0.7,
               color="#1B2A4A", label="Transformer", zorder=3)
    _fit_chaotic(ax, in_dist["lambda"], in_dist["ce_tf"], color="#1B2A4A")
    _draw_floor(ax, in_dist["lambda"], in_dist["ce_floor"], floor_orders)
    _draw_ref(ax, np.nanmax(in_dist["lambda"]))
    ax.set_title("In-distribution (quadratic, trained $r$)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel 2 -- out-of-family
    ax = axes[1]
    all_lam = np.concatenate([d["lambda"] for d in families])
    pooled_floor = {k: np.concatenate([d["ce_floor"][k] for d in families]) for k in floor_orders}
    for d in families:
        ax.scatter(d["lambda"], d["ce_tf"], s=12, alpha=0.6,
                   color=d["color"], label=f"{d['name']}", zorder=3)
    _draw_floor(ax, all_lam, pooled_floor, floor_orders)
    _draw_ref(ax, np.nanmax(all_lam))
    ax.set_title("Out-of-family (zero-shot: tent / sine / cubic)")
    ax.set_xlabel(r"$\lambda$")
    ax.legend(fontsize=8, loc="upper left")

    # data-driven y-limit (robust to any residual outliers)
    cand = [in_dist["ce_tf"]] + [d["ce_tf"] for d in families]
    cand += [pooled_floor[k] for k in floor_orders] + [in_dist["ce_floor"][k] for k in floor_orders]
    flat = np.concatenate([np.asarray(c)[np.isfinite(c)] for c in cand])
    y_top = float(np.nanpercentile(flat, 99)) * 1.15 if len(flat) else 1.2
    for ax in axes:
        ax.set_ylim(-0.05, max(y_top, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# One-call convenience (for a Colab cell)
# ---------------------------------------------------------------------------

def run_figure1(model, device, n_bins, context_len, burn_in=0,
                floor_orders=(1,), families=("tent", "sine", "cubic"),
                n_eval=20, n_fit=40, traj_len=150, in_dist_lambdas=None, r_grid=None,
                n_params=100, lam_steps=20000, save_path=None, seed=99):
    """Compute both panels (per-parameter oracle floor) and plot.

    Returns (fig, in_dist, family_data). floor_orders=(1,) gives the order-1
    transfer-operator floor; add 2 for a higher-order line (needs larger n_fit).
    """
    in_dist = panel_in_distribution(
        n_bins, context_len, orders=floor_orders, model=model, device=device,
        r_grid=r_grid, lambdas=in_dist_lambdas, n_eval=n_eval, n_fit=n_fit,
        traj_len=traj_len, burn_in=burn_in, lam_steps=lam_steps, seed=seed)
    fam_data = [
        panel_family(name, n_bins, context_len, orders=floor_orders, model=model,
                     device=device, n_params=n_params, n_eval=n_eval, n_fit=n_fit,
                     traj_len=traj_len, burn_in=burn_in, lam_steps=lam_steps, seed=seed)
        for name in families
    ]
    fig = plot_figure1(in_dist, fam_data, floor_orders=floor_orders, save_path=save_path)
    return fig, in_dist, fam_data
