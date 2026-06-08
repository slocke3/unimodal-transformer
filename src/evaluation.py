import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats

from .maps import (
    iterate_map, iterate_general, tokenize_trajectory, detokenize,
    compute_lyapunov, compute_lyapunov_general, FAMILIES,
)


REGIME_COLORS = {
    "periodic":    "#2196F3",
    "bifurcation": "#FF9800",
    "chaotic":     "#F44336",
}


def classify_regime(lyapunov, tol=0.02):
    if lyapunov < -tol:
        return "periodic"
    elif abs(lyapunov) <= tol:
        return "bifurcation"
    return "chaotic"


@torch.no_grad()
def evaluate_per_r(model, r_grid, device, context_len, n_bins,
                   burn_in=0, n_eval_per_r=30, traj_len=150, seed=99):
    """
    Compute mean cross-entropy and top-1 accuracy per r value.
    Returns: (ce_per_r, acc_per_r) as numpy arrays of shape (len(r_grid),)
    """
    model.eval()
    rng = np.random.default_rng(seed)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    window_size = context_len + 1
    ce_per_r  = np.empty(len(r_grid))
    acc_per_r = np.empty(len(r_grid))

    for i, r in enumerate(r_grid):
        contexts, targets = [], []
        for _ in range(n_eval_per_r):
            x0 = rng.uniform(0.05, 0.95)
            traj = iterate_map(x0, r, burn_in + traj_len)[burn_in:]
            tokens = tokenize_trajectory(traj, n_bins)
            for t in range(len(tokens) - window_size):
                contexts.append(tokens[t : t + context_len])
                targets.append(tokens[t + context_len])

        ctx = torch.tensor(np.array(contexts), dtype=torch.long).to(device)
        tgt = torch.tensor(np.array(targets),  dtype=torch.long).to(device)
        logits = model(ctx)
        ce_per_r[i]  = criterion(logits, tgt).item()
        acc_per_r[i] = (logits.argmax(dim=-1) == tgt).float().mean().item()

        if (i + 1) % 100 == 0:
            print(f"  eval: {i+1}/{len(r_grid)}")

    return ce_per_r, acc_per_r


@torch.no_grad()
def evaluate_family(model, family_name, device, context_len, n_bins,
                    burn_in=0, traj_len=150, n_eval=30, seed=99):
    """Evaluate model on a named family from FAMILIES registry."""
    fam = FAMILIES[family_name]
    return evaluate_general(
        model=model, map_fn=fam["map_fn"], params=fam["params"],
        device=device, context_len=context_len, n_bins=n_bins,
        burn_in=burn_in, traj_len=traj_len, n_eval=n_eval, seed=seed,
    )


@torch.no_grad()
def evaluate_general(model, map_fn, params, device, context_len, n_bins,
                     burn_in=0, traj_len=150, n_eval=30, seed=99):
    """Evaluate model on trajectories from an arbitrary map family."""
    model.eval()
    rng = np.random.default_rng(seed)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    window_size = context_len + 1
    ce_per_param  = np.empty(len(params))
    acc_per_param = np.empty(len(params))

    for i, p in enumerate(params):
        contexts, targets = [], []
        for _ in range(n_eval):
            x0 = rng.uniform(0.05, 0.95)
            traj = iterate_general(x0, map_fn, p, burn_in + traj_len)[burn_in:]
            tokens = tokenize_trajectory(traj, n_bins)
            for t in range(len(tokens) - window_size):
                contexts.append(tokens[t : t + context_len])
                targets.append(tokens[t + context_len])

        ctx = torch.tensor(np.array(contexts), dtype=torch.long).to(device)
        tgt = torch.tensor(np.array(targets),  dtype=torch.long).to(device)
        logits = model(ctx)
        ce_per_param[i]  = criterion(logits, tgt).item()
        acc_per_param[i] = (logits.argmax(dim=-1) == tgt).float().mean().item()

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(params)}  p={p:.3f}  ce={ce_per_param[i]:.3f}")

    return ce_per_param, acc_per_param


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ce_vs_lyapunov(r_grid, lyapunovs, ce_per_r, save_path=None, figsize=(12, 5)):
    regimes = np.array([classify_regime(l) for l in lyapunovs])
    colors  = np.array([REGIME_COLORS[reg] for reg in regimes])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.scatter(r_grid, ce_per_r, c=colors, s=6, alpha=0.6, linewidths=0)
    ax2 = ax.twinx()
    ax2.plot(r_grid, lyapunovs, color="black", lw=0.8, alpha=0.4)
    ax2.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)
    ax2.set_ylabel(r"$\lambda(r)$", fontsize=10)
    ax.set_xlabel(r"Parameter $r$", fontsize=11)
    ax.set_ylabel("Cross-entropy (nats)", fontsize=11)
    ax.set_title("Prediction loss across bifurcation diagram", fontsize=11)
    for regime, color in REGIME_COLORS.items():
        ax.scatter([], [], c=color, s=20, label=regime.capitalize())
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for regime, color in REGIME_COLORS.items():
        mask = regimes == regime
        if mask.sum() > 0:
            ax.scatter(lyapunovs[mask], ce_per_r[mask],
                       c=color, s=8, alpha=0.5, linewidths=0, label=regime.capitalize())

    chaotic_mask = regimes == "chaotic"
    if chaotic_mask.sum() > 10:
        slope, intercept, r_val, _, _ = stats.linregress(
            lyapunovs[chaotic_mask], ce_per_r[chaotic_mask])
        x_line = np.linspace(lyapunovs[chaotic_mask].min(), lyapunovs[chaotic_mask].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5,
                label=fr"Fit (chaotic): slope={slope:.2f}, $R^2$={r_val**2:.2f}")

    ax.set_xlabel(r"$\lambda(r)$", fontsize=11)
    ax.set_ylabel("Cross-entropy (nats)", fontsize=11)
    ax.set_title(r"Cross-entropy vs $\lambda(r)$", fontsize=11)
    ax.legend(fontsize=8)
    ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history["train_losses"], label="Train", lw=1.5)
    if history["val_losses"]:
        ax.semilogy(history["val_losses"], label="Val", lw=1.5)
    ax.axvline(history["best_epoch"] - 1, color="gray", ls="--", lw=1,
               label=f"Best epoch ({history['best_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy (log scale)")
    ax.set_title("Training curves")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@torch.no_grad()
def plot_trajectory_comparison(model, r_values, device, context_len, n_bins,
                                rollout_steps=100, burn_in=0, seed=42, save_path=None):
    n = len(r_values)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(seed)
    model.eval()

    for ax, r in zip(axes, r_values):
        x0 = rng.uniform(0.1, 0.9)
        traj = iterate_map(x0, r, burn_in + context_len + rollout_steps)[burn_in:]
        tokens = tokenize_trajectory(traj, n_bins)

        context_tokens = tokens[:context_len]
        future_tokens  = tokens[context_len : context_len + rollout_steps]

        ctx_tensor = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(device)
        rollout_np = model.predict_rollout_greedy(ctx_tensor, n_steps=rollout_steps).squeeze(0).cpu().numpy()

        context_centers = detokenize(context_tokens, n_bins)
        future_centers  = detokenize(future_tokens, n_bins)
        rollout_centers = detokenize(rollout_np, n_bins)
        match_rate = np.mean(rollout_np == future_tokens)

        t_ctx    = np.arange(context_len)
        t_future = np.arange(context_len, context_len + rollout_steps)

        ax.plot(t_ctx,    context_centers, color="#1B2A4A", lw=1.2, alpha=0.6, label="Context")
        ax.plot(t_future, future_centers,  color="#1B2A4A", lw=1.5, label="True")
        ax.plot(t_future, rollout_centers, color="#378ADD", lw=1.5, alpha=0.8, label="Model")
        ax.axvline(context_len, color="gray", ls="--", lw=0.8, alpha=0.5)

        lya = compute_lyapunov(r)
        ax.set_title(rf"$r={r}$  |  $\lambda={lya:.3f}$  |  token match={match_rate:.1%}", fontsize=11)
        ax.set_ylabel("$x_n$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("Time step $n$")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig