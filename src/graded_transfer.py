"""
Graded density-warp transfer (the h_eps experiment).

Hold the DYNAMICS fixed and slide only the invariant density, then measure how
the logistic-trained model's transfer cross-entropy responds. We warp logistic
r=4 orbits by

    h_eps(x) = (1-eps) x + eps * (2/pi) arcsin(sqrt(x)),

which interpolates from the identity (eps=0: density arcsine == logistic) to the
logistic->tent conjugacy (eps=1: density uniform == tent). Every warped sequence
is an orbit of a map CONJUGATE to logistic (same kneading / same dynamics), so
the "correct" transfer is perfect for all eps -- any CE rise is therefore a
representation artifact of the (uniform) tokenization, not a real dynamical
difference.

Bonus: because eps=1 orbits are warped logistic orbits (not iterated tent), we
avoid the tent float precision-collapse entirely -- a cleaner tent-density
measurement than the original Figure 1.

torch is lazy (in the CE path); pass model=None for the torch-free
unseen-mass / occupancy curves.
"""
import numpy as np

from src.occupancy import (
    _logistic_orbits, occupancy, transition_matrix_counts,
    unseen_transition_mass, _windows_from_orbits,
)


def warp_h(x, eps):
    """h_eps: identity (eps=0) -> logistic->tent conjugacy (eps=1). Fixes 0,1/2,1."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return (1.0 - eps) * x + eps * (2.0 / np.pi) * np.arcsin(np.sqrt(x))


def _ce_on_orbits(model, orbits, n_bins, context_len, device, batch_size=2048):
    """Mean next-token CE, batched over windows to avoid CUDA OOM."""
    import torch
    import torch.nn as nn
    ctx, tgt = _windows_from_orbits(orbits, n_bins, context_len)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, n = 0.0, len(tgt)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            c = torch.as_tensor(ctx[i:i + batch_size], dtype=torch.long, device=device)
            t = torch.as_tensor(tgt[i:i + batch_size], dtype=torch.long, device=device)
            total += float(crit(model(c), t).item())
    return total / max(n, 1)


def graded_density_transfer(model, device, n_bins=64, eval_context=50,
                            eps_values=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                            n_traj=500, traj_len=200, burn_in=50, seed=0):
    """Warp logistic r=4 orbits by h_eps and measure transfer CE / OOD-ness vs eps.

    model=None -> CE is NaN (unseen-mass and occupancy still computed).
    """
    base = _logistic_orbits([4.0], n_traj, traj_len, burn_in, seed)
    pool = _logistic_orbits(list(np.linspace(0.5, 4.0, 200)), 30, 150, 50, seed + 1)
    T_pool = transition_matrix_counts(pool, n_bins)
    occ_ref = occupancy(base, n_bins)  # logistic r=4 (== eps=0)

    res = {"eps": [], "ce": [], "unseen": [], "occ_l1": []}
    for eps in eps_values:
        warped = [warp_h(o, eps) for o in base]
        occ = occupancy(warped, n_bins)
        res["eps"].append(float(eps))
        res["occ_l1"].append(float(0.5 * np.abs(occ - occ_ref).sum()))
        res["unseen"].append(unseen_transition_mass(transition_matrix_counts(warped, n_bins), T_pool))
        res["ce"].append(_ce_on_orbits(model, warped, n_bins, eval_context, device)
                         if model is not None else float("nan"))
        print(f"  eps={eps:.1f}  unseen={res['unseen'][-1]:.1%}  occL1={res['occ_l1'][-1]:.3f}"
              f"  CE={res['ce'][-1]:.3f}")
    return res


def plot_graded(res, save_path=None):
    import matplotlib.pyplot as plt
    eps = res["eps"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    a1.plot(eps, res["ce"], "o-", color="#1B2A4A", lw=1.8, label="transfer CE")
    a1.set_xlabel(r"warp $\varepsilon$   (0 = logistic, 1 = tent density)")
    a1.set_ylabel("transfer CE (nats)")
    a1.set_title("Transfer CE vs density warp\n(dynamics held identical)")
    a1b = a1.twinx()
    a1b.plot(eps, [u * 100 for u in res["unseen"]], "s--", color="#D62728", alpha=0.6)
    a1b.set_ylabel("% transitions unseen in training", color="#D62728")

    a2.plot(res["occ_l1"], res["ce"], "o-", color="#1B2A4A", lw=1.8)
    for x, y, e in zip(res["occ_l1"], res["ce"], eps):
        a2.annotate(f"{e:.1f}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a2.set_xlabel("occupancy $L_1$ distance to logistic $r=4$ (statistical distance)")
    a2.set_ylabel("transfer CE (nats)")
    a2.set_title("Transfer CE vs statistical distance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
