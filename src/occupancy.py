"""
Bin-occupancy diagnostic for the out-of-family CE blow-up (torch-free).

Hypothesis: the quadratic-trained model blows up on tent orbits because, under a
uniform partition, tent orbits occupy bins / make transitions that logistic
orbits essentially never do -- so the model was trained to assign them ~0
probability, giving huge cross-entropy.

We check three things, all without the model:
  1. marginal bin occupancy of tent(s=2) vs logistic(r=4) vs logistic pooled
     over the training range -- does tent put mass where logistic does not?
  2. fraction of tent's order-1 transition mass that never occurs in the
     logistic training pool (an OOD lower bound: the model literally never saw
     these transitions, so it cannot have learned to predict them);
  3. the conjugacy preview: mapping tent(s=2) through h(x)=sin^2(pi x/2) into
     logistic r=4 coordinates should make its occupancy MATCH logistic r=4 --
     the coordinate fix, visible at the distribution level.
"""
import numpy as np

from src.maps import iterate_general, iterate_map, tent_map, tokenize_trajectory


def conjugacy_h(x):
    """Tent(s=2) -> logistic(r=4) conjugacy: h(x) = sin^2(pi x / 2)."""
    return np.sin(np.pi * np.asarray(x) / 2.0) ** 2


def _tent_orbits(n_traj, traj_len, seed, s=2.0):
    """Short tent(s=2) orbits (float precision collapses ~52 steps, like doubling)."""
    rng = np.random.default_rng(seed)
    orbits = []
    for _ in range(n_traj):
        x0 = rng.uniform(1e-6, 1 - 1e-6)
        orbits.append(iterate_general(x0, tent_map, s, traj_len))
    return orbits


def _logistic_orbits(params, n_per, traj_len, burn_in, seed):
    rng = np.random.default_rng(seed)
    orbits = []
    for r in params:
        for _ in range(n_per):
            x0 = rng.uniform(0.05, 0.95)
            orbits.append(iterate_map(x0, r, burn_in + traj_len)[burn_in:])
    return orbits


def occupancy(orbits, n_bins):
    counts = np.zeros(n_bins)
    for o in orbits:
        np.add.at(counts, tokenize_trajectory(np.asarray(o), n_bins), 1)
    return counts / counts.sum()


def transition_matrix_counts(orbits, n_bins):
    T = np.zeros((n_bins, n_bins))
    for o in orbits:
        tok = tokenize_trajectory(np.asarray(o), n_bins)
        for i, j in zip(tok[:-1], tok[1:]):
            T[i, j] += 1
    return T


def unseen_transition_mass(T_test, T_train):
    """Fraction of T_test's transition mass on (i,j) pairs with zero train count."""
    total = T_test.sum()
    if total == 0:
        return np.nan
    unseen = T_test[(T_train == 0)].sum()
    return float(unseen / total)


def transition_diagnostic(n_bins=64, traj_len=25, seed=0):
    """Return the transition matrices needed to *visualize* the 78% unseen mass:
    tent's moves, and the pooled-logistic (all r) transition support they are
    scored against. Also the h(tent) control."""
    train_r = np.linspace(0.5, 4.0, 200)
    tent = _tent_orbits(n_traj=6000, traj_len=traj_len, seed=seed)
    tent_h = [conjugacy_h(o) for o in tent]
    log_pool = _logistic_orbits(train_r, n_per=30, traj_len=150, burn_in=50, seed=seed + 2)

    T_tent = transition_matrix_counts(tent, n_bins)
    T_tenth = transition_matrix_counts(tent_h, n_bins)
    T_pool = transition_matrix_counts(log_pool, n_bins)
    return {
        "n_bins": n_bins,
        "T_tent": T_tent, "T_tent_h": T_tenth, "T_pool": T_pool,
        "unseen_tent": unseen_transition_mass(T_tent, T_pool),
        "unseen_tent_h": unseen_transition_mass(T_tenth, T_pool),
    }


def plot_transition_diagnostic(result, save_path=None):
    """Three panels of the (i -> j) transition plane at bin resolution N:
      (1) pooled-logistic support (every move any training r makes),
      (2) tent's moves colored seen (grey) vs UNSEEN-in-training (red),
      (3) h(tent)'s moves -- all seen (the conjugacy fix).
    The red mass in panel 2 is the 78%."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    N = result["n_bins"]
    Tt, Th, Tp = result["T_tent"], result["T_tent_h"], result["T_pool"]
    seen_pool = Tp > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # transpose so current bin i is on x, next bin j is on y (reads as j = f(i))
    axes[0].imshow(seen_pool.T, origin="lower", cmap="Greys", aspect="auto")
    axes[0].set_title(f"Logistic pooled (all $r$): support\n({seen_pool.mean():.0%} of cells ever used)")

    # tent: 0=absent, 1=seen-in-pool, 2=UNSEEN-in-pool
    cat = np.zeros((N, N))
    cat[(Tt > 0) & seen_pool] = 1
    cat[(Tt > 0) & ~seen_pool] = 2
    cmap = ListedColormap(["white", "#999999", "#D62728"])
    axes[1].imshow(cat.T, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_title(f"Tent moves: red = UNSEEN in training\n({result['unseen_tent']:.0%} of tent's transition mass)")

    cath = np.zeros((N, N))
    cath[(Th > 0) & seen_pool] = 1
    cath[(Th > 0) & ~seen_pool] = 2
    axes[2].imshow(cath.T, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[2].set_title(f"$h$(tent) moves after conjugacy\n({result['unseen_tent_h']:.0%} unseen -- the fix)")

    for ax in axes:
        ax.set_xlabel("current bin $i$"); ax.set_ylabel("next bin $j$")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_transition_overlay(result, save_path=None):
    """Overlay the logistic support and the tent moves on the SAME axes so the
    non-overlap is visible. Left: logistic (light) vs tent (red = off-support).
    Right: logistic (light) vs h(tent) (bent onto the support)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    N = result["n_bins"]
    Tt, Th, Tp = result["T_tent"], result["T_tent_h"], result["T_pool"]
    S = Tp > 0
    # 0 neither, 1 logistic-only, 2 overlap, 3 map-off-support
    cmap = ListedColormap(["white", "#Bcd4ef".replace("Bc", "bc"), "#1B2A4A", "#D62728"])

    def layer(M):
        c = np.zeros((N, N))
        c[S & ~M] = 1
        c[S & M] = 2
        c[~S & M] = 3
        return c.T  # current on x, next on y

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    a1.imshow(layer(Tt > 0), origin="lower", cmap=cmap, vmin=0, vmax=3, aspect="auto")
    a1.set_title(f"Tent vs logistic: {result['unseen_tent']:.0%} of tent's moves\n"
                 f"fall OUTSIDE the logistic support (red)")
    a2.imshow(layer(Th > 0), origin="lower", cmap=cmap, vmin=0, vmax=3, aspect="auto")
    a2.set_title(f"After conjugacy $h$: {result['unseen_tent_h']:.0%} outside\n"
                 f"($h$(tent) lands on the logistic parabola)")
    legend = [Patch(color="#bcd4ef", label="logistic support (all $r$)"),
              Patch(color="#1B2A4A", label="overlap"),
              Patch(color="#D62728", label="map move, unseen in training")]
    for ax in (a1, a2):
        ax.set_xlabel("current bin $i$"); ax.set_ylabel("next bin $j$")
    a1.legend(handles=legend, fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_occupancy_diagnostic(n_bins=64, traj_len=25, seed=0):
    """Compute the three checks. Returns a dict of arrays/scalars for plotting."""
    train_r = np.linspace(0.5, 4.0, 200)

    tent = _tent_orbits(n_traj=6000, traj_len=traj_len, seed=seed)
    tent_h = [conjugacy_h(o) for o in tent]                       # -> logistic r=4 coords
    log4 = _logistic_orbits([4.0], n_per=6000, traj_len=150, burn_in=50, seed=seed + 1)
    log_pool = _logistic_orbits(train_r, n_per=30, traj_len=150, burn_in=50, seed=seed + 2)

    occ = {
        "tent":    occupancy(tent, n_bins),
        "tent_h":  occupancy(tent_h, n_bins),
        "log4":    occupancy(log4, n_bins),
        "logpool": occupancy(log_pool, n_bins),
    }

    T_tent = transition_matrix_counts(tent, n_bins)
    T_tenth = transition_matrix_counts(tent_h, n_bins)
    T_pool = transition_matrix_counts(log_pool, n_bins)

    out = {
        "n_bins": n_bins,
        "occ": occ,
        "unseen_tent":   unseen_transition_mass(T_tent, T_pool),
        "unseen_tent_h": unseen_transition_mass(T_tenth, T_pool),
        "occ_L1_tent_vs_log4":   float(0.5 * np.abs(occ["tent"] - occ["log4"]).sum()),
        "occ_L1_tenth_vs_log4":  float(0.5 * np.abs(occ["tent_h"] - occ["log4"]).sum()),
    }
    return out


def _windows_from_orbits(orbits, n_bins, context_len):
    ctx, tgt = [], []
    for o in orbits:
        tok = tokenize_trajectory(np.asarray(o), n_bins)
        for t in range(len(tok) - context_len - 1):
            ctx.append(tok[t:t + context_len]); tgt.append(tok[t + context_len])
    return np.asarray(ctx), np.asarray(tgt)


def conjugacy_ce_test(model, device, n_bins, eval_context=30,
                      n_traj=3000, traj_len=35, seed=0):
    """Model-side confirmation: cross-entropy of the (quadratic-trained) model on
    raw tent vs conjugacy-mapped h(tent) vs logistic r=4.

    eval_context (30) is < the base context length (50) so tent orbits stay
    within the float precision budget (~52 steps); the model accepts shorter
    contexts. Expect raw tent >> h(tent) ~ logistic r=4.
    """
    import torch
    import torch.nn as nn
    crit = nn.CrossEntropyLoss()

    def ce(orbits):
        ctx, tgt = _windows_from_orbits(orbits, n_bins, eval_context)
        with torch.no_grad():
            c = torch.as_tensor(ctx, dtype=torch.long, device=device)
            t = torch.as_tensor(tgt, dtype=torch.long, device=device)
            return float(crit(model(c), t).item())

    tent = _tent_orbits(n_traj, traj_len, seed)
    tent_h = [conjugacy_h(o) for o in tent]
    log4 = _logistic_orbits([4.0], n_traj, traj_len, burn_in=50, seed=seed + 1)
    return {"raw_tent": ce(tent), "h_tent": ce(tent_h), "logistic_r4": ce(log4)}


def plot_occupancy(result, save_path=None):
    import matplotlib.pyplot as plt
    occ, N = result["occ"], result["n_bins"]
    centers = (np.arange(N) + 0.5) / N
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))

    a1.plot(centers, occ["tent"], color="#1D9E75", lw=1.6, label="tent (s=2), raw")
    a1.plot(centers, occ["log4"], color="#1B2A4A", lw=1.6, label="logistic r=4")
    a1.plot(centers, occ["logpool"], color="#999999", lw=1.2, ls="--", label="logistic pooled (all r)")
    a1.set_title("Raw occupancy: tent lives where logistic r=4 does not")
    a1.set_xlabel("$x$"); a1.set_ylabel("occupancy")
    a1.legend(fontsize=8)

    a2.plot(centers, occ["tent_h"], color="#1D9E75", lw=1.6, label="$h$(tent), conjugacy-mapped")
    a2.plot(centers, occ["log4"], color="#1B2A4A", lw=1.6, ls="--", label="logistic r=4")
    a2.set_title("After conjugacy $h$: occupancy matches logistic r=4")
    a2.set_xlabel("$x$"); a2.set_ylabel("occupancy")
    a2.legend(fontsize=8)

    fig.suptitle(
        f"Unseen order-1 transition mass: raw tent = {result['unseen_tent']:.1%}, "
        f"h(tent) = {result['unseen_tent_h']:.1%}", y=1.02, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    res = run_occupancy_diagnostic()
    print("Occupancy L1 distance to logistic r=4:")
    print(f"  raw tent    : {res['occ_L1_tent_vs_log4']:.3f}")
    print(f"  h(tent)     : {res['occ_L1_tenth_vs_log4']:.3f}  (should be ~0)")
    print("Fraction of order-1 transition mass UNSEEN in logistic training pool:")
    print(f"  raw tent    : {res['unseen_tent']:.1%}")
    print(f"  h(tent)     : {res['unseen_tent_h']:.1%}  (should be ~0)")
