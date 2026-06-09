"""
Validate the operator-floor harness against the doubling map (x -> 2x mod 1),
where the answer is known analytically:

  * dyadic partition (n_bins = 2^m) -> exact Markov, uniform invariant measure
  * order-1 entropy rate = ln 2 = lambda
  * held-out k-gram cross-entropy -> ln 2
  * for n_bins = 2: symbols are i.i.d. fair coin, so the Ulam matrix is
    [[.5, .5], [.5, .5]] with spectrum {1, 0} (spectral gap = 1)

Also illustrates (no assertions) that under a UNIFORM (non-natural) partition the
floor CE decreases with order toward the true entropy rate -- and the *shape* of
that decrease is a signature of the dynamics: for r=4 (chaotic, conjugate to
doubling) it decays slowly toward ln2; for a periodic window (r=3.5, period 4) it
collapses sharply once order >= the period. Under the natural/dyadic partition
(the doubling map above) it would already be Markov at order 1 -- which is exactly
why the kneading-natural tokenization matters.

Run from the repo root:  python scripts/validate_baselines.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.maps import (
    doubling_map, doubling_deriv, iterate_general, iterate_map,
    tokenize_trajectory, compute_lyapunov_general,
)
from src.baselines import KGramModel, make_context_target_pairs, operator_floor_curve

LN2 = np.log(2.0)


def doubling_tokens(n_traj, traj_len, n_bins, seed):
    """Short doubling-map trajectories (float64 collapses after ~52 steps)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_traj):
        x0 = rng.uniform(1e-6, 1 - 1e-6)
        traj = iterate_general(x0, lambda x, _p: doubling_map(x), None, traj_len)
        seqs.append(tokenize_trajectory(traj, n_bins))
    return seqs


def logistic_tokens(n_traj, traj_len, n_bins, r, seed):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_traj):
        x0 = rng.uniform(0.05, 0.95)
        traj = iterate_map(x0, r, traj_len)
        seqs.append(tokenize_trajectory(traj, n_bins))
    return seqs


def check(name, value, expected, tol):
    ok = abs(value - expected) < tol
    flag = "OK " if ok else "XX "
    print(f"  [{flag}] {name}: {value:.5f}  (expect {expected:.5f}, tol {tol})")
    return ok


def main():
    print(f"lambda(doubling) = "
          f"{compute_lyapunov_general(lambda x, _p: doubling_map(x), lambda x, _p: doubling_deriv(x), None):.5f}"
          f"   (ln2 = {LN2:.5f})\n")

    all_ok = True

    # --- N = 2: i.i.d. fair coin, exact control --------------------------
    print("Doubling map, n_bins=2, order=1 (i.i.d. fair coin):")
    train = doubling_tokens(n_traj=4000, traj_len=25, n_bins=2, seed=0)
    test = doubling_tokens(n_traj=1000, traj_len=25, n_bins=2, seed=1)
    m = KGramModel(n_bins=2, order=1, alpha=0.0).fit(train)
    T = m.transition_matrix()
    spec = m.spectrum()
    pi = m.stationary_distribution()
    ctx, tgt = make_context_target_pairs(test, context_len=1)
    ce = m.cross_entropy(ctx, tgt)

    print("  transition matrix:", np.round(T, 4).tolist())
    print("  spectrum:", np.round(spec, 4).tolist())
    print("  stationary:", np.round(pi, 4).tolist())
    all_ok &= check("entropy rate", m.entropy_rate(), LN2, 0.02)
    all_ok &= check("held-out CE ", ce, LN2, 0.02)
    all_ok &= check("|2nd eigenvalue|", abs(spec[1]), 0.0, 0.05)
    all_ok &= check("stationary[0]", pi[0], 0.5, 0.02)

    # --- N = 4: dyadic Markov, still h = ln2 -----------------------------
    print("\nDoubling map, n_bins=4, order=1 (dyadic Markov):")
    train4 = doubling_tokens(n_traj=8000, traj_len=25, n_bins=4, seed=2)
    test4 = doubling_tokens(n_traj=2000, traj_len=25, n_bins=4, seed=3)
    m4 = KGramModel(n_bins=4, order=1, alpha=0.0).fit(train4)
    ctx4, tgt4 = make_context_target_pairs(test4, context_len=1)
    print("  stationary:", np.round(m4.stationary_distribution(), 4).tolist())
    all_ok &= check("entropy rate", m4.entropy_rate(), LN2, 0.03)
    all_ok &= check("held-out CE ", m4.cross_entropy(ctx4, tgt4), LN2, 0.03)

    # --- Illustration: when does memory (order) help? (no assertions) -----
    print("\nIllustration -- floor CE vs order shows where memory matters:")

    print("  r=4 (fully chaotic) under UNIFORM n_bins=8 (non-natural partition):")
    print(f"  CE decays slowly with order toward ln2 = {LN2:.3f} -- uniform bins")
    print("  are not Markov, so memory keeps helping (=> kneading partition matters).")
    ce4 = operator_floor_curve(
        logistic_tokens(2000, 150, n_bins=8, r=4.0, seed=10),
        *make_context_target_pairs(logistic_tokens(500, 150, n_bins=8, r=4.0, seed=11), context_len=4),
        n_bins=8, orders=[1, 2, 4], alpha=1e-3)
    for k in (1, 2, 4):
        print(f"      order {k}: CE = {ce4[k]['ce']:.4f}  acc = {ce4[k]['acc']:.3f}")

    print("  r=3.5 (period-4 window): CE drops sharply once order >= period --")
    print("  memory is essential and the floor reflects it.")
    ce35 = operator_floor_curve(
        logistic_tokens(2000, 120, n_bins=8, r=3.5, seed=12),
        *make_context_target_pairs(logistic_tokens(500, 120, n_bins=8, r=3.5, seed=13), context_len=5),
        n_bins=8, orders=[1, 2, 4], alpha=1e-3)
    for k in (1, 2, 4):
        print(f"      order {k}: CE = {ce35[k]['ce']:.4f}  acc = {ce35[k]['acc']:.3f}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
