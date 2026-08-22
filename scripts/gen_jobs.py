"""
Generate a jobs.txt for a SLURM array sweep: one line = one train_subset.py
argument string. Line number N == SLURM_ARRAY_TASK_ID (see run_array.slurm).

Torch-free -- run on the login node.

Window mode (experiment 1: extrapolation / sliding window):
    tile [lo, hi] into non-overlapping windows of each --widths.  Each window
    receives --n_train_traj total trajectories, distributed over an r-grid whose
    density is set by --base_n_traj.
    python scripts/gen_jobs.py --mode window --widths 0.125 0.25 0.5 1.0 \
        --n_train_traj 8000

Strategy mode (experiment 2: informative-prompt placements over the full range):
    python scripts/gen_jobs.py --mode strategy \
        --placements uniform_r uniform_lambda chaotic bifurcation --ms 8 16

Budget mode (experiment 3: how much data to learn the family?):
    full-range uniform-random r, one trajectory per r (m == N == n_train_traj).
    Two arms per budget, so the two training protocols can be compared:
      fixed  -- exactly --max_steps gradient steps, no early stopping, FINAL
                model evaluated. Equal optimization budget at every N, so small
                budgets are allowed to overfit (that is the measurement).
      early  -- epoch mode + early stopping on val, BEST checkpoint evaluated.
                Its epoch ceiling is set per budget so it may spend at most the
                same --max_steps, leaving the checkpoint rule as the only
                difference between the arms.
    python scripts/gen_jobs.py --mode budget --arms both --max_steps 30000 \
        --out_base runs_budget

Boundary mode (prefix/suffix generalization):
    python scripts/gen_jobs.py --mode boundary \
        --r_maxes 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 \
        --r_mins 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5
"""
import argparse

import numpy as np


def tile_starts(width, lo, hi):
    """Starts of ~non-overlapping windows of `width` covering [lo, hi]."""
    k = max(1, int(np.ceil((hi - lo) / width)))
    return list(np.linspace(lo, hi - width, k))


def steps_per_epoch(n_traj, traj_len, context_len, val_frac, batch_size):
    """Optimizer steps in one epoch for a budget of `n_traj` trajectories.

    Mirrors train_subset.py: the val split is taken by trajectory, and each
    remaining trajectory yields (traj_len - context_len) sliding windows.
    """
    n_train = n_traj - int(val_frac * n_traj)
    n_examples = n_train * (traj_len - context_len)
    return max(1, -(-n_examples // batch_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["window", "strategy", "boundary", "budget"],
                    default="window")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[4000, 2000, 1000, 500, 250, 100],
                    help="budget mode: data budgets N (uniform-random full-range, "
                         "1 traj/r)")
    ap.add_argument("--arms", choices=["fixed", "early", "both"], default="both",
                    help="budget mode: fixed-step arm, early-stopping arm, or both")
    ap.add_argument("--max_steps", type=int, default=30000,
                    help="budget mode: gradient-step budget. Trained exactly in the "
                         "fixed arm; an epoch ceiling in the early arm.")
    ap.add_argument("--log_points", type=int, default=60,
                    help="budget mode: train/val samples recorded per fixed-step run")
    ap.add_argument("--traj_len", type=int, default=150,
                    help="used to predict steps/epoch when capping the early arm")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="used to predict steps/epoch when capping the early arm")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="used to predict steps/epoch when capping the early arm")
    ap.add_argument("--widths", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--r_maxes", type=float, nargs="+",
                    default=list(np.arange(1.0, 4.01, 0.25)))
    ap.add_argument("--r_mins", type=float, nargs="+",
                    default=list(np.arange(0.5, 3.51, 0.25)))
    ap.add_argument("--ms", type=int, nargs="+", default=[8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--placements", nargs="+",
                    default=["uniform_r", "uniform_lambda", "chaotic", "bifurcation"])
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=4.0)
    ap.add_argument("--n_train_traj", type=int, default=8000,
                    help="total training trajectories per job")
    ap.add_argument("--base_n_traj", type=int, default=8000,
                    help="window mode: r-density = base_n_traj/(hi-lo)")
    ap.add_argument("--context_len", type=int, default=50)
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--out_base", default="runs")
    ap.add_argument("--jobs_file", default="jobs.txt")
    a = ap.parse_args()

    common = (f"--n_train_traj {a.n_train_traj} --context_len {a.context_len} "
              f"--n_bins {a.n_bins} --full_lo {a.lo} --full_hi {a.hi}")
    lines = []

    if a.mode == "window":
        # Preserve a common r-value density while holding each window's total
        # trajectory budget fixed at --n_train_traj.
        density = a.base_n_traj / (a.hi - a.lo)
        for w in a.widths:
            m = max(2, round(density * w))
            win_common = (f"--n_train_traj {a.n_train_traj} --context_len {a.context_len} "
                          f"--n_bins {a.n_bins} --full_lo {a.lo} --full_hi {a.hi}")
            for s in tile_starts(w, a.lo, a.hi):
                for seed in a.seeds:
                    name = f"win_w{w:g}_s{s:.3f}_seed{seed}"
                    lines.append(
                        f"--placement uniform_r_random --start {s:.4f} --width {w:g} "
                        f"--m {m} --seed {seed} {win_common} --out_dir {a.out_base}/{name}")
    elif a.mode == "budget":
        # data-budget sweep: full-range uniform-random r, one trajectory per r
        # (m == N == n_train_traj). Every job gets the SAME gradient-step budget
        # so that "how much data to learn the family" is not confounded with how
        # much optimization each budget received.
        span = a.hi - a.lo
        arms = ["fixed", "early"] if a.arms == "both" else [a.arms]
        for N in a.budgets:
            spe = steps_per_epoch(N, a.traj_len, a.context_len,
                                  a.val_frac, a.batch_size)
            # Early arm: cap epochs at the same step budget, and scale patience
            # so "no improvement" always means the same number of steps (~10% of
            # the budget) rather than the same number of tiny epochs.
            max_epochs = max(1, -(-a.max_steps // spe))
            patience = min(max_epochs, max(8, -(-(a.max_steps // 10) // spe)))
            for arm in arms:
                if arm == "fixed":
                    protocol = (f"--max_steps {a.max_steps} "
                                f"--log_points {a.log_points}")
                else:
                    protocol = f"--max_epochs {max_epochs} --patience {patience}"
                for seed in a.seeds:
                    name = f"budget_N{N}_{arm}_seed{seed}"
                    lines.append(
                        f"--placement uniform_r_random --start {a.lo} --width {span:g} "
                        f"--m {N} --n_train_traj {N} --context_len {a.context_len} "
                        f"--n_bins {a.n_bins} --traj_len {a.traj_len} "
                        f"--batch_size {a.batch_size} --val_frac {a.val_frac} "
                        f"--full_lo {a.lo} --full_hi {a.hi} {protocol} "
                        f"--seed {seed} --out_dir {a.out_base}/{name}")
    elif a.mode == "strategy":
        span = a.hi - a.lo
        for pl in a.placements:
            for m in a.ms:
                for seed in a.seeds:
                    name = f"strat_{pl}_m{m}_seed{seed}"
                    lines.append(
                        f"--placement {pl} --start {a.lo} --width {span:g} "
                        f"--m {m} --seed {seed} {common} --out_dir {a.out_base}/{name}")
    else:
        density = a.base_n_traj / (a.hi - a.lo)
        intervals = (
            [("rmax", a.lo, boundary) for boundary in a.r_maxes]
            + [("rmin", boundary, a.hi) for boundary in a.r_mins]
        )
        for kind, start, end in intervals:
            if not (a.lo <= start < end <= a.hi):
                raise ValueError(
                    f"invalid {kind} interval [{start}, {end}] "
                    f"for full range [{a.lo}, {a.hi}]"
                )
            width = end - start
            m = max(2, round(density * width))
            for seed in a.seeds:
                boundary = end if kind == "rmax" else start
                name = f"{kind}_{boundary:.3f}_seed{seed}"
                lines.append(
                    f"--placement uniform_r_random --start {start:g} --width {width:g} "
                    f"--m {m} --seed {seed} {common} --out_dir {a.out_base}/{name}")

    with open(a.jobs_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(lines)} jobs -> {a.jobs_file}")
    if a.mode == "window":
        for w in a.widths:
            print(f"  width {w:g}: {len(tile_starts(w, a.lo, a.hi))} windows "
                  f"(m={max(2, round(a.base_n_traj / (a.hi - a.lo) * w))}) "
                  f"x {len(a.seeds)} seed")
    elif a.mode == "boundary":
        print(f"  r_max prefixes: {len(a.r_maxes)} x {len(a.seeds)} seed")
        print(f"  r_min suffixes: {len(a.r_mins)} x {len(a.seeds)} seed")
    elif a.mode == "budget":
        print(f"  budgets N={a.budgets} (1 traj/r) x arms={a.arms} "
              f"x {len(a.seeds)} seed | step budget {a.max_steps}")
        for N in a.budgets:
            spe = steps_per_epoch(N, a.traj_len, a.context_len,
                                  a.val_frac, a.batch_size)
            print(f"    N={N:>5}: {spe:>5} steps/epoch -> fixed arm sees "
                  f"{a.max_steps / spe:6.1f} epochs; early arm capped there")
    print(f"\nsubmit with:\n  mkdir -p logs {a.out_base}\n"
          f"  sbatch --array=1-{len(lines)} scripts/run_array.slurm")


if __name__ == "__main__":
    main()
