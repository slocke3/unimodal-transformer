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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["window", "strategy", "boundary"],
                    default="window")
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
    print(f"\nsubmit with:\n  mkdir -p logs {a.out_base}\n"
          f"  sbatch --array=1-{len(lines)} scripts/run_array.slurm")


if __name__ == "__main__":
    main()
