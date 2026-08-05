"""
Generate a jobs.txt for a SLURM array sweep: one line = one train_subset.py
argument string. Line number N == SLURM_ARRAY_TASK_ID (see run_array.slurm).

Torch-free -- run on the login node.

Window mode (experiment 1: extrapolation / sliding window):
    tile [lo, hi] into non-overlapping windows of each --widths, sweep --ms/--seeds.
    python scripts/gen_jobs.py --mode window --widths 0.4 0.8 1.0 --ms 8

Strategy mode (experiment 2: informative-prompt placements over the full range):
    python scripts/gen_jobs.py --mode strategy \
        --placements uniform_r uniform_lambda chaotic bifurcation --ms 8 16
"""
import argparse

import numpy as np


def tile_starts(width, lo, hi):
    """Starts of ~non-overlapping windows of `width` covering [lo, hi]."""
    k = max(1, int(np.ceil((hi - lo) / width)))
    return list(np.linspace(lo, hi - width, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["window", "strategy"], default="window")
    ap.add_argument("--widths", type=float, nargs="+", default=[0.4, 0.8, 1.0])
    ap.add_argument("--ms", type=int, nargs="+", default=[8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--placements", nargs="+",
                    default=["uniform_r", "uniform_lambda", "chaotic", "bifurcation"])
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=4.0)
    ap.add_argument("--n_train_traj", type=int, default=8000,
                    help="strategy mode only: total training trajectories")
    ap.add_argument("--base_n_traj", type=int, default=8000,
                    help="window mode: base run size; r-density = base_n_traj/(hi-lo)")
    ap.add_argument("--traj_per_r", type=int, default=1,
                    help="window mode: trajectories per r (base run uses 1)")
    ap.add_argument("--context_len", type=int, default=50)
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--out_base", default="runs")
    ap.add_argument("--jobs_file", default="jobs.txt")
    a = ap.parse_args()

    common = (f"--n_train_traj {a.n_train_traj} --context_len {a.context_len} "
              f"--n_bins {a.n_bins} --full_lo {a.lo} --full_hi {a.hi}")
    lines = []

    if a.mode == "window":
        # density-matched to the base run: r-count = density * width, 1 traj/r,
        # so each window model == the base model restricted to [start, start+w].
        density = a.base_n_traj / (a.hi - a.lo)
        for w in a.widths:
            m = max(2, round(density * w))
            n_traj = a.traj_per_r * m
            win_common = (f"--n_train_traj {n_traj} --context_len {a.context_len} "
                          f"--n_bins {a.n_bins} --full_lo {a.lo} --full_hi {a.hi}")
            for s in tile_starts(w, a.lo, a.hi):
                for seed in a.seeds:
                    name = f"win_w{w:g}_s{s:.3f}_seed{seed}"
                    lines.append(
                        f"--placement uniform_r_random --start {s:.4f} --width {w:g} "
                        f"--m {m} --seed {seed} {win_common} --out_dir {a.out_base}/{name}")
    else:
        span = a.hi - a.lo
        for pl in a.placements:
            for m in a.ms:
                for seed in a.seeds:
                    name = f"strat_{pl}_m{m}_seed{seed}"
                    lines.append(
                        f"--placement {pl} --start {a.lo} --width {span:g} "
                        f"--m {m} --seed {seed} {common} --out_dir {a.out_base}/{name}")

    with open(a.jobs_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(lines)} jobs -> {a.jobs_file}")
    if a.mode == "window":
        for w in a.widths:
            print(f"  width {w:g}: {len(tile_starts(w, a.lo, a.hi))} windows "
                  f"x {len(a.ms)} m x {len(a.seeds)} seed")
    print(f"\nsubmit with:\n  mkdir -p logs {a.out_base}\n"
          f"  sbatch --array=1-{len(lines)} scripts/run_array.slurm")


if __name__ == "__main__":
    main()
