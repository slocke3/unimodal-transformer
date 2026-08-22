# Running the r-subset sweeps on della (SLURM)

Handoff notes for launching the training sweeps on Princeton's della cluster.
The GPU work runs here; the analysis is torch-free and runs anywhere (see the
end). If anything here is stale or della-specific, ping Simon/Claude — the two
lines most likely to need editing are flagged **ADJUST**.

## What this runs

Each job trains **one** transformer on a chosen subset of `r`-values from the
logistic family and evaluates its cross-entropy across the whole family. A SLURM
job array runs many such jobs in parallel — one per line of `jobs.txt`.

The actual model/data/training code is the existing `src/` package; the three
`scripts/` files are just a CLI wrapper + scheduling shell around it. Nothing in
`src/` needs to change.

## One-time setup

```bash
# on the della login node
git clone <repo-url> unimodal-transformer
cd unimodal-transformer

# create the env (needs torch + numpy + scipy; see requirements.txt)
module load anaconda3/2024.6        # ADJUST: check `module avail anaconda`
conda create -y -n unimodal python=3.11
conda activate unimodal
pip install -r requirements.txt
```

No data download is needed — trajectories are generated on the fly, so compute
nodes never need internet.

## Adjust the SLURM script for della

Open `scripts/run_array.slurm` and edit the environment block (**ADJUST**):

```bash
module load anaconda3/2024.6        # della's actual anaconda module
conda activate unimodal             # the env created above
```

You may also need a partition line depending on della's config, e.g.:
```bash
#SBATCH --partition=gpu             # add if GPU jobs require an explicit partition
```
Resource requests are already set: 1 GPU, 16 GB, 4 CPUs, 1 h wall-time. The model
is tiny (~0.8 M params), so 16 GB is generous; 1 h is comfortably above the
~45 min/model seen on a Colab T4 (della A100s should be faster).

## Test ONE job before the full array

Always confirm the env + timing with a single task first:

```bash
python scripts/gen_jobs.py --mode window --widths 0.4 --n_train_traj 8000
# writes jobs.txt
mkdir -p logs runs
sbatch --array=1-1 scripts/run_array.slurm                     # just task 1
squeue -u $USER                                                # watch it
cat logs/*_1.out                                               # its output
```

A finished job prints `[train_subset] DONE in <sec>s | mean CE in-window=…
complement=… hist-overlap=…` and writes performance, histogram, checkpoint,
configuration, and training-history artifacts under `runs/<name>/`.
Check the wall-time in `params.json` and, if it's near 1 h, bump `--time` in the
SLURM script before the big array.

## Launch the full sweep

```bash
# experiment 1 — sliding window (extrapolation):
python scripts/gen_jobs.py --mode window \
    --widths 1.0 0.5 0.25 0.125 --n_train_traj 8000
#   -> writes 53 jobs and prints the exact sbatch line

mkdir -p logs runs
sbatch --array=1-53%20 scripts/run_array.slurm      # %20 = at most 20 concurrent
```

`%20` throttles concurrency to be polite on a shared cluster — raise/lower to
taste. Use the job count that `gen_jobs.py` printed for `--array=1-<N>`.

For experiment 2 (informative-prompt placements over the full range):
```bash
python scripts/gen_jobs.py --mode strategy \
    --placements uniform_r uniform_lambda chaotic bifurcation --ms 8 16 32
sbatch --array=1-<N>%20 scripts/run_array.slurm
```

For prefix/suffix boundary sweeps (quarter-step defaults, 26 jobs including
separate full-range `rmax` and `rmin` runs):
```bash
python scripts/gen_jobs.py --mode boundary --out_base runs_hist_boundary
JOBS_FILE=jobs.txt sbatch --array=1-26%20 scripts/run_array.slurm
```

## Monitor

```bash
squeue -u $USER                 # queued/running tasks
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after they run (timing/mem)
tail -f logs/<jobid>_<task>.out # live output of one task
scancel <jobid>                 # cancel the whole array if needed
```

## Getting results back to Simon (the important bit)

Simon can't see della's filesystem, and the analysis is **torch-free** (just
loads `.npz` and plots), so the clean handoff is to send him the small eval
files. Each `runs/<name>/eval_per_r.npz` is a few hundred floats (~KB) — tiny,
unlike the `best.pt` checkpoints.

Easiest options:
- **commit the eval files** (not checkpoints): `git add -f runs/**/eval_per_r.npz runs/**/params.json && git commit && git push` — Simon pulls and runs the analysis; **or**
- drop the `runs/` folder (or just the `.npz`+`params.json`) in a shared Drive/Globus location.

Keep the `best.pt` checkpoints on della (they're bigger and only needed if we
re-evaluate models later).

## Note for future budget sweeps: two checkpoints, one run

The August 2026 budget sweep trained **two arms** per budget -- a fixed-step arm
(exactly `--max_steps` gradient steps, no early stopping, final model evaluated)
and an early-stopping arm -- which is 12 jobs for 6 budgets.

Next time, run only the fixed-step arm and have it save a **best-validation
checkpoint alongside the final one**. The final weights give the fixed-step
result and the best-val checkpoint gives the early-stopping result, so one run
yields both numbers. That halves the job count, which matters because queue wait
on della normally dominates the actual compute.

It is also the better comparison. Two separate arms differ in optimizer
trajectory as well as in checkpoint rule -- each has its own cosine schedule and
its own sequence of updates -- so a gap between them is not attributable to
checkpoint selection alone. Two checkpoints off a single run hold the whole
optimization path fixed and isolate exactly which point along it you report.

Implementation sketch: `Trainer._train_fixed_steps` already evaluates validation
loss at intervals during the fixed-step loop; track the running minimum there and
save a separate checkpoint tag when it improves, then have `train_subset.py`
evaluate both tags and write two sets of per-r results.

**Do not edit anything under `src/` while an array is still queued.** Pending
array tasks import the training code fresh when they start, so a mid-array edit
makes later tasks run different code from earlier ones and the sweep is no longer
internally consistent. Wait until every task has reached a terminal state.

## What each script is

| file | role |
|------|------|
| `scripts/train_subset.py` | trains ONE model on one r-subset, evals across r, writes `eval_per_r.npz`. `--help` for all args (torch-free). |
| `scripts/gen_jobs.py`     | expands window, strategy, or prefix/suffix boundary grids into `jobs.txt`; prints the submit command. |
| `scripts/run_array.slurm` | array driver: task N runs line N of `jobs.txt`. |

Key run knobs (defaults in parentheses): `--context_len (50)`, `--n_bins (64)`,
`--n_train_traj (8000 total per job, split over m)`, `--r_eval_points (300)`.
Total training data is held fixed via `--n_train_traj`, so varying `m` changes
*where* the training r's are, not *how much* data.

## eval_per_r.npz contents

Alongside `r_grid`, `ce_per_r`, `acc_per_r`, `in_window`, and `train_r`, each
file stores `x_bin_edges`, `train_token_hist`, `eval_token_hist`,
`eval_token_hist_per_r`, `hist_overlap_per_r`, and
`aggregate_hist_overlap`. Histogram counts represent model exposures: every
context occurrence plus every target occurrence is counted.
