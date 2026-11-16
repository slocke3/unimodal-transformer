"""Write the jobs files for the rebuttal sweeps (run with scripts/runpod/run_jobs.py).

  ctxshort   FWzs: contexts L in {1, 2, 5}, otherwise identical to the context
             ladder in runs_ctx (scalar output, 64 input bins, 32k trajectories,
             trajectory length L+100, 320k steps), same 11 task counts, 3 seeds.
             Plus three L=10 runs that repeat existing runs_ctx configurations
             under the same loader as the new runs, as a check that the new
             points can be read against the old ones.
  capacity   68V6: the task-diversity sweep (cross-entropy, 64 output bins) at
             two input resolutions whose crossovers differ (32 and 128 bins),
             with a smaller (64 wide, 2 layers, 3 seeds) and a larger (256
             wide, 4 layers, 2 seeds) transformer than the 128-wide, 4-layer
             model used throughout. Steps as in taskdiv_gap: 160k for <=32
             bins, 320k above.
  henon      Scope (all reviewers): the task-diversity sweep on the Henon family
             x' = 1 - a x^2 + y, y' = 0.3 x, observing only x (src/maps.py),
             tasks a in [0, 1.4]. --stage pilot: 4 task counts x {32, 128} input
             bins x 1 seed, cross-entropy. --stage full: 7 task counts x
             {16, 32, 64, 128, 256} x 3 seeds, both objectives.

  probe      Checkpoints for the probes, retrained on RunPod because the
             originals are on della: the 64-bin task-diversity models (both
             objectives) at m = 100, 1000, 8000, and the [3, 3.25] window models
             under both objectives (exact copies of jobs_win_fixed lines 43-44).

  python scripts/rebuttal/make_jobs.py ctxshort      # -> jobs_rb_ctxshort.txt
  python scripts/rebuttal/make_jobs.py capacity      # -> jobs_rb_capacity.txt

--device_data (GPU-resident batches, src/device_loader.py) is on by default;
pass --no_device_data for the original DataLoader.
"""
import argparse

CTX_MS = [100, 150, 250, 350, 500, 700, 1000, 1400, 2000, 4000, 8000]
TASKDIV_MS = [100, 250, 500, 1000, 2000, 4000, 8000]
# (architecture, seeds). Base model: d_model 128, 4 heads, 4 layers (~0.8M).
SIZES = {"small": ("--d_model 64 --n_heads 4 --n_layers 2", (0, 1, 2)),      # ~0.1M
         "large": ("--d_model 256 --n_heads 8 --n_layers 4", (0, 1))}        # ~3.2M
FAMILY = ("--placement uniform_r_random --start 0.5 --width 3.5 --n_train_traj 32000 "
          "--n_bins 64 --n_bins_out 64 --full_lo 0.5 --full_hi 4.0 --max_val_traj 600")


def ctxshort(extra):
    lines = []
    for L in (1, 2, 5):
        for m in CTX_MS:
            for s in (0, 1, 2):
                lines.append(f"--m {m} --n_bins_in 64 --context_len {L} --traj_len {L + 100} "
                             f"{FAMILY} --max_steps 320000 --log_points 160 "
                             f"--output_mode scalar --seed {s}{extra} "
                             f"--out_dir runs_ctx_short/L{L}_m{m}_seed{s}")
    for m in (250, 1000, 4000):
        lines.append(f"--m {m} --n_bins_in 64 --context_len 10 --traj_len 110 "
                     f"{FAMILY} --max_steps 320000 --log_points 160 "
                     f"--output_mode scalar --seed 0{extra} "
                     f"--out_dir runs_ctx_short/L10check_m{m}_seed0")
    return lines


def capacity(extra):
    lines = []
    for size, (arch, seeds) in SIZES.items():
        for n in (32, 128):
            steps, logs = (160000, 80) if n <= 32 else (320000, 160)
            for m in TASKDIV_MS:
                for s in seeds:
                    lines.append(f"--m {m} --n_bins_in {n} --context_len 50 {FAMILY} "
                                 f"--max_steps {steps} --log_points {logs} {arch} "
                                 f"--seed {s}{extra} "
                                 f"--out_dir runs_capacity/{size}_in{n}_bins_m{m}_seed{s}")
    return lines


def henon(extra, stage):
    hfam = FAMILY.replace("--start 0.5 --width 3.5", "--start 0.0 --width 1.4").replace(
        "--full_lo 0.5 --full_hi 4.0", "--full_lo 0.0 --full_hi 1.4")
    if stage == "pilot":
        ms, ns, seeds, outs = [100, 500, 2000, 8000], [32, 128], [0], ["bins"]
    else:
        ms, ns, seeds, outs = TASKDIV_MS, [16, 32, 64, 128, 256], [0, 1, 2], ["bins", "mse"]
    lines = []
    for o in outs:
        mode = "" if o == "bins" else " --output_mode scalar"
        for n in ns:
            steps, logs = (160000, 80) if n <= 32 else (320000, 160)
            for m in ms:
                for s in seeds:
                    lines.append(f"--family henon --m {m} --n_bins_in {n} --context_len 50 "
                                 f"{hfam} --max_steps {steps} --log_points {logs}{mode} "
                                 f"--seed {s}{extra} "
                                 f"--out_dir runs_henon/in{n}_{o}_m{m}_seed{s}")
    return lines


def probe(extra):
    lines = []
    for o, mode in (("bins", ""), ("mse", " --output_mode scalar")):
        for m in (100, 1000, 8000):
            lines.append(f"--m {m} --n_bins_in 64 --context_len 50 {FAMILY} "
                         f"--max_steps 320000 --log_points 160{mode} --seed 0{extra} "
                         f"--out_dir runs_probe/in64_{o}_m{m}_seed0")
    win = ("--placement uniform_r_random --start 3.0000 --width 0.25 --m 571 --seed 0 "
           "--n_train_traj 8000 --context_len 50 --n_bins 64 --full_lo 0.5 --full_hi 4.0 "
           "--max_steps 160000 --log_points 80 --max_val_traj 600")
    lines.append(f"{win}{extra} --out_dir runs_probe/win_w0.25_s3.000_ce_seed0")
    lines.append(f"{win} --output_mode scalar{extra} --out_dir runs_probe/win_w0.25_s3.000_mse_seed0")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=["ctxshort", "capacity", "henon", "probe"])
    ap.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--no_device_data", action="store_true")
    a = ap.parse_args()
    extra = "" if a.no_device_data else " --device_data --num_workers 0"
    if a.sweep == "henon":
        lines, out = henon(extra, a.stage), f"jobs_rb_henon_{a.stage}.txt"
    else:
        lines = {"ctxshort": ctxshort, "capacity": capacity, "probe": probe}[a.sweep](extra)
        out = f"jobs_rb_{a.sweep}.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
