#!/usr/bin/env python3
"""Measure how many runs to pack per GPU on this pod, and what a sweep costs.

Takes one real line from a jobs file, shortens it to --steps, and runs c
copies at once on GPU 0 for each c in --conc. From the "Step N/S ... Xs" lines
the trainer prints, it gets the steady-state step rate per run, then projects
the full-length run time and the sweep throughput.

    python scripts/runpod/bench_pod.py jobs_win_fixed.txt --line 1 --conc 1 4 8 12

Pick the concurrency with the highest "full runs / GPU-hour" whose per-run
time is still acceptable, and pass it to run_jobs.py --per_gpu.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEP_RE = re.compile(r"Step\s+(\d+)/\d+ .*\|\s+([\d.]+)s")


def flag(toks, name, default=None):
    vals = [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == name]
    return vals[-1] if vals else default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs_file")
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--extra", default="")
    ap.add_argument("--train_script",
                    default=os.environ.get("TRAIN_SCRIPT", "scripts/train_subset.py"))
    a = ap.parse_args()

    os.chdir(REPO)
    line = open(a.jobs_file).read().splitlines()[a.line - 1]
    toks = shlex.split(line)
    full_steps = int(flag(toks, "--max_steps", 0)) or None
    # Validate at the same step interval as the real run, so the measured rate
    # carries the same validation overhead.
    full_logs = int(flag(toks, "--log_points", 15))
    log_points = max(3, round(a.steps * full_logs / full_steps)) if full_steps else 8
    print(f"[bench] line {a.line} of {a.jobs_file}; full run = {full_steps} steps; "
          f"bench = {a.steps} steps; vCPUs = {os.cpu_count()}", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"])

    rows = []
    for c in a.conc:
        root = os.path.join("bench", f"c{c}")
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        procs = []
        t0 = time.time()
        for k in range(c):
            args = (f"{line} {a.extra} --max_steps {a.steps} --log_points {log_points} "
                    f"--seed {k} --out_dir {root}/run{k}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
                       PYTHONPATH=REPO)
            log = open(os.path.join(root, f"run{k}.log"), "w")
            procs.append((subprocess.Popen(
                [sys.executable, a.train_script, *shlex.split(args)],
                stdout=log, stderr=subprocess.STDOUT, env=env), log))
        for p, log in procs:
            p.wait()
            log.close()
        wall = time.time() - t0
        rates, fails = [], 0
        for k, (p, _) in enumerate(procs):
            pts = [(int(s), float(t)) for s, t in
                   STEP_RE.findall(open(os.path.join(root, f"run{k}.log")).read())]
            if p.returncode != 0 or len(pts) < 2:
                fails += 1
                continue
            (s0, t0_), (s1, t1) = pts[0], pts[-1]
            rates.append((s1 - s0) / (t1 - t0_))
        if not rates:
            print(f"[bench] c={c}: all runs failed; see {root}/run0.log", flush=True)
            continue
        rate = sorted(rates)[len(rates) // 2]                 # steps/s per run
        overhead = max(wall - a.steps / rate, 0.0)            # setup + eval
        full_h = ((full_steps or a.steps) / rate + overhead) / 3600
        rows.append((c, rate, c * rate, full_h, c / full_h, fails))
        print(f"[bench] c={c}: {rate:.0f} steps/s per run, {c * rate:.0f} aggregate, "
              f"overhead {overhead:.0f}s, failures {fails}", flush=True)

    print("\nconc | steps/s/run | aggregate | full run (h) | full runs per GPU-hour")
    for c, r, agg, fh, thr, fails in rows:
        print(f"{c:4d} | {r:11.0f} | {agg:9.0f} | {fh:12.2f} | {thr:6.2f}"
              + (f"   ({fails} failed)" if fails else ""))
    print("\nCost of an N-run sweep on this GPU type ~= N / (runs per GPU-hour) "
          "x hourly price.")


if __name__ == "__main__":
    main()
