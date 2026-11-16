#!/usr/bin/env python3
"""Run a jobs file on one RunPod machine, packing several runs onto each GPU.

The RunPod counterpart of scripts/run_array.slurm: line N of the jobs file is
the argument list for one training run. The model is ~0.8M parameters, so one
run leaves most of a GPU idle; --per_gpu runs several at once on each card.

A run counts as finished when its --out_dir holds params.json, the last file
train_subset.py writes. Finished runs are skipped, so rerunning the same
command after a pod restart (or a failure) picks up only what is missing.

    python scripts/runpod/run_jobs.py jobs_win_fixed.txt --lines 1-1      # test one
    python scripts/runpod/run_jobs.py jobs_win_fixed.txt --per_gpu 6      # full sweep
    TRAIN_SCRIPT=scripts/train_asym_band.py python scripts/runpod/run_jobs.py jobs_asym.txt

Per-run logs go to logs/<jobs-file-stem>/<line>.out and a one-line progress
summary to logs/<jobs-file-stem>/progress.txt. Each launch runs from a frozen copy
of src/ and scripts/ (logs/<stem>/code/), so pushing new code mid-sweep is safe.
"""
import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_lines(spec, n):
    if spec is None:
        return list(range(1, n + 1))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return [i for i in out if 1 <= i <= n]


def out_dir_of(args):
    toks = shlex.split(args)
    # argparse keeps the last occurrence, so do the same here.
    dirs = [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "--out_dir"]
    return dirs[-1] if dirs else None


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, check=True).stdout
        return [str(i) for i, l in enumerate(out.splitlines()) if l.startswith("GPU")]
    except (OSError, subprocess.CalledProcessError):
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs_file")
    ap.add_argument("--per_gpu", type=int, default=4,
                    help="concurrent runs per GPU (set from bench_pod.py)")
    ap.add_argument("--gpus", default=None,
                    help="comma list of GPU indices; default: all visible")
    ap.add_argument("--lines", default=None,
                    help="1-based line selection, e.g. 1-1 or 1-20,35")
    ap.add_argument("--extra", default="",
                    help="arguments appended to every line (later flags win), "
                         "e.g. '--num_workers 1'")
    ap.add_argument("--threads", type=int, default=1,
                    help="OMP/MKL threads per run, to avoid CPU oversubscription")
    ap.add_argument("--train_script",
                    default=os.environ.get("TRAIN_SCRIPT", "scripts/train_subset.py"))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    os.chdir(REPO)
    with open(a.jobs_file) as f:
        raw = [l.strip() for l in f]
    sel = parse_lines(a.lines, len(raw))
    todo, skipped = [], 0
    for i in sel:
        line = raw[i - 1]
        if not line or line.startswith("#"):
            continue
        args = f"{line} {a.extra}".strip()
        od = out_dir_of(args)
        if od is None:
            sys.exit(f"line {i} has no --out_dir: {line[:80]}")
        if os.path.exists(os.path.join(od, "params.json")):
            skipped += 1
            continue
        todo.append((i, args, od))

    gpus = a.gpus.split(",") if a.gpus else detect_gpus()
    if not gpus:
        sys.exit("no GPUs visible (nvidia-smi -L found none); pass --gpus to override")
    stem = os.path.splitext(os.path.basename(a.jobs_file))[0]
    log_dir = os.path.join("logs", stem)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[run_jobs] {a.jobs_file}: {len(todo)} to run, {skipped} already done | "
          f"GPUs {','.join(gpus)} x {a.per_gpu} | trainer {a.train_script}", flush=True)
    if a.dry_run:
        for i, args, od in todo:
            print(f"  line {i}: {od}")
        return
    if not todo:
        return

    # Freeze the code this sweep runs: runs import src/ when they START, so a
    # later `sync.sh push` would otherwise change the code mid-sweep. Each
    # launch copies src/ and scripts/ to logs/<stem>/code/ and runs from there;
    # output paths stay relative to the repo root.
    code = os.path.join(log_dir, "code")
    shutil.rmtree(code, ignore_errors=True)
    for d in ("src", "scripts"):
        shutil.copytree(d, os.path.join(code, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    code = os.path.abspath(code)
    train_script = os.path.join(code, a.train_script)

    running = {}          # Popen -> (line, gpu, t_start, log handle)
    load = {g: 0 for g in gpus}
    done, failed = 0, []
    t0 = time.time()

    def stop(signum, frame):
        print(f"\n[run_jobs] signal {signum}: terminating {len(running)} runs", flush=True)
        for p in running:
            p.terminate()
        for p in running:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(1)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def report():
        msg = (f"[{time.strftime('%H:%M:%S')}] {stem}: done {done}/{len(todo)} "
               f"failed {len(failed)} running {len(running)} "
               f"pending {len(todo) - done - len(failed) - len(running)} "
               f"elapsed {(time.time() - t0) / 60:.1f} min")
        print(msg, flush=True)
        with open(os.path.join(log_dir, "progress.txt"), "w") as f:
            f.write(msg + "\n" + "".join(f"FAILED line {l}: {p}\n" for l, p in failed))

    queue = list(todo)
    while queue or running:
        while queue:
            g = min(gpus, key=lambda x: load[x])
            if load[g] >= a.per_gpu:
                break
            i, args, od = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
                       PYTHONPATH=code + os.pathsep + os.environ.get("PYTHONPATH", ""))
            log_path = os.path.join(log_dir, f"{i}.out")
            lf = open(log_path, "w")
            lf.write(f"=== line {i} on GPU {g} at {time.ctime()} ===\nargs: {args}\n")
            lf.flush()
            p = subprocess.Popen([sys.executable, train_script, *shlex.split(args)],
                                 stdout=lf, stderr=subprocess.STDOUT, env=env)
            running[p] = (i, g, time.time(), lf, log_path)
            load[g] += 1
        time.sleep(5)
        changed = False
        for p in [p for p in running if p.poll() is not None]:
            i, g, ts, lf, log_path = running.pop(p)
            lf.close()
            load[g] -= 1
            changed = True
            if p.returncode == 0:
                done += 1
            else:
                failed.append((i, log_path))
                print(f"[run_jobs] line {i} FAILED (exit {p.returncode}) -> {log_path}",
                      flush=True)
        if changed:
            report()

    report()
    if failed:
        print("[run_jobs] rerun the same command to retry failed lines", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
