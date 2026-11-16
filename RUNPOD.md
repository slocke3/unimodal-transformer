# Running sweeps on RunPod

The RunPod counterpart of `CLUSTER.md`. Same jobs files (`scripts/gen_jobs.py`),
same trainers; one pod runs several jobs per GPU instead of one SLURM task each.
Pods are created in the RunPod website under the PI's team (which bills the
team) and driven over SSH. API keys are not needed.

## One-time: SSH key

On the laptop, `pbcopy < ~/.ssh/id_ed25519.pub`, then in the RunPod console (team
selected) paste it under **Settings → SSH Public Keys**. Pods created afterwards
accept that key.

## Create a pod

**Pods → Deploy**, with the team selected:

- **GPU:** any one card that will actually deploy: RTX 4090 / 5090, L40S, A6000,
  A40, A100, H100. The model is ~0.8M parameters, so memory is irrelevant and
  `bench_pod.py` decides the cost per run. More vCPUs allow more runs per GPU.
- **Template:** RunPod PyTorch (2.4 or newer). Leave SSH / TCP port 22 exposed.
- **Disk:** container 30 GB, volume 50 GB (mounted at `/workspace`).
- **If it says "no instances available":** set GPU count to 1, remove any network
  volume (it pins the pod to one datacenter), clear the region and CUDA-version
  filters, and try Community Cloud as well as Secure Cloud.

When it is running, **Connect → "SSH over exposed TCP"** gives
`ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`. Put that in `~/.ssh/config`:

```
Host runpod
    HostName <IP>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
```

The port changes each time the pod restarts.

## Run a sweep

```bash
# laptop
python scripts/gen_jobs.py ...                 # writes a jobs file as for della
scripts/runpod/sync.sh push                    # code + jobs files -> /workspace

# pod  (ssh runpod; cd /workspace/unimodal-transformer)
bash scripts/runpod/setup_pod.sh               # deps + GPU check; rerun after restarts
python scripts/runpod/bench_pod.py jobs_X.txt --conc 1 4 8 12   # pick --per_gpu
tmux new -s sweep
python scripts/runpod/run_jobs.py jobs_X.txt --lines 1-1        # one test run
python scripts/runpod/run_jobs.py jobs_X.txt --per_gpu 8        # the rest
#   Ctrl-b d detaches; the sweep keeps running if you disconnect

# laptop
scripts/runpod/sync.sh status                  # progress + GPU utilisation
scripts/runpod/sync.sh pull runs_X             # npz/json/png back; add --ckpt for .pt
```

`run_jobs.py` skips any run whose `--out_dir` already has `params.json`, so
rerunning the same command after a crash or restart only does what is missing.
Other trainers: `TRAIN_SCRIPT=scripts/train_asym_band.py python scripts/runpod/run_jobs.py ...`.
Each launch copies `src/` and `scripts/` to `logs/<jobs-stem>/code/` and runs from
that copy, so pushing new code while a sweep runs doesn't change it. Rerunning
the command (e.g. to resume) takes a fresh copy of whatever code is there then.

`--device_data` (in `train_subset.py`) keeps the training tensors on the GPU and
gathers batches there instead of using a DataLoader. Same sampling scheme (a
fresh random permutation each pass), different random stream, much less
per-step overhead.

## Billing

A running pod bills by the hour whether or not anything is training. **Stop** it
when a sweep finishes: `/workspace` and checkpoints are kept, and only storage is
billed. **Terminate** it to delete everything. Pull results first.
