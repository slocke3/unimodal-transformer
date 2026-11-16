#!/bin/bash
# Move code to a RunPod pod and results back. Run from your laptop.
#
#   scripts/runpod/sync.sh push                 # code + jobs files -> pod
#   scripts/runpod/sync.sh pull runs_win_rep    # results of one sweep -> laptop
#   scripts/runpod/sync.sh pull runs_win_rep --ckpt   # ...including checkpoints
#   scripts/runpod/sync.sh status               # progress of every sweep on the pod
#   scripts/runpod/sync.sh ssh                  # shell on the pod
#
# The pod is the SSH host alias in $RUNPOD_HOST (default "runpod"), set up in
# ~/.ssh/config from the pod's "SSH over exposed TCP" command, e.g.
#
#   Host runpod
#       HostName 194.68.245.12
#       Port 22045
#       User root
#       IdentityFile ~/.ssh/id_ed25519
#
# The port changes whenever the pod is restarted; update it there.
set -eo pipefail

HOST="${RUNPOD_HOST:-runpod}"
REMOTE="${RUNPOD_DIR:-/workspace/unimodal-transformer}"
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

case "$1" in
  push)
    # Code, configs and jobs files only. Results, figures, the paper and
    # notebooks stay local; run directories on the pod are never overwritten.
    rsync -az --stats \
      --exclude '.git/' --exclude 'runs*/' --exclude 'figures*/' \
      --exclude 'paper*/' --exclude 'outputs/' --exclude 'notebooks/' \
      --exclude 'experiments/' --exclude 'logs/' --exclude 'bench/' \
      --exclude '__pycache__/' --exclude '*.zip' --exclude '.claude/' \
      ./ "$HOST:$REMOTE/"
    ;;
  pull)
    [ -n "$2" ] || { echo "usage: $0 pull <runs_dir> [--ckpt]"; exit 1; }
    filt=(--include '*/' --include '*.npz' --include '*.json' --include '*.png')
    [ "$3" = "--ckpt" ] && filt+=(--include '*.pt')
    rsync -az --stats --prune-empty-dirs "${filt[@]}" --exclude '*' \
      "$HOST:$REMOTE/$2/" "./$2/"
    rsync -az "$HOST:$REMOTE/logs/" ./logs/ 2>/dev/null || true
    ;;
  status)
    ssh "$HOST" "cd $REMOTE && cat logs/*/progress.txt 2>/dev/null; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    ;;
  ssh)
    exec ssh "$HOST"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 1
    ;;
esac
