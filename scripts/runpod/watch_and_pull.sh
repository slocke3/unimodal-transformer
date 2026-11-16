#!/bin/bash
# Wait for a sweep on the pod to finish, then pull its results (with checkpoints)
# and check the file counts match. Run from the laptop, kept awake:
#
#   caffeinate -i scripts/runpod/watch_and_pull.sh jobs_rb_ctxshort runs_ctx_short [--stop]
#
# Polls the pod every 10 minutes. With --stop, and only if every file arrived,
# it then stops the pod (runpodctl on the pod, using the key RunPod injects into
# the container). Stopping ends GPU billing; the pod's disk is not kept.
set -o pipefail
JOBS="$1"; RUNS="$2"; STOP="$3"
[ -n "$JOBS" ] && [ -n "$RUNS" ] || { echo "usage: $0 <jobs_stem> <runs_dir>"; exit 1; }
HOST="${RUNPOD_HOST:-runpod}"
REMOTE="${RUNPOD_DIR:-/workspace/unimodal-transformer}"
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

while true; do
  line=$(ssh -o ConnectTimeout=20 "$HOST" "head -1 $REMOTE/logs/$JOBS/progress.txt" 2>/dev/null)
  echo "$(date '+%H:%M') $line"
  case "$line" in *"running 0 pending 0"*) break ;; esac
  sleep 600
done

scripts/runpod/sync.sh pull "$RUNS" --ckpt | tail -2
n_local=$(find "$RUNS" -type f \( -name '*.npz' -o -name '*.json' -o -name '*.png' -o -name '*.pt' \) | wc -l | tr -d ' ')
n_remote=$(ssh "$HOST" "cd $REMOTE && find $RUNS -type f \( -name '*.npz' -o -name '*.json' -o -name '*.png' -o -name '*.pt' \) | wc -l" | tr -d ' ')
echo "files: local $n_local, pod $n_remote"
[ "$n_local" = "$n_remote" ] && echo "PULL COMPLETE" || { echo "PULL MISMATCH"; exit 2; }

if [ "$STOP" = "--stop" ]; then
  ssh "$HOST" 'export $(tr "\0" "\n" < /proc/1/environ | grep "^RUNPOD_API_KEY=") && runpodctl stop pod "$(tr "\0" "\n" < /proc/1/environ | sed -n "s/^RUNPOD_POD_ID=//p")"' 2>&1 \
    | grep -v -i "config file"
  echo "stop requested at $(date '+%H:%M')"
fi
