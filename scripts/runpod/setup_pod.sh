#!/bin/bash
# One-time setup on a fresh RunPod pod (PyTorch template). Run on the pod from
# the repo root after `scripts/runpod/sync.sh push` has copied the code:
#
#   cd /workspace/unimodal-transformer && bash scripts/runpod/setup_pod.sh
#
# /workspace is the persistent volume: it survives stopping the pod, while the
# container disk (including anything pip installs) is reset on restart, so
# rerun this script after a restart.
set -eo pipefail

if ! command -v rsync >/dev/null || ! command -v tmux >/dev/null; then
    apt-get update -qq && apt-get install -y -qq rsync tmux >/dev/null
fi

# torch comes with the template; install the rest without touching torch.
# The template's system Python is "externally managed" (PEP 668); the pod is a
# throwaway container, so overriding that is safe here.
python -m pip install -q --break-system-packages \
    "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7" "pyyaml>=6.0"

python - <<'EOF'
import os, torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} "
      f"| GPUs: {torch.cuda.device_count()} | vCPUs: {os.cpu_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GiB")
x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
print("GPU matmul OK")
EOF
df -h /workspace | tail -1
