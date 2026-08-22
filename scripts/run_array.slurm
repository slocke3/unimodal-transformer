#!/bin/bash
# SLURM array driver for the r-subset sweep (Princeton della).
#
# Each array task reads its own line from jobs.txt (line N == task ID) and trains
# one model. Generate jobs.txt first with scripts/gen_jobs.py, then:
#
#   mkdir -p logs runs
#   sbatch --array=1-<NJOBS>%20 scripts/run_array.slurm
#
# (%20 caps concurrent tasks at 20 -- adjust to taste / priority.)
#
#SBATCH --job-name=unimodal-sweep
#SBATCH --output=logs/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# Keep this just OVER 1 hour: <=1h is routed to della's test QOS (long waits),
# while a padded request backfills poorly behind the 12-24h jobs that dominate
# the queue. Observed task runtimes are 5-20 min, so 1:05 is ~3-12x headroom.
#SBATCH --time=01:05:00

set -eo pipefail

# --- environment (ADJUST FOR DELLA) ----------------------------------------
# Della's torch2 environment is provided through the 2022.5 Anaconda module.
module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2          # <- project environment with PyTorch

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

JOBS_FILE="${JOBS_FILE:-jobs.txt}"
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")
if [ -z "${ARGS}" ]; then
    echo "No job on line ${SLURM_ARRAY_TASK_ID} of ${JOBS_FILE}"; exit 1
fi

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "args: ${ARGS}"

python scripts/train_subset.py ${ARGS}
