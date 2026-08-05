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
#SBATCH --time=01:00:00

set -euo pipefail

# --- environment (ADJUST FOR DELLA) ----------------------------------------
# della: `module load anaconda3/<version>` then activate the project env that
# has torch + numpy. Confirm the exact module name with `module avail anaconda`.
module purge
module load anaconda3/2024.6 || module load anaconda3
conda activate unimodal        # <- the env you created for this project

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1

JOBS_FILE="${JOBS_FILE:-jobs.txt}"
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")
if [ -z "${ARGS}" ]; then
    echo "No job on line ${SLURM_ARRAY_TASK_ID} of ${JOBS_FILE}"; exit 1
fi

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "args: ${ARGS}"

python scripts/train_subset.py ${ARGS}
