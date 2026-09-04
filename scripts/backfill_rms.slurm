#!/bin/bash
#SBATCH --job-name=backfill-rms
#SBATCH --output=logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,mig
#SBATCH --mem=16G
# Just over an hour: <=1h is routed to della's test QOS. The work is ~15 min.
#SBATCH --time=01:05:00
set -eo pipefail
module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python scripts/backfill_rms.py --runs runs_div_controlled
