#!/bin/bash
#SBATCH --job-name=induction
#SBATCH --output=logs/%j_induction.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:05:00
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_tilt_prelim
module purge; module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python scripts/probe_induction.py
echo DONE
