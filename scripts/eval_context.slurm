#!/bin/bash
#SBATCH --job-name=evalctx
#SBATCH --output=logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,mig
#SBATCH --mem=16G
# Just over an hour: <=1h routes to della's test QOS.
#SBATCH --time=01:30:00
set -eo pipefail
module purge; module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch2
cd "${SLURM_SUBMIT_DIR}"; export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python scripts/eval_context.py --runs 'runs_divbins_320k/in64_mse_m*_seed0' \
    --out figures_taskdiv/eval_context.npz
