#!/bin/bash
# Probe #1: read the model's implied return map out of a trained checkpoint.
#SBATCH --job-name=rmap-probe
#SBATCH --output=logs/%j_rmap_probe.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:05:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_asym
module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

python scripts/probe_return_map.py \
    --runs runs_asym/asym_w0_seed0 runs_asym/asym_w0.4_seed0 \
    --R 1.0 --alphas 1.0 0.9 0.8 0.7 0.6 0.5 \
    --out_dir figures_asym
echo "DONE"
