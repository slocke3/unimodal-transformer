#!/bin/bash
# Baselines + the Figure-3 analogue for the asym2 sweep. CPU only, torch-free.
#SBATCH --job-name=fig3-analogue
#SBATCH --output=logs/%j_fig3.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_asym2
module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "=== model-free reference predictors on the held-out region ==="
python scripts/baseline_band_bayes.py \
    --out figures_asym2/band_bayes_baseline.npz

echo
echo "=== figure ==="
python scripts/plot_figure3_analogue.py \
    --runs_dir runs_asym2 \
    --baseline figures_asym2/band_bayes_baseline.npz \
    --out_dir figures_asym2
echo "DONE"
