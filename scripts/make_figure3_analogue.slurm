#!/bin/bash
# Baselines + the Figure-3 analogue for the asym2 sweep. CPU only, torch-free.
#SBATCH --job-name=fig3-analogue
#SBATCH --output=logs/%j_fig3.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

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
# Cost scales as sum_w n_cand(w) x n_cells x n_contexts. The first attempt
# (855 candidates, 36 cells, 21600 contexts) got through one order in 3 h, so
# the candidate grid and the held-out sample are both thinned here. The
# quantities are means over held-out cells, so this costs precision, not bias.
python scripts/baseline_band_bayes.py \
    --cand_alpha_step 0.025 --n_cand_R 8 \
    --R_eval 0.4 0.65 0.85 1.0 --n_traj 4 \
    --eps_grid 3e-3 3e-2 3e-1 \
    --out figures_asym2/band_bayes_baseline.npz

echo
echo "=== figure ==="
python scripts/plot_figure3_analogue.py \
    --runs_dir runs_asym2 \
    --baseline figures_asym2/band_bayes_baseline.npz \
    --out_dir figures_asym2
echo "DONE"
