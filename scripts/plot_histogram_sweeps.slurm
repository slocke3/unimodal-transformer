#!/bin/bash
# Post-process histogram-enabled window and boundary sweeps after training.
#
# Submit with dependencies on both training arrays:
#   sbatch --dependency=afterok:<window_job>:<boundary_job> \
#       scripts/plot_histogram_sweeps.slurm

#SBATCH --job-name=unimodal-plots
#SBATCH --output=logs/%j_plots.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:03

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_hist

module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

python scripts/plot_histogram_sweeps.py \
    --window_runs runs_hist_window \
    --boundary_runs runs_hist_boundary \
    --output_dir figures_hist
