#!/bin/bash
# Read the representation-only control off the finished asym sweep.
#
#   sbatch --dependency=afterany:<array_job_id> scripts/make_conjugacy_figure.slurm
#
# CPU only; the analysis is torch-free (numpy dynamics + matplotlib).

#SBATCH --job-name=asym-conj
#SBATCH --output=logs/%j_conjugacy.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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

python scripts/analyze_conjugacy_curve.py \
    --runs_dir runs_asym --out_dir figures_asym
echo "DONE"
