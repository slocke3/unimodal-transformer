#!/bin/bash
#SBATCH --job-name=markov-figs
#SBATCH --output=logs/%j_markov_figs.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:05:00
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs figures_markov
module purge; module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch2
export PYTHONUNBUFFERED=1; export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
n=0; for d in runs_markov/*/; do [ -f "${d}results.json" ] && n=$((n+1)) || echo "  INCOMPLETE: $d"; done
echo "${n}/14 runs completed"
python scripts/plot_markov.py --runs_dir runs_markov --out_dir figures_markov
echo DONE
