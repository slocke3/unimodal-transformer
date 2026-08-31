#!/bin/bash
#SBATCH --job-name=cont-figs
#SBATCH --output=logs/%j_cont_figs.out
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=01:05:00
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs figures_cont
module purge; module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch2
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
n=0; for d in runs_cont/*/; do [ -f "${d}eval_cont.npz" ] && n=$((n+1)) || echo "  INCOMPLETE: $d"; done
echo "${n}/22 runs completed"
python scripts/plot_cont_sweep.py --runs_dir runs_cont --out_dir figures_cont
echo DONE
