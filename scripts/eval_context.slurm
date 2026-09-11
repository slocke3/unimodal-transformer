#!/bin/bash
#SBATCH --job-name=evalctx
#SBATCH --output=logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,mig
#SBATCH --mem=16G
# Two previous attempts timed out at 1:30. The cost is forward passes, not data:
# attend_last hands the encoder an explicit non-causal mask, which disables the
# fused attention kernel and costs 95.6 ms against 37.2 ms per batch. A MIG slice
# is roughly a seventh of a full GPU, which turned 16 minutes per model into 95.
# --r_stride 5 cuts the r grid from 500 to 100 points and reproduces the
# aggregate to within 2%, so this is ~20 min on a full GPU and ~2.5h on a slice.
#SBATCH --time=04:00:00
set -eo pipefail
module purge; module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate torch2
cd "${SLURM_SUBMIT_DIR}"; export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
# Denser than the first pass, which resolved the collapse from 30 to 50 -- two
# decades of error -- on three points. 3-step spacing above L=20 pins its shape;
# below 8 the curve is noisy and dense sampling there buys nothing.
python scripts/eval_context.py --runs 'runs_divbins_320k/in64_mse_m*_seed0' \
    --contexts 1 2 3 4 6 8 10 13 16 20 23 26 29 32 35 38 41 44 47 50 \
    --r_stride 5 --out figures_taskdiv/eval_context.npz
