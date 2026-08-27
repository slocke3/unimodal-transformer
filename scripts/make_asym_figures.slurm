#!/bin/bash
# Post-process the asymmetric-family band sweep once training finishes.
#
#   sbatch --dependency=afterany:<array_job_id> scripts/make_asym_figures.slurm
#
# afterany, not afterok: if a task dies we still want figures for the rest plus
# an explicit note of what is missing. CPU only -- the analysis is torch-free.

#SBATCH --job-name=asym-figs
#SBATCH --output=logs/%j_asym_figs.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:05:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs "${OUT_DIR:-figures_asym}"

module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# Which sweep to analyse; override with RUNS_DIR/OUT_DIR/N_EXPECTED at submit.
RUNS_DIR="${RUNS_DIR:-runs_asym}"
OUT_DIR="${OUT_DIR:-figures_asym}"
N_EXPECTED="${N_EXPECTED:-10}"
echo "analysing ${RUNS_DIR} -> ${OUT_DIR}"

echo "=== completion audit ==="
found=0
for d in "${RUNS_DIR}"/*/; do
    [ -d "$d" ] || continue
    if [ -f "${d}eval_asym.npz" ]; then found=$((found + 1));
    else echo "  INCOMPLETE: $d"; fi
done
echo "  ${found}/${N_EXPECTED} runs produced an eval file"

echo
echo "=== convergence check ==="
python - <<'PY'
import json, glob, os
import os as _os
for d in sorted(glob.glob(_os.environ.get("RUNS_DIR", "runs_asym") + "/*/history.json")):
    h = json.load(open(d))
    v = h.get("val_losses") or []
    if len(v) < 10:
        continue
    drop = (v[-10] - v[-1]) / max(v[-10], 1e-12)
    print(f"  {os.path.dirname(d):<32} final_val={v[-1]:.4f} "
          f"last-10%-drop={drop:+.1%}  "
          f"{'STILL IMPROVING' if drop > 0.02 else 'converged'}")
PY

echo
echo "=== figures ==="
python scripts/plot_asym_bands.py --runs_dir "${RUNS_DIR}" --out_dir "${OUT_DIR}"
python scripts/analyze_conjugacy_curve.py --runs_dir "${RUNS_DIR}" --out_dir "${OUT_DIR}"
echo "DONE"
