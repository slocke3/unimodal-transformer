#!/bin/bash
# Post-process the tilted-logistic sweep. CPU only; the analysis is torch-free.
#SBATCH --job-name=tilt-figs
#SBATCH --output=logs/%j_tilt_figs.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:05:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_tilt
module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "=== completion audit ==="
n=0; for d in runs_tilt/*/; do [ -f "${d}eval_asym.npz" ] && n=$((n+1)) || echo "  INCOMPLETE: $d"; done
echo "  ${n}/20 runs produced an eval file"

echo; echo "=== convergence ==="
python - <<'PY'
import json, glob, os
for d in sorted(glob.glob("runs_tilt/*/history.json")):
    v = (json.load(open(d)).get("val_losses") or [])
    if len(v) < 10: continue
    drop = (v[-10] - v[-1]) / max(v[-10], 1e-12)
    print(f"  {os.path.dirname(d):<34} final_val={v[-1]:.4f} "
          f"drop={drop:+.1%} {'STILL IMPROVING' if drop > 0.02 else 'converged'}")
PY

echo; echo "=== floor-relative analysis ==="
python scripts/plot_tilted_sweep.py --runs_dir runs_tilt --out_dir figures_tilt

echo; echo "=== per-band CE surfaces ==="
python scripts/plot_asym_bands.py --runs_dir runs_tilt --out_dir figures_tilt \
    || echo "  (band plots skipped)"
echo DONE
