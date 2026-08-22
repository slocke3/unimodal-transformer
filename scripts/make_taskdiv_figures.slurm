#!/bin/bash
# Post-process the task-diversity sweeps once the training array finishes.
#
# Submit with a dependency on the training array:
#   sbatch --dependency=afterany:<array_job_id> scripts/make_taskdiv_figures.slurm
#
# afterany rather than afterok on purpose: if one task dies we still want the
# figures for everything that did land, plus an explicit audit of what is
# missing. The plot script globs whatever runs exist and reports the roster.
#
# CPU only -- the whole analysis path is torch-free (numpy + matplotlib).

#SBATCH --job-name=taskdiv-figs
#SBATCH --output=logs/%j_taskdiv_figs.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:05:00

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs figures_taskdiv

module purge
module load anaconda3/2022.5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch2

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "=== completion audit ==="
expected=17
found=0
for d in runs_div_uncontrolled/*/ runs_div_controlled/*/; do
    [ -d "$d" ] || continue
    if [ -f "${d}eval_per_r.npz" ]; then
        found=$((found + 1))
    else
        echo "  INCOMPLETE: $d (no eval_per_r.npz)"
    fi
done
echo "  ${found}/${expected} runs produced an eval file"

echo
echo "=== convergence check: is the largest run still improving at the end? ==="
# The point of the 160k-step budget is that every task count is trained to
# convergence; if the tail of the val curve is still falling, it was not.
python - <<'PY'
import json, glob, os
for d in sorted(glob.glob("runs_div_*/*/history.json")):
    try:
        h = json.load(open(d))
    except Exception as exc:
        print(f"  {d}: unreadable ({exc})"); continue
    v = h.get("val_losses") or []
    if len(v) < 10:
        continue
    tail_drop = (v[-10] - v[-1]) / max(v[-10], 1e-12)
    flag = "STILL IMPROVING" if tail_drop > 0.02 else "converged"
    print(f"  {os.path.dirname(d):<45} final_val={v[-1]:.4f} "
          f"last-10%-drop={tail_drop:+.1%}  {flag}")
PY

echo
echo "=== figures ==="
python scripts/plot_taskdiv.py \
    --uncontrolled_runs runs_div_uncontrolled \
    --controlled_runs runs_div_controlled \
    --out_dir figures_taskdiv

echo
echo "=== overfitting curves (fixed-step arm, both sweeps) ==="
python scripts/plot_budget_sweep.py \
    --runs_dir runs_div_uncontrolled \
    --out_dir figures_taskdiv || echo "  (skipped: budget-sweep plot not applicable)"

echo "DONE"
