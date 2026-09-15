#!/bin/bash
# Answer two questions the headline metrics cannot answer on their own.
#
#   sbatch slurm/diagnose_metrics.sbatch
#
# 1. IS THE DISTRIBUTIONAL RESULT REAL, or just corpus statistics?
#    The corpus is ONE 8:1:1 random split, so train and test share a distribution BY
#    CONSTRUCTION. A model that learned only the training marginal -- ignoring the cube
#    entirely -- would therefore still score well on FCD/Frag/Scaf against the test reference.
#    So: score 115615 REAL TRAINING molecules against that same reference, at the same n as
#    the generated set. That is the floor. If the generated set does not beat it, the
#    distributional metrics are reporting the corpus, not the model.
#    This is the distributional analogue of --shuffle-cubes.
#
# 2. WHY IS METRIC-3D mol_stable ~0.17 EVEN ON CORRECT MOLECULES?
#    Two suspects, never separated: QM9-tuned distance margins (H/C/N/O/F only) vs this
#    corpus's Si/P/S/Cl/Br, and a CHARGE-BLIND valence table (`allowed_bonds`) against a
#    corpus with ~24k O-1 / ~14k N1 / ~13k S1. tools/diagnose_stability3d.py strata the
#    ground-truth failures by both, and prints 2D on the same molecules as the control.

#SBATCH --job-name=stm-diag-metrics
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail
mkdir -p slurm/logs
module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT=${OUT:-/scratch/project_2001912/stm-eval/diagnose_metrics}
N=${N:-115615}                       # match the generated set exactly; FCD is n-sensitive
mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="

echo
echo "############ 1. WHY METRIC-3D FAILS ON CORRECT MOLECULES ############"
srun python3 tools/diagnose_stability3d.py --config-qm9 configs/base_stm.py -n 3000 \
  2>&1 | tee "$OUT/stability3d.txt"

echo
echo "############ 2. DISTRIBUTIONAL NULL: real TRAINING molecules, n=$N ############"
echo "### if the generated set does not beat this, FCD/Frag/Scaf measured the corpus ###"
srun python3 tools/eval_cond.py \
    --config-qm9 configs/base_stm.py \
    --ground-truth "$N" --ground-truth-split train \
    --workers "$SLURM_CPUS_PER_TASK" --n-jobs 8 \
    --out "$OUT/null_train.json" 2>&1 | tee "$OUT/null_train.txt"

echo
echo "=== outputs ==="; ls -la "$OUT"
echo "=== finished @ $(date -Is) ==="
