#!/bin/bash
# Score already-generated molecules on CPU. No GPU.
#
#   sbatch metric_evaluation/score_cpu.sbatch
#
# Scoring needs no GPU at all -- it is RDKit fingerprinting and graph matching over pairs
# that already exist on disk. Running it inside the generation job held a GH200 idle for
# hours; job 1176643 was cancelled mid-scoring for exactly that reason, losing nothing,
# because mols_test.pkl is scoring's INPUT and was already complete.
#
# Why this is worth parallelising and generation is not: measured 68.4 ms/pair
# single-threaded, dominated by Fraggle (45.3 ms) and MCES (18.1 ms), neither of which
# threads internally -- but the 23123 targets are completely independent. Generation, by
# contrast, is 250 SEQUENTIAL denoising steps on a GPU that is already saturated.

#SBATCH --job-name=stm-score
#SBATCH --account=project_2001912
# `small` is the natural home (384 cores) but had 0 idle nodes; `interactive` has a QOS cap
# of cpu=32,mem=63520M, which is still ~32x the serial path and starts immediately. Switch
# back to small (and raise --cpus-per-task) when the partition drains.
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

# Each worker is a separate process; letting BLAS/OMP also thread inside them oversubscribes
# the node and makes it slower, not faster.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

OUT=${OUT:-/scratch/project_2001912/stm-eval/testset_full}
PKL=${PKL:-$OUT/test/mols_test.pkl}
WORKERS=${WORKERS:-$SLURM_CPUS_PER_TASK}

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
echo "cores $SLURM_CPUS_PER_TASK   workers $WORKERS"
[ -s "$PKL" ] || { echo "MISSING $PKL"; exit 1; }

srun python3 tools/paper_metrics.py \
    --cells "stm_test=$PKL" \
    --ks 1 3 5 \
    --workers "$WORKERS" \
    --out "$OUT/metrics.csv"

echo "=== finished @ $(date -Is) ==="
