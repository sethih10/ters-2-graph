#!/bin/bash
# Paper metrics over the WHOLE test split — matched cubes only, 5 samples per target.
#
#   sbatch metric_evaluation/run_metrics_testset.sbatch
#
# Differences from run_metrics_gpumedium.sbatch, all deliberate:
#   * ONE cell (test, matched). No shuffled null.
#   * 5 samples per target, so k tops out at 5 -- there is no top-10 here.
#   * the whole split (23123 targets), not a 20-target sample.
#
# >>> Without the null column, the similarity numbers are NOT interpretable on their own. <<<
# Measured on the 20-target run, a shuffled cube still scores 0.194 Tanimoto Morgan and 0.435
# MACCS -- that is the prior, given that the sampler is handed the true atom count. So read
# top-k (where the null was 0.000) as the real number here, and treat the similarity columns
# as descriptive only. The null cells from paper_metrics/ remain on disk for comparison.
#
# Sizing, from the measured rate of ~8 molecules/s at 200/batch (job 1176026):
#   23123 targets x 5 samples = 115615 molecules  ->  ~4 h at that rate, less at a big batch.
# BATCH_MOLS=400 => 2000 molecules per forward pass. Generation is forward-only under
# no_grad, so it is far lighter than the 23.4 GiB the same model used TRAINING at batch 128;
# the GPU has 95 GiB. If it OOMs, halve BATCH_MOLS -- that is the only knob that matters.

#SBATCH --job-name=stm-testset
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=211G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

CKPT=${CKPT:-/scratch/project_2001912/stm-runs/stm_20260908_1134521/checkpoints/checkpoint_3.pth}
OUT=${OUT:-/scratch/project_2001912/stm-eval/testset_full}
N_MOLS=${N_MOLS:-23123}          # the whole test split
N_SAMPLES=${N_SAMPLES:-5}
BATCH_MOLS=${BATCH_MOLS:-400}    # x N_SAMPLES = molecules per forward pass

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -s "$CKPT" ] || { echo "MISSING checkpoint: $CKPT"; exit 1; }
python3 -c "import pulp, myopic_mces" || { echo "pulp/myopic_mces missing"; exit 1; }
echo "targets $N_MOLS x $N_SAMPLES samples = $((N_MOLS * N_SAMPLES)) molecules, "\
     "$((BATCH_MOLS * N_SAMPLES)) per forward pass"

srun python3 metric_evaluation/run_metrics.py \
    --ckpt "$CKPT" --out "$OUT" \
    --cells test \
    --n-mols "$N_MOLS" --n-samples "$N_SAMPLES" --ks 1 3 5 \
    --batch-mols "$BATCH_MOLS" --no-geometry \
    --csv "$OUT/metrics.csv"

echo "=== finished @ $(date -Is) ==="
