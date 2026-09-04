#!/bin/bash
# Verify the REAL entry point (main.py -> run_lib.train) starts and takes steps on GPU.
# The probe exercises a hand-built step; this exercises the shipped training loop,
# including the evaluation imports and the checkpoint machinery the probe skips.
# snapshot_freq is forced low so checkpoints are actually WRITTEN here; snapshot_sampling
# is deliberately NOT overridden, so this also proves the config's value is survivable.
#SBATCH --job-name=ters-verify
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=211G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
set -euo pipefail
mkdir -p slurm/logs
module purge
module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"
echo "=== $SLURM_JOB_ID @ $(date -Is) ==="
# ml_collections allows overriding existing keys even under lock_config=True.
srun python3 main.py \
    --config configs/diffspectra_ters.py \
    --config_original_qm9 configs/base_ters.py \
    --mode train \
    --workdir "exp/verify_${SLURM_JOB_ID}" \
    --config.training.n_iters=30 \
    --config.training.log_freq=10 \
    --config.training.snapshot_freq=10 \
    --config.training.snapshot_freq_for_preemption=10
echo "=== finished @ $(date -Is) ==="
