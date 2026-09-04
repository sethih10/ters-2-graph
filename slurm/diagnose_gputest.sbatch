#!/bin/bash
# Verify wandb logging end to end on a compute node, and MEASURE whether that node
# can reach the internet at all -- which is what decides offline vs online mode.
#SBATCH --job-name=ters-diag
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
set -uo pipefail          # NOT -e: the connectivity probes are expected to fail
mkdir -p slurm/logs
module purge
module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"


# Answers "is the model using the cube at all?" without needing sampling.py (D-8).
python -m tools.diagnose_conditioning \
    --ckpt exp/ters_20260828_925597/checkpoints/checkpoint_5.pth \
    --batches 4 --seeds 5
echo "finished : $(date -Is)"
