#!/bin/bash
# Evaluate the STM run (job 1134521) on gputest.
#
#   sbatch slurm/eval_stm_gputest.sbatch
#
# Runs the project's OWN established diagnostics, in the order that answers the questions
# raised by the training curve (docs/09-research-summary.md 5.2):
#
#   1. diagnose_progress   ckpt_1 (25k) vs ckpt_3 (75k) at FIXED t.
#      >>> This is the one that matters. <<< The aggregate training loss went flat at ~3.70
#      by step 10000 and moved 2.0% over the next 78000 steps -- 0.27x the step-to-step
#      noise. On TERS that same flatness was shown to be an ARTEFACT of averaging over a
#      uniformly drawn t: the model had halved its loss at low t while the curve showed
#      nothing. So: did the STM model keep learning after its curve went flat, or not?
#   2. diagnose_conditioning  matched vs shuffled vs zeroed cube on ckpt_3 -- does the model
#      actually USE the z-stack, or is the conditioning decorative?
#   3. diagnose_cond_by_t     WHERE in t the stack matters.
#   4. sample_ters            generation + the geometry/cube payload for visualisation.
#
# DISK: /scratch is over its 250 GB project quota (250.5 G used, 251 G hard limit, in the
# 6-day grace period) and $HOME has ~80 MB free. Outputs here are JSON + a .npz of a few MB,
# which fits the ~500 MB of headroom under the hard limit -- but do not add to this without
# freeing space first, and do not write checkpoints from this job.

#SBATCH --job-name=stm-eval
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

set -uo pipefail          # NOT -e: a later probe must still run if an earlier one fails
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

RUN=/scratch/project_2001912/stm-runs/stm_20260908_1134521
OUT=/scratch/project_2001912/stm-eval
mkdir -p "$OUT"
CFG="--config configs/diffspectra_stm.py --config-qm9 configs/base_stm.py"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -la "$RUN/checkpoints/"

echo
echo "################ 1. PROGRESS: ckpt_1 (25k) vs ckpt_3 (75k) at fixed t ################"
echo "### did it keep learning after the training curve went flat at step ~10000? ###"
srun python3 tools/diagnose_progress.py $CFG \
    --ckpts "$RUN/checkpoints/checkpoint_1.pth" "$RUN/checkpoints/checkpoint_3.pth" \
    --batches 2 --seeds 3 2>&1 | tee "$OUT/progress.txt"

echo
echo "################ 2. CONDITIONING: does the model use the z-stack? ################"
srun python3 tools/diagnose_conditioning.py $CFG \
    --ckpt "$RUN/checkpoints/checkpoint_3.pth" \
    --batches 4 --seeds 4 2>&1 | tee "$OUT/conditioning.txt"

echo
echo "################ 3. COND-BY-T: where in t does the stack matter? ################"
srun python3 tools/diagnose_cond_by_t.py $CFG \
    --ckpt "$RUN/checkpoints/checkpoint_3.pth" \
    --batches 2 --seeds 3 2>&1 | tee "$OUT/cond_by_t.txt"

echo
echo "################ 4. GENERATION (test split) ################"
srun python3 tools/sample_ters.py $CFG \
    --ckpt "$RUN/checkpoints/checkpoint_3.pth" \
    --split test --n-mols 5 --n-samples 5 \
    --out "$OUT/generated" 2>&1 | tee "$OUT/generation.txt"

echo
echo "=== outputs ==="; ls -la "$OUT" "$OUT/generated" 2>/dev/null
echo "=== quota after ==="; lfs quota -h -p 602001912 /scratch 2>/dev/null | tail -1
echo "=== finished @ $(date -Is) ==="
