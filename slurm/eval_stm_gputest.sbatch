#!/bin/bash
# Evaluate an STM checkpoint on gputest.
#
#   sbatch slurm/eval_stm_gputest.sbatch                    # the FINAL checkpoint, 150k
#   CKPT=.../checkpoint_3.pth OUT=.../ckpt3 sbatch ...      # any other one
#
# Runs the project's OWN established diagnostics. Defaults updated 2026-09-11, when run
# 1134521 finished its full 150000 steps (resumed as job 1256193):
#
#   1. diagnose_progress   PREV (ckpt_3, 75k) vs CKPT (ckpt_6, 150k) at FIXED t.
#      The aggregate curve is flat and stays flat -- 3.70 at step 10000, 3.78 at 150000 --
#      and docs/09 5.2/5b says twice over that this is an ARTEFACT of averaging over a
#      uniformly drawn t, not a finding. At fixed t, 25k->75k fell 21.5% at t=0.05 while the
#      curve showed nothing. So: did the second half of training buy anything at all?
#   2. diagnose_conditioning  matched vs shuffled vs zeroed cube on CKPT. At 75k the z-stack
#      was worth +0.98% of loss against TERS's +10.5%, and the embeddings sat in a
#      0.984-cosine cone. docs/09 5b predicts MORE STEPS CANNOT FIX THAT. This is the test:
#      if the cone and the zeroed-cube cost are unchanged at 150k, the prediction holds and
#      step budget is conclusively not the lever.
#   3. diagnose_cond_by_t     WHERE in t the stack matters.
#   4. sample_cond            generation + the geometry/cube payload for visualisation.
#
# OUT defaults to a PER-CHECKPOINT directory, because the 75k artefacts (stm-eval/*.txt) are
# what the 150k numbers get compared against -- writing both to the same path destroys the
# comparison. DISK: the 250 GB quota that killed the training run is gone (2.0 TB allocation,
# ~1.8 TB free as of 2026-09-11); these outputs are JSON + a few MB of .npz regardless.

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

RUN=${RUN:-/scratch/project_2001912/stm-runs/stm_20260908_1134521}
CKPT=${CKPT:-$RUN/checkpoints/checkpoint_6.pth}          # 150k, the finished model
PREV=${PREV:-$RUN/checkpoints/checkpoint_3.pth}          # 75k, what it is compared against
OUT=${OUT:-/scratch/project_2001912/stm-eval/ckpt6_150k}
mkdir -p "$OUT"
CFG="--config configs/diffspectra_stm.py --config-qm9 configs/base_stm.py"

for f in "$CKPT" "$PREV"; do
    [ -s "$f" ] || { echo "MISSING checkpoint: $f"; exit 1; }
done
echo "checkpoint : $CKPT"
echo "compared to: $PREV"
echo "output     : $OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -la "$RUN/checkpoints/"

echo
echo "################ 1. PROGRESS: PREV vs CKPT at fixed t ################"
echo "### did the second half of training buy anything the flat curve cannot show? ###"
srun python3 tools/diagnose_progress.py $CFG \
    --ckpts "$PREV" "$CKPT" \
    --batches 2 --seeds 3 2>&1 | tee "$OUT/progress.txt"

echo
echo "################ 2. CONDITIONING: does the model use the z-stack? ################"
srun python3 tools/diagnose_conditioning.py $CFG \
    --ckpt "$CKPT" \
    --batches 4 --seeds 4 2>&1 | tee "$OUT/conditioning.txt"

echo
echo "################ 3. COND-BY-T: where in t does the stack matter? ################"
srun python3 tools/diagnose_cond_by_t.py $CFG \
    --ckpt "$CKPT" \
    --batches 2 --seeds 3 2>&1 | tee "$OUT/cond_by_t.txt"

echo
echo "################ 4. GENERATION (test split) ################"
srun python3 tools/sample_cond.py $CFG \
    --ckpt "$CKPT" \
    --split test --n-mols 5 --n-samples 5 \
    --out "$OUT/generated" 2>&1 | tee "$OUT/generation.txt"

echo
echo "=== outputs ==="; ls -la "$OUT" "$OUT/generated" 2>/dev/null
echo "=== quota after ==="; lfs quota -g project_2001912 /scratch 2>/dev/null | tail -1
echo "=== finished @ $(date -Is) ==="
