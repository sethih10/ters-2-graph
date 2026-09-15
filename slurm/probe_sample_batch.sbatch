#!/bin/bash
# How large should --batch-mols be for SAMPLING? Measured, not reasoned. ~12 min on gputest.
#
#   sbatch slurm/probe_sample_batch.sbatch
#
# slurm/probe_batch_size.py answers this for TRAINING and is not reusable here: it drives
# losses.get_step_fn, i.e. forward + backward + optimizer state. Sampling runs forward-only
# under torch.no_grad(), so it stores no activations for backward, no gradients and no
# optimizer moments -- its per-molecule memory is a different and much smaller number, and
# the training figures in configs/diffspectra_stm.py (batch 256 -> 46.0 GiB) do not transfer.
#
# TWO EFFECTS PULL AGAINST EACH OTHER, which is why this needs measuring rather than a guess:
#
#   + Bigger batches amortise the per-STEP overhead. Sampling makes 250 sequential forward
#     calls per chunk, and each carries fixed cost. From the two points already measured --
#     30 molecules -> 30.8 ms/step, 200 molecules -> 104 ms/step -- the fixed part is ~18 ms
#     and the asymptotic ceiling is ~9.3 mol/s, against 7.7 today. So the whole prize from
#     here is about +20%, not a multiple.
#
#   - Bigger chunks raise max_n. EVERY molecule in a chunk is padded to the chunk's largest,
#     and the STM size histogram is median 23 / p95 35 / max 54. A 40-target chunk may top out
#     near 40; a 160-target chunk is near-certain to contain a 54-atom molecule and then pays
#     54^2 edge cost on all 800 of its molecules. This is the effect the naive extrapolation
#     above ignores, and it is why the answer might be "200 is already right".
#
# Reports mol/s, peak GiB and the max_n actually seen, per batch size.
#
# MEASURED so far (job 1229477, which timed out after two sizes -- see NM below):
#     batch 200   max_n 38   6.7 mol/s   peak 2.7 GiB
#     batch 400   max_n 40   7.1 mol/s   peak 4.8 GiB
# i.e. doubling the batch buys +6% and costs 2.1 GiB of 95.5. MEMORY IS NOT THE CONSTRAINT
# here -- sampling runs forward-only under no_grad, so it uses ~12 MB/molecule against
# training's ~180 MB/molecule (configs/diffspectra_stm.py: batch 256 -> 46.0 GiB). The limit
# is the 250 SEQUENTIAL denoising steps, which no batch size removes.

#SBATCH --job-name=stm-probe-batch
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

RUN=${RUN:-/scratch/project_2001912/stm-runs/stm_20260908_1134521}
CKPT=${CKPT:-$RUN/checkpoints/checkpoint_3.pth}
OUT=${OUT:-/scratch/project_2001912/stm-eval/probe_sample_batch}
NSAMP=5
# batch-mols -> molecules per chunk at 5 samples: 40->200, 80->400, 160->800, 320->1600.
SIZES=${SIZES:-"40 80 160 320"}
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for BM in $SIZES; do
  # 2 chunks: one warm-up (allocator growth, cuDNN autotune) and one measured. Was 3, which
  # made cost quadratic in BM -- at BM=320 that size alone is 960 targets = 4800 molecules =
  # ~11 min, and the 15:00 gputest cap killed job 1229477 after only two sizes. The steady
  # state is reached after one chunk; a third bought nothing but wall-clock.
  NM=$((BM * 2))
  echo
  echo "################ --batch-mols $BM  ($((BM * NSAMP)) molecules/chunk, $NM targets) ####"
  # tee the whole log, THEN filter: an OOM traceback is the most useful output this probe can
  # produce and must not be lost to a grep.
  python3 -m tools.sample_cond --modality stm \
      --ckpt "$CKPT" --split test \
      --n-mols "$NM" --n-samples "$NSAMP" --batch-mols "$BM" --seed 0 \
      --no-geometry --out "$OUT/bm$BM" > "$OUT/bm$BM.log" 2>&1
  RC=$?
  grep -E "batch [0-9]+/" "$OUT/bm$BM.log" || true
  if [ $RC -ne 0 ]; then
    echo "  batch-mols $BM FAILED (rc=$RC) -- that IS the answer for this size:"
    grep -E "CUDA out of memory|RuntimeError|Error" "$OUT/bm$BM.log" | head -3
  fi
done

echo
echo "################ SUMMARY (last chunk of each size = steady state) ################"
printf '%-12s %-10s %s\n' 'batch-mols' 'molecules' 'steady-state chunk'
for BM in $SIZES; do
  L=$(grep -E "batch 2/2" "$OUT/bm$BM.log" 2>/dev/null | tail -1)
  printf '%-12s %-10s %s\n' "$BM" "$((BM * NSAMP))" "${L:-FAILED or no steady-state chunk}"
done
echo
echo "Pick the smallest size whose mol/s is within a few percent of the best: a larger chunk"
echo "raises max_n, and every molecule in a chunk is padded to the chunk max."
echo "=== finished @ $(date -Is) ==="
