#!/bin/bash
# The FULL test split, sharded across 4 GPUs as 4 independent single-GPU processes.
#
#   sbatch slurm/generate_stm_sharded.sbatch
#   NMOLS=4000 TIME=01:00:00 sbatch slurm/generate_stm_sharded.sbatch     # smaller trial
#
# WHY SHARDING AND NOT DataParallel. MEASURED, gputest job 1229202, identical seed and batches:
#
#     1 GPU   3.9 mol/s          4 GPUs via DataParallel   0.9 mol/s      <- 4.3x SLOWER
#
# (both at 30-molecule batches. At the 200-molecule batches this script uses, one GPU does
#  8.0 mol/s -- job 1229276 -- so a 30-molecule batch understates a real run by ~2x. The
#  DataParallel RATIO is what matters here, and it does not improve with batch size.)
#
# The molecules are IDENTICAL either way (that job compared all 100 SMILES), so this is a
# throughput result, not a correctness one. models/utils.py:27 wraps every model in
# torch.nn.DataParallel, which re-replicates the module to every device on EVERY forward call;
# sampling makes 250 sequential forward calls per chunk, so a 587 MB model is broadcast 250
# times per chunk. Batch size cannot fix it -- the cost is per STEP, not per molecule.
#
# Four separate processes replicate nothing. Each owns one GPU and a disjoint stride of the
# target list (--shard k --n-shards 4), so scaling should be close to linear. The permutation
# is seeded, so the union of the shards is exactly the target set an unsharded run would use.
#
# STRIDED, not contiguous: targets are already in random order, but a contiguous block can
# still draw an unlucky run of 50-atom molecules and become the straggler every other GPU waits
# on. `wait` makes the job as long as its slowest shard.
#
# BATCH SIZE, measured rather than guessed (job 1229477):
#     batch 200 molecules   max_n 38   6.7 mol/s   peak 2.7 GiB
#     batch 400 molecules   max_n 40   7.1 mol/s   peak 4.8 GiB
# Memory is nowhere near the constraint -- sampling is forward-only under no_grad, ~12 MB per
# molecule, against training's ~180 MB (configs/diffspectra_stm.py: batch 256 -> 46.0 GiB).
# The limit is the 250 SEQUENTIAL denoising steps per chunk, which no batch size removes, so
# the whole prize from here is single-digit percent. --batch-mols 80 takes that and stops;
# pushing further mostly raises max_n, and every molecule in a chunk is padded to the chunk
# max (STM sizes: median 23, p95 35, max 54).
#
# Merge before scoring -- no scorer understands a partial cell:
#   python3 -m tools.merge_shards --shards $OUT/shard*/ --out $OUT/merged
#   OUT=$OUT/merged sbatch metric_evaluation/score_run.sbatch
# (this script does the merge itself if every shard exits 0)

#SBATCH --job-name=stm-gen-sharded
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00     # ~1.3 h expected (5781 targets/shard at 8.0 mol/s + startup)
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

RUN=${RUN:-/scratch/project_2001912/stm-runs/stm_20260908_1134521}
CKPT=${CKPT:-$RUN/checkpoints/checkpoint_3.pth}
NMOLS=${NMOLS:-23123}                 # the whole test split
NSAMP=${NSAMP:-5}
BM=${BM:-80}      # 400 molecules/chunk. MEASURED job 1229477: +6% over 200, peak 4.8 GiB of 95.5
SPLIT=${SPLIT:-test}
NSHARD=${NSHARD:-4}
OUT=${OUT:-/scratch/project_2001912/stm-eval/gen_${SPLIT}_${NMOLS}x${NSAMP}_sharded}
mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ckpt   : $CKPT"
echo "out    : $OUT"
echo "targets: $NMOLS x $NSAMP = $((NMOLS * NSAMP)) molecules across $NSHARD shards"
echo "quota before: $(lfs quota -h -p 602001912 /scratch 2>/dev/null | tail -1)"

# Each shard loads its own cubes up front (~917 KB per target), so with 4 shards the whole
# split is ~21 GB of host RAM spread over four processes on a 848 GiB node. Expect several
# minutes of silence per shard before its first batch line.
T0=$SECONDS
PIDS=()
for k in $(seq 0 $((NSHARD - 1))); do
  CUDA_VISIBLE_DEVICES=$k \
    python3 -m tools.sample_cond --modality stm \
      --ckpt "$CKPT" --split "$SPLIT" \
      --n-mols "$NMOLS" --n-samples "$NSAMP" --batch-mols "$BM" --seed 0 \
      --shard "$k" --n-shards "$NSHARD" \
      --no-geometry --save-sdf --out "$OUT/shard$k" > "$OUT/shard$k.log" 2>&1 &
  PIDS+=($!)
  echo "shard $k -> GPU $k, pid ${PIDS[-1]}, log $OUT/shard$k.log"
done

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "shard $i OK"; else echo "shard $i FAILED"; FAIL=1; fi
done
echo "all shards done in $((SECONDS - T0)) s"
for k in $(seq 0 $((NSHARD - 1))); do
  echo "--- shard $k tail ---"; tail -3 "$OUT/shard$k.log" 2>/dev/null
done

if [ "$FAIL" -ne 0 ]; then
  echo "NOT merging: at least one shard failed. Re-run the failed shard(s) with the same"
  echo "--seed/--n-shards and merge by hand -- a partial merge would silently drop targets."
  exit 1
fi

echo
echo "################ MERGE ################"
# trailing slash: match the shard DIRECTORIES, not the shard*.log files beside them
python3 -m tools.merge_shards --shards "$OUT"/shard*/ --out "$OUT/merged" --split "$SPLIT"
RC=$?

echo "quota after : $(lfs quota -h -p 602001912 /scratch 2>/dev/null | tail -1)"
echo
echo "Next:  OUT=$OUT/merged sbatch metric_evaluation/score_run.sbatch"
echo "=== finished @ $(date -Is) ==="
exit $RC
