#!/bin/bash
# Conditional generation from STM z-stacks at scale. ONE script for both production runs --
# they differ only in --n-mols, so forking them would be two copies of the same 60 lines.
#
#   # Run A -- 1000 test targets x 5 samples = 5000 molecules
#   NMOLS=1000 TIME=02:00:00 sbatch slurm/generate_stm.sbatch
#
#   # Run B -- the whole test split, 23123 x 5 = 115615 molecules
#   NMOLS=23123 TIME=<from the probe> sbatch slurm/generate_stm.sbatch
#
# ONE GPU, DELIBERATELY. Do not "speed this up" by asking for four.
# MEASURED, gputest job 1229202, 20 targets x 5 samples, identical seed and batches:
#
#     1 GPU   3.9 mol/s          4 GPUs   0.9 mol/s        <- 4.3x SLOWER
#
# Both at 30-molecule batches. MEASURED again at this script's 200-molecule batches
# (--batch-mols 40, job 1229276): 8.0 mol/s on one GPU -- so a small batch understates a
# real run by ~2x, and 115615 molecules is ~4.0 h on one GPU.
#
# The molecules are IDENTICAL either way (that job's gate compared all 100 SMILES), so this is
# purely a throughput result and not a correctness one. The cause is torch.nn.DataParallel,
# which models/utils.py:27 applies to every model: it re-replicates the module to every device
# on EVERY forward call. Sampling makes 250 sequential forward calls per chunk, so a 587 MB
# model is broadcast 250 times per chunk -- hundreds of GB of interconnect traffic to save a
# few seconds of arithmetic. Batch size does not fix it: the replication cost is per STEP.
#
# To actually use more GPUs, run independent single-GPU processes over disjoint target shards
# (no replication at all). That is a different mechanism from DataParallel and is not this
# script.
#
# Submit slurm/probe_generation.sbatch first on any new checkpoint or node type: it is the
# correctness gate (DataParallel + the flattened edge_mask at models/dmt.py:436) and it
# measures the mol/s that sets TIME here. Sizing this from an estimate is how the STM training
# config ended up wrong by 7.7x.
#
# MATCHED ARM ONLY, 5 SAMPLES -- a deliberate choice, with two consequences that must travel
# with every number this produces:
#   * --ks 1 3 5, NEVER 10. paper_metrics refuses a k above --n-samples rather than reporting a
#     top-10 from 5 candidates, so the docs/09 headline (top-10 = 0.350) has NO counterpart here.
#   * No shuffled null. The sampler is handed the TRUE atom count (tools/sample_cond.py:178), so
#     a Tanimoto in isolation mostly reports the prior. These runs measure scale and
#     distributional quality, NOT the matched-minus-shuffled gap; docs/experimental.md 2d
#     remains the reference for what the null is worth.
#
# --no-geometry is NOT optional at this scale: the geometry payload is ~143 KB per TARGET
# (a full cube projection each), i.e. 3.3 GB for the full split, and /scratch sits at 176 G of
# a 250 G quota. This project has already lost two jobs to quota kills.
#
# --batch-mols: chunks are padded up to a multiple of the GPU count (sample_cond.py, see the
# DataParallel note there), so any value is SAFE. It trades memory against per-chunk overhead;
# 40 targets x 5 = 200 molecules per chunk was chosen to sit well under the measured 46 GiB that
# batch 256 used during training.

#SBATCH --job-name=stm-generate
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1          # ONE. See the header -- 4 is 4.3x slower.
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

RUN=${RUN:-/scratch/project_2001912/stm-runs/stm_20260908_1134521}
CKPT=${CKPT:-$RUN/checkpoints/checkpoint_3.pth}
NMOLS=${NMOLS:-1000}
NSAMP=${NSAMP:-5}
BM=${BM:-40}
SPLIT=${SPLIT:-test}
OUT=${OUT:-/scratch/project_2001912/stm-eval/gen_${SPLIT}_${NMOLS}x${NSAMP}}
mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ckpt   : $CKPT"
echo "out    : $OUT"
echo "targets: $NMOLS x $NSAMP samples = $((NMOLS * NSAMP)) molecules, batch-mols $BM"
echo "GPUs visible: $(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)"
echo "quota before: $(lfs quota -h -p 602001912 /scratch 2>/dev/null | tail -1)"

# NOTE the startup cost before the first batch line appears: sample_cond loads one 917 KB cube
# per target up front (tools/sample_cond.py:172-181), so the full split reads ~21 GB off Lustre
# serially before generation starts. Silence for several minutes at the beginning is expected.
srun python3 -m tools.sample_cond --modality stm \
    --ckpt "$CKPT" --split "$SPLIT" \
    --n-mols "$NMOLS" --n-samples "$NSAMP" --batch-mols "$BM" --seed 0 \
    --no-geometry --save-sdf --out "$OUT"
RC=$?

echo
echo "=== outputs ==="; ls -la "$OUT"
echo "quota after : $(lfs quota -h -p 602001912 /scratch 2>/dev/null | tail -1)"
echo "generation exit: $RC"
echo
echo "Next -- metrics (separate job, CPU-heavy, does not hold 4 GPUs):"
echo "  OUT=$OUT sbatch metric_evaluation/score_run.sbatch"
echo "=== finished @ $(date -Is) ==="
exit $RC
