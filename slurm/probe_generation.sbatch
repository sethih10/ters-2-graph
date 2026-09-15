#!/bin/bash
# THE GATE for multi-GPU generation. Nothing larger is submitted until this passes. ~10 min.
#
#   sbatch slurm/probe_generation.sbatch
#
# Answers two questions, in this order, because the second is meaningless if the first fails:
#
#   1. CORRECTNESS. Does 4-GPU generation produce the SAME molecules as 1-GPU at the same seed?
#   2. THROUGHPUT.  What is the real mol/s at 1 and at 4 GPUs? That sets --time for the full run.
#
# WHY 1 IS IN DOUBT. models/utils.py:27 wraps every model in torch.nn.DataParallel, so this
# script already uses all visible GPUs with no sharding code. But DMT.forward takes `edge_mask`
# PRE-FLATTENED to [B*N*N, 1] and immediately does
#
#     adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)          # models/dmt.py:436
#
# with `bs` taken from xh.shape. DataParallel scatters xh and edge_mask along dim 0
# INDEPENDENTLY, so the two describe the same molecules only when the batch divides evenly
# across devices. B=101 on 4 GPUs with N=10: xh -> 26/26/26/23, edge_mask -> 2525 rows each =
# 25.25 molecules. tools/sample_cond.py now pads each chunk up to a multiple of the device count
# and drops the extras -- this job is what proves that padding is right rather than plausible.
#
# --batch-mols 6 is deliberate, not a round number: 6 x 5 = 30 molecules per chunk, and 30 % 4
# = 2, so EVERY chunk exercises the padding path, and the final chunk (20 % 6 = 2 targets = 10
# molecules) exercises the ragged tail too. A tidy batch that happens to divide by 4 would
# prove nothing.
#
# IF THE SMILES DIFFER: do not "fix" it by using more GPUs carefully. It would mean DataParallel
# is silently mis-splitting rather than raising, which also casts doubt on any multi-GPU TRAINING
# this repo has done. Report it, run the full split single-GPU, and open a defect.

#SBATCH --job-name=stm-probe-gen
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:4
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
OUT=${OUT:-/scratch/project_2001912/stm-eval/probe_generation}
NMOLS=${NMOLS:-20}
NSAMP=${NSAMP:-5}
BM=${BM:-6}
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ckpt: $CKPT   targets: $NMOLS x $NSAMP samples   batch-mols: $BM"

for NG in 1 4; do
  echo
  echo "################ $NG GPU(s) ################"
  DEV=$(seq -s, 0 $((NG - 1)))
  # SECONDS, not /usr/bin/time -- the GNU binary is not installed on these nodes (job 1229189).
  T0=$SECONDS
  CUDA_VISIBLE_DEVICES="$DEV" \
    python3 -m tools.sample_cond --modality stm \
      --ckpt "$CKPT" --split test \
      --n-mols "$NMOLS" --n-samples "$NSAMP" --batch-mols "$BM" --seed 0 \
      --no-geometry --out "$OUT/gpu$NG" 2>&1 | tee "$OUT/gpu$NG.log"
  echo "WALL_${NG}GPU $((SECONDS - T0)) s (incl. model + dataset load)" | tee -a "$OUT/gpu$NG.log"
done

echo
echo "################ CORRECTNESS: 1 GPU vs 4 GPU ################"
python3 - "$OUT/gpu1/samples_test.json" "$OUT/gpu4/samples_test.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
# keyed by ds_index: `results` is sorted by best_tanimoto and ties can reorder
sm = lambda d: {r['ds_index']: [s['smiles'] for s in r['samples']] for r in d['results']}
x, y = sm(a), sm(b)
if set(x) != set(y):
    print(f'FAIL: different targets\n  1gpu {sorted(x)}\n  4gpu {sorted(y)}'); sys.exit(1)
bad = [i for i in x if x[i] != y[i]]
for i in bad:
    print(f'  target {i}:\n    1gpu {x[i]}\n    4gpu {y[i]}')
print(f'{len(x)} targets, {sum(len(v) for v in x.values())} samples')
if bad:
    print(f'FAIL: {len(bad)} target(s) differ -- DataParallel is mis-splitting the batch. '
          f'Run the full split on ONE GPU and open a defect (see this script header).')
    sys.exit(1)
print('SMILES IDENTICAL -- 4-GPU DataParallel generation is correct')
print(f"1gpu stability {a['stability']}  rdkit {a['rdkit']}")
print(f"4gpu stability {b['stability']}  rdkit {b['rdkit']}")
PY
GATE=$?

echo
echo "################ THROUGHPUT ################"
grep -h "WALL_" "$OUT"/gpu*.log 2>/dev/null || true
grep -hE "mol/s" "$OUT"/gpu1.log | tail -1 | sed 's/^/1 GPU: /'
grep -hE "mol/s" "$OUT"/gpu4.log | tail -1 | sed 's/^/4 GPU: /'
echo
echo "Use the 4-GPU mol/s to size slurm/generate_stm_full.sbatch --time:"
echo "  115615 molecules / <mol/s> = seconds, then add ~10 min of startup (23123 cube reads)."
echo "gate exit: $GATE"
echo "=== finished @ $(date -Is) ==="
exit $GATE
