#!/bin/bash
# Measure the v2 STM encoder before anyone trains with it.
#
#   sbatch slurm/probe_stm_encoder_v2.sbatch
#
# Four things, in the order that decides whether v2 is worth a 12 h training slot:
#
#   1. --self-test  shapes, gradients, residual wiring. Cheap, and it runs on CPU anyway;
#                   here it guards against the GPU build differing from the login-node one.
#   2. --compare    v1 vs v2 depth, width, parameters, z schedule.
#   3. --bench      params, PEAK MEMORY and fwd+bwd time at the real batch of 128. This is
#                   the number that decides feasibility: v1 training peaks at 23.4 GiB of 95
#                   (slurm/train_stm_gpumedium.sbatch), and v2's blocks are 12x wider. A
#                   cost not measured is a guess, and this project has been wrong by 7.7x
#                   that way before (docs/05-progress-log.md, 2026-09-08).
#   4. --cone       THE OPEN QUESTION. docs/09-research-summary.md 5b calls the untrained
#                   cone "the cheapest measurement available": the trained encoder sits at
#                   mean pairwise cosine 0.967, and whether that is a property of the DATA
#                   or damage done BY training is not known. Run on real val cubes, both
#                   encoders, same seed. Random-init near 0.97 supports hypothesis (a);
#                   far below it supports (c) and points at an auxiliary loss instead.

#SBATCH --job-name=stm-enc-v2
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail          # NOT -e: a later probe must still run if an earlier one fails
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

H5=${H5:-/scratch/project_2001912/stm-data/val.h5}
OUT=${OUT:-/scratch/project_2001912/stm-eval/encoder_v2}
mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -s "$H5" ] || { echo "MISSING: $H5"; exit 1; }

echo; echo "################ 1. SELF-TEST ################"
srun python3 -m models.stm_encoder_v2 --self-test 2>&1 | tee "$OUT/self_test.txt"

echo; echo "################ 2. v1 vs v2 ################"
srun python3 -m models.stm_encoder_v2 --compare  2>&1 | tee "$OUT/compare.txt"

echo; echo "################ 3. COST at the real batch size ################"
srun python3 -m models.stm_encoder_v2 --bench --batch 128 2>&1 | tee "$OUT/bench.txt"

echo; echo "################ 4. UNTRAINED CONE on real cubes ################"
echo "### trained reference: v1 0.984 @75k, 0.967 @150k; TERS 0.925 @28k ###"
srun python3 -m models.stm_encoder_v2 --cone "$H5" -n 256 2>&1 | tee "$OUT/cone.txt"

echo; echo "=== outputs ==="; ls -la "$OUT"
echo "=== finished @ $(date -Is) ==="
