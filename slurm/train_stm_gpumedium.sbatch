#!/bin/bash
# STM training run on gpumedium.  ONE GPU -- for the same reason train_gpumedium.sbatch is:
# molecule-views seen is `base_batch_size * 28000` no matter how many GPUs, because the
# config multiplies batch_size by num_gpus and divides n_iters by it. Extra GPUs finish the
# same work sooner and cost more GPU-hours; models/utils.py:27 is DataParallel, not DDP.
#
# ---------------------------------------------------------------------------
# BEFORE the first run
# ---------------------------------------------------------------------------
# 1. Build the graph sidecars (once, ~40 min for all three splits, CPU-only):
#
#      module load python-pytorch/2.10
#      export PYTHONPATH=$HOME/.cache/ters-overlay:$PWD
#      python3 -m datasets.preprocess_stm \
#          --data-dir /scratch/project_2001912/stm-data \
#          --out-dir  /scratch/project_2001912/stm-graphs --stats
#
#    Already done as of 2026-09-08: 184157 / 23067 / 23123 molecules kept
#    (99.988% / 100% / 100%; the 23 train losses are hypervalent-Si valence errors).
#
# 2. Sanity-check the pipeline:  python3 -m tests.test_stm     # 47 checks
#
# ---------------------------------------------------------------------------
# Everything below is MEASURED (gputest job 1134273, slurm/smoke_stm_gputest.sbatch)
# ---------------------------------------------------------------------------
# Corpus:  10 elements, max_node 54, pos_norm 7, spatial_cut_off 0.1276
#          (= 2.5 A; longest of 8437270 RDKit bonds is 2.437 A).
#
# Cost on one GH200, from 150 REAL training steps through main.py:
#          0.279 s/step at batch 128   (3.58 steps/s, 454 mol/s)
#          23.4 GiB peak of 95         (batch 256 also fits at 46.0 GiB)
#
#   150000 steps x 0.279 s = 11.6 h, inside the 16 h requested below.
#   = 19.2M molecule-views = ~104 epochs over the 184157-molecule train split.
#
# The TERS-inherited guess of 2.16 s/step was wrong by 7.7x -- the STM cube is 28x smaller
# than a TERS cube and the conditioning encoder is only 1.49M parameters. That is why this
# job is sized at 150000 steps rather than TERS's 28000, which would run for 2.2 h and see
# the corpus only ~19.5 times.
#
# DISK: an STM checkpoint is 587 MB and /scratch had 13 GiB free. snapshot_freq=25000 gives
# 6 checkpoints = 3.4 GB. Do NOT lower it without re-checking `df -h /scratch` -- at 5000 it
# would write 17.6 GB and fill the filesystem partway through the run.
#
# If cut short, resubmit with the SAME --workdir: run_lib.py:212 restores from
# checkpoints-meta automatically.

#SBATCH --job-name=stm-train
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72             # one GPU's proportional share of 288
#SBATCH --mem=211G                     # one GPU's share; 217G gets clamped
#SBATCH --time=16:00:00                # 11.6 h expected; 38% margin
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
# Mandatory: the module has torch/h5py/PyG but not rdkit, absl, ml_collections, ase, yaml,
# fcd_torch, moses, pulp or myopic_mces. Those live in the overlay.
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

# Checkpoints are ~660 MB each and a run keeps 5+, so ~3.7 GB per run. $HOME is a 15 GB quota
# and was at 13.9 GB when this was written -- a run writing there dies on quota partway
# through, which is how ters-ae lost job 989420. Default to /scratch; override with
# TERS_WORKDIR.
WORKDIR="${TERS_WORKDIR:-/scratch/project_2001912/stm-runs/stm_$(date +%Y%m%d)_${SLURM_JOB_ID}}"
mkdir -p "$(dirname "$WORKDIR")"

echo "=============================================================="
echo "job      : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "node     : $SLURM_JOB_NODELIST"
echo "workdir  : $WORKDIR"
echo "started  : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Fail fast rather than an hour in: inputs present, and enough disk for the checkpoints.
for f in /scratch/project_2001912/stm-data/train.h5 \
         /scratch/project_2001912/stm-graphs/graphs_train.pt; do
    [ -s "$f" ] || { echo "MISSING: $f -- see 'BEFORE the first run' above"; exit 1; }
done

# 6 checkpoints x 587 MB + one 587 MB checkpoints-meta = ~4.1 GB. Refuse to start without
# 6 GB, because running out of disk 8 h in loses the whole run (this is how ters-ae lost
# job 989420, on $HOME quota rather than /scratch).
AVAIL_GB=$(df -BG --output=avail /scratch/project_2001912 | tail -1 | tr -dc '0-9')
echo "scratch free: ${AVAIL_GB} GiB (need ~4.1 for checkpoints)"
[ "$AVAIL_GB" -ge 6 ] || { echo "REFUSING TO START: only ${AVAIL_GB} GiB free on /scratch"; exit 1; }

# Re-submitting with the SAME --workdir resumes rather than restarts.
srun python3 main.py \
    --config configs/diffspectra_stm.py \
    --config_original_qm9 configs/base_stm.py \
    --mode train \
    --workdir "$WORKDIR"

echo "finished : $(date -Is)"
