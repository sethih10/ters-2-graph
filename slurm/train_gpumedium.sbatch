#!/bin/bash
# Full TERS training run on gpumedium.  ONE GPU -- deliberately.
#
#   sbatch slurm/train_gpumedium.sbatch
#
# ---------------------------------------------------------------------------
# Why 1 GPU and not 4
# ---------------------------------------------------------------------------
# Molecule-views seen is `base_batch_size * 60000` NO MATTER how many GPUs you
# use: configs/diffspectra_ters.py multiplies batch_size by num_gpus and divides
# n_iters by it. Extra GPUs therefore do not train the model longer or better --
# they only finish the same work sooner, and you pay per GPU-hour.
#
#                       1x gh200          4x gh200 (DataParallel)
#   steps               28000             7000
#   wall clock          ~16.8 h           ~6-7 h
#   billed GPU-hours    ~17               ~25
#
# models/utils.py:27 uses DataParallel: one process that scatters each batch and
# gathers the outputs every step. That realistically returns ~2.5-3x on 4 cards,
# not 4x, so the extra three cost more than they save. 1 GPU is also the exact
# configuration every number here was measured on -- the 4-GPU path has never run.
#
# To go back to 4 GPUs: --gres=gpu:gh200:4, --cpus-per-task=288, --mem=0.
# Nothing in the config needs editing; it reads torch.cuda.device_count().
# Do NOT use torchrun or --ntasks>1 -- DataParallel is not DDP, and N ranks
# would each build the full model and train N redundant copies.
#
# ---------------------------------------------------------------------------
# Measured settings already baked into the config (gputest 923931 / 923974)
# ---------------------------------------------------------------------------
#   data.num_workers = 8    batch fetch 2.70 s -> 0.03 s (~2.2x overall)
#   base_batch_size  = 128  41.7 GiB peak of 95
#   n_iters          = 28000  at a MEASURED 2.16 s/step = 16.8 h (job 925143, 500 steps).
#                             Do not size a run from slurm/probe_batch_size.py's mol/s --
#                             it was 4x optimistic. See D-10 in docs/05-progress-log.md.
#
# Partition facts, read from the live scheduler (see slurm/template.sbatch):
#   gpumedium  max 36 h  |  max 4 nodes  |  4x gh200 per node
#   per GPU share: 72 cores, ~211 GiB host RAM

#SBATCH --job-name=ters-train
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72             # one GPU's proportional share of 288
#SBATCH --mem=211G                     # one GPU's share; 217G gets clamped
#SBATCH --time=18:00:00                # 16.8 h expected. Shorter than
                                       # 36 h so the job queues sooner.
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module purge
module load python-pytorch/2.10
# Mandatory: the module has torch/h5py/PyG but not rdkit, absl, ml_collections,
# ase, yaml, fcd_torch, moses, pulp or myopic_mces. Those live in the overlay.
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

# Checkpoints are ~660 MB each and a run keeps 5+, so ~3.7 GB per run. $HOME is a 15 GB
# quota and was at 14 GB when this default was changed -- a run writing there dies on quota
# partway through, which is how ters-ae lost job 989420. Default to /scratch (250 GB, 180-day
# cleanup); override with TERS_WORKDIR to put a run somewhere else.
WORKDIR="${TERS_WORKDIR:-/scratch/project_2001912/ters-runs/ters_$(date +%Y%m%d)_${SLURM_JOB_ID}}"
mkdir -p "$(dirname "$WORKDIR")"

echo "=============================================================="
echo "job      : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "node     : $SLURM_JOB_NODELIST"
echo "workdir  : $WORKDIR"
echo "started  : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# run_lib.train() restores from checkpoints-meta automatically (run_lib.py:212),
# so re-submitting with the SAME --workdir resumes rather than restarts. If this
# is cut short, reuse the workdir printed above.
srun python3 main.py \
    --config configs/diffspectra_ters.py \
    --config_original_qm9 configs/base_ters.py \
    --mode train \
    --workdir "$WORKDIR"

echo "finished : $(date -Is)"
