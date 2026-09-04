#!/bin/bash
# =============================================================================
# Roihu (CSC) GPU batch template  --  copy this file, do not edit it in place.
#
#   sbatch slurm/template.sbatch
#
# Hardware facts below were read off the live cluster with
#   sinfo -p gputest,gpumedium -o "%P %l %D %c %m %G"
#   scontrol show node rg1103
# not from documentation, so they reflect what the scheduler will actually give.
#
#   GH200 node = 288 cores (4 sockets x 72) | 848 GiB RAM | 4x gh200 GPU
#   => one GPU's proportional share is 72 cores and ~217 GiB.
#
#   partition    max walltime   max nodes   notes
#   gputest      15:00          2           for smoke tests; queue is short
#   gpumedium    1-12:00:00     4           36 h; the real training partition
#   gpularge     1-12:00:00     >4          only if you actually scale out
#
# Billing on GPU partitions is per *GPU-hour*, so reserving the proportional
# core share costs nothing extra -- but reserving more than your share of cores
# can leave GPUs on the node unschedulable for everyone else. Keep the ratio.
# =============================================================================

#SBATCH --job-name=ters-job
#SBATCH --account=project_2001912      # mandatory on CSC; job is rejected without it
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # see "Why ntasks=1" below
#SBATCH --gres=gpu:gh200:1             # <GPUs PER NODE>, max 4
#SBATCH --cpus-per-task=72             # 72 x (GPUs per node)
#SBATCH --mem=211G                     # 211G x (GPUs per node). The real per-GPU share is
                                       # 217086 MB = 211.99 GiB, so "217G" gets clamped by slurm
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out  # %x=job-name %j=jobid
#SBATCH --error=slurm/logs/%x-%j.out

# -----------------------------------------------------------------------------
# Why ntasks=1
# -----------------------------------------------------------------------------
# models/utils.py:27 wraps the model in torch.nn.DataParallel -- ONE python
# process that fans a batch out over every visible GPU. It is not DDP, so this
# must NOT be launched with torchrun or with srun --ntasks=N: N ranks would each
# build the full model, each claim all GPUs, and quietly train N redundant
# copies. One task, N GPUs, is correct for this repo.
#
# (config.training.distributed / world_size / local_rank exist but nothing calls
# init_process_group -- they are vestigial. Only local_rank is read, to pick
# config.device. See configs/diffspectra_ters.py:161.)
# -----------------------------------------------------------------------------

set -euo pipefail
mkdir -p slurm/logs

module purge
module load python-pytorch/2.10

# The module ships torch/h5py/torch_geometric but NOT rdkit, absl, ml_collections,
# ase or yaml -- those live in a user overlay. Without this line the job dies on
# `import rdkit` about two seconds in. See docs/05-progress-log.md D-2.
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

echo "=============================================================="
echo "job        : $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "node(s)    : $SLURM_JOB_NODELIST"
echo "gpus       : ${SLURM_GPUS_ON_NODE:-?} x gh200"
echo "cpus/task  : ${SLURM_CPUS_PER_TASK:-?}"
echo "started    : $(date -Is)"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "=============================================================="

# ---- your command goes here -------------------------------------------------
srun python3 main.py \
    --config configs/diffspectra_ters.py \
    --config_original_qm9 configs/base_ters.py \
    --mode train \
    --workdir exp/ters_template
# -----------------------------------------------------------------------------

echo "finished   : $(date -Is)"
