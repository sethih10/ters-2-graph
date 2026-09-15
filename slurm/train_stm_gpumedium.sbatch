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
# 2. Sanity-check the pipeline:  python3 -m tests.test_stm     # 73 checks
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
# If cut short, resubmit with the SAME workdir: run_lib.py:216 restores from
# checkpoints-meta automatically.  export TERS_WORKDIR=<the old workdir> -- without it this
# script mints a NEW dated workdir and silently trains from step 0.
#
# If checkpoints-meta/checkpoint.pth was lost (job 1134521 lost it to a quota-truncated
# torch.save) but checkpoints/checkpoint_N.pth survived, copy one over it first: they are the
# same format, optimizer and EMA included.  The preflight below then reports the resume step.

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
#
# NOT `df`: /scratch is a shared 6 PB Lustre filesystem, so df reports the whole device
# (3.6 PB free on 2026-09-11) and this guard passed unconditionally. That is precisely why it
# did not catch the PROJECT QUOTA that killed job 1134521 at step 88000 -- 250.5 G against a
# 251 G hard limit, while df reported petabytes free. A df check is the wrong check on a
# quota'd Lustre filesystem (docs/05-progress-log.md, 2026-09-09). The binding number is the
# project's own allocation, which csc-workspaces reports.
quota_avail_gb() {
    if command -v csc-workspaces >/dev/null 2>&1; then
        csc-workspaces 2>/dev/null | awk '
            function gb(s,   n, u) {
                n = s + 0; u = toupper(substr(s, length(s)))
                if (u == "T") return n * 1024
                if (u == "M") return n / 1024
                if (u == "K") return n / 1048576
                return n
            }
            $1 == "/scratch/project_2001912" && $2 ~ /\// {
                split($2, q, "/"); printf "%d\n", gb(q[2]) - gb(q[1]); exit
            }'
        return
    fi
    # Fallback: Lustre group accounting. blimit reads 0 here (not Lustre-enforced), so the
    # cap is the known 2 TB allocation rather than anything lfs reports.
    lfs quota -g project_2001912 /scratch 2>/dev/null | awk '
        $1 ~ /scratch/ && $2 ~ /^[0-9]+\*?$/ {
            gsub(/\*/, "", $2); printf "%d\n", 2048 - ($2 / 1048576); exit
        }'
}
AVAIL_GB=$(quota_avail_gb)
if [ -z "$AVAIL_GB" ]; then
    # A parse failure is not evidence of a full filesystem, so warn rather than refuse -- but
    # say so loudly, because it means the run is going in unguarded.
    echo "WARNING: could not read the project quota; starting WITHOUT a disk guard"
else
    echo "scratch quota free: ${AVAIL_GB} GiB (need ~4.1 for checkpoints)"
    [ "$AVAIL_GB" -ge 6 ] || { echo "REFUSING TO START: only ${AVAIL_GB} GiB left of the project quota"; exit 1; }
fi

# Resume assertion. run_lib.py:216 restores from <workdir>/checkpoints-meta/checkpoint.pth and
# starts from step 0 with only a log line if it is missing or unreadable -- so a resubmit meant
# to resume can quietly repeat 16 h of work. Job 1134521 lost this file to a torch.save
# truncated by the quota. Load it here, on the compute node, exactly the way utils.py:22 will
# (weights_only=False included -- see the comment there).
META="$WORKDIR/checkpoints-meta/checkpoint.pth"
if [ -f "$META" ]; then
    python3 -c '
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
missing = [k for k in ("model", "optimizer", "ema", "step") if k not in ck]
if missing:
    sys.exit("RESUME FILE INCOMPLETE: missing %s" % missing)
print("resume file OK -- training will continue from step %d" % int(ck["step"]))
' "$META" || { echo "REFUSING TO START: $META is unreadable or incomplete. Copy a good checkpoints/checkpoint_N.pth over it to resume, or delete it to train from step 0."; exit 1; }
else
    echo "no resume file at $META -- training starts from step 0"
fi

# Re-submitting with the SAME workdir (export TERS_WORKDIR) resumes rather than restarts.
#
# The wandb run id defaults to a hash of the workdir (ters_wandb.py:198) so a preempted job
# rejoins its own run. That is wrong when resuming from a checkpoint OLDER than the furthest
# step the run already logged: wandb drops the replayed steps and the curve stalls. Set
# TERS_WANDB_RUN_ID to give the resumed leg its own clean curve in the same project.
WANDB_ARGS=()
if [ -n "${TERS_WANDB_RUN_ID:-}" ]; then
    WANDB_ARGS+=("--config.training.wandb.run_id=${TERS_WANDB_RUN_ID}")
fi

srun python3 main.py \
    --config configs/diffspectra_stm.py \
    --config_original_qm9 configs/base_stm.py \
    --mode train \
    --workdir "$WORKDIR" \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}

echo "finished : $(date -Is)"
