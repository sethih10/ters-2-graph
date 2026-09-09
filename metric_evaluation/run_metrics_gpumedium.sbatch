#!/bin/bash
# DiffSpectra paper-metric evaluation of the STM run, full protocol.
#
#   sbatch metric_evaluation/run_metrics_gpumedium.sbatch
#
# 4 cells x 20 spectra x 10 samples x 250 ancestral steps = 800 generated molecules.
# Sized from a real measurement: 25 molecules took ~2 min inside eval job 1173868, so
# ~16 min/cell and ~65 min total. 2 h requested leaves margin; gputest's 15 min cap cannot
# hold the full protocol, which is why this is gpumedium.
#
# Half the cells are NULLS (--shuffle-cubes) and they are not padding. The sampler is handed
# the true atom count, so a Tanimoto in isolation mostly reports the prior; the
# matched-minus-shuffled gap is the part attributable to the STM measurement. A null that
# scores well above zero means the protocol leaked -- treat it as a bug, not a result.

#SBATCH --job-name=stm-metrics
#SBATCH --account=project_2001912
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=211G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

CKPT=${CKPT:-/scratch/project_2001912/stm-runs/stm_20260908_1134521/checkpoints/checkpoint_3.pth}
OUT=${OUT:-/scratch/project_2001912/stm-eval/paper_metrics}
# Targets per cell. 20 reproduces the TERS protocol exactly; raise it for a real estimate of
# the split average (the test split is 23123 molecules, so 20 is 0.09% of it).
N_MOLS=${N_MOLS:-20}
# Targets per generation batch. 20 (= 200 molecules) is the configuration measured to fit in
# 23.4 GiB of 95; generation is a single forward pass per batch, so this is the knob that
# decides whether a large N_MOLS runs at all rather than OOMing.
BATCH_MOLS=${BATCH_MOLS:-20}

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -s "$CKPT" ] || { echo "MISSING checkpoint: $CKPT"; exit 1; }

# Fail before an hour of sampling if the ILP solver is absent -- MCES is one of the paper's
# seven metrics and a silently blank column is worse than a refusal.
python3 -c "import pulp, myopic_mces" || { echo "pulp/myopic_mces missing from the overlay"; exit 1; }

echo "targets/cell: $N_MOLS   generation batch: $BATCH_MOLS targets"
srun python3 metric_evaluation/run_metrics.py \
    --ckpt "$CKPT" --out "$OUT" \
    --n-mols "$N_MOLS" --n-samples 10 --ks 1 3 5 10 \
    --batch-mols "$BATCH_MOLS" ${EXTRA_ARGS:-} \
    --csv "$OUT/metrics.csv"

echo "=== finished @ $(date -Is) ==="
