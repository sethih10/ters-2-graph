#!/bin/bash
# STM smoke test on gputest: prove the pipeline runs on a real GH200, and MEASURE the two
# numbers configs/diffspectra_stm.py currently marks STM-UNMEASURED -- peak memory at the
# configured batch size, and the real s/step.
#
#   sbatch slurm/smoke_stm_gputest.sbatch
#
# gputest caps at 15 minutes and 2 nodes. ONE GPU is deliberate: the probe reports per-GPU
# peak memory, and that number only means something when one GPU holds the whole batch.
#
# Ordered so the memory numbers survive a timeout: the batch sweep runs BEFORE the timing
# run, because an OOM at batch 128 is the failure that would kill an 18 h job an hour in,
# while a missing s/step only costs a re-submit.
#
# Read the D-10 warning in docs/05-progress-log.md before quoting the probe's mol/s: it
# measures a handful of steps, and losses.py:357 adds a second forward pass on a
# `random() < 0.5` draw, which a 4-step sample cannot average out. Step 3 below exists
# precisely because of that -- it runs the REAL training loop for 150 steps instead.

#SBATCH --job-name=stm-smoke
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=211G                     # per-GPU share is 217086 MB = 211.99 GiB; 217G clamps
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "--- 1. dataset + config sanity -----------------------------------------"
srun python3 - <<'PY'
import os.path as osp, torch, h5py
from configs.diffspectra_stm import get_config
c = get_config()
print('data.root        ', c.data.root)
print('graph_root       ', c.data.graph_root)
print('atom_types       ', c.data.atom_types)
print('max_node         ', c.data.max_node)
print('stm_in_channels  ', c.model.stm_in_channels)
print('normalize_factors', c.model.normalize_factors)
print('spatial_cut_off  ', c.model.spatial_cut_off)
for s in ('train', 'val', 'test'):
    h5 = osp.join(c.data.root, f'{s}.h5')
    pt = osp.join(c.data.graph_root, f'graphs_{s}.pt')
    with h5py.File(h5, 'r') as f:
        n_cube = f[f'{s}/X'].shape[0]
        z = len(f[s].attrs['z_slices'])
    recs = torch.load(pt, weights_only=False)
    worst = max(int(g['h5_index']) for g in recs)
    assert worst < n_cube, f'{s}: sidecar index {worst} >= {n_cube} cubes'
    assert z == c.model.stm_in_channels, f'{s}: {z} z-slices != stm_in_channels'
    print(f'  {s:5s} {len(recs):6d} graphs / {n_cube:6d} cubes   max h5_index {worst}  '
          f'z_slices {z}  OK')
PY

echo
echo "--- 2. batch-size / memory sweep on one GH200 --------------------------"
echo "    (the number that matters: does base_batch_size=128 fit, and with what headroom)"
srun python3 slurm/probe_batch_size.py --config stm --sizes 64,128,256 --steps 4

echo
echo "--- 3. REAL training loop, 150 steps, for an honest s/step -------------"
# main.py with n_iters overridden. This is the same code path the 18 h job runs, so the
# s/step it prints is the one to size --time from. Workdir is throwaway.
SMOKE_WORKDIR="/scratch/project_2001912/stm-runs/smoke_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$SMOKE_WORKDIR")"
srun python3 main.py \
    --config configs/diffspectra_stm.py \
    --config_original_qm9 configs/base_stm.py \
    --mode train \
    --workdir "$SMOKE_WORKDIR" \
    --config.training.n_iters=150 \
    --config.training.snapshot_freq=100000 \
    --config.training.snapshot_freq_for_preemption=100000 \
    --config.training.log_freq=25 \
    --config.training.wandb.enabled=False

echo
echo "--- s/step summary -----------------------------------------------------"
python3 tools/parse_train_log.py "slurm/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null \
    | tail -20 || echo "(parse_train_log unavailable; read the 'training_step' lines above)"

echo
echo "  smoke workdir (delete when done): $SMOKE_WORKDIR"
echo "=== finished @ $(date -Is) ==="
