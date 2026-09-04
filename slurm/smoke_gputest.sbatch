#!/bin/bash
# Smoke test on gputest: prove the pipeline runs on a real GH200 and measure how
# large the batch can go, before burning gpumedium hours on a 36 h job.
#
#   sbatch slurm/smoke_gputest.sbatch
#
# gputest caps at 15 minutes and 2 nodes. ONE GPU is deliberate: the probe reports
# per-GPU peak memory, and that number is only meaningful when one GPU holds the
# whole batch. See slurm/template.sbatch for the hardware/partition facts.

#SBATCH --job-name=ters-smoke
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=211G                     # per-GPU share is 217086 MB = 211.99 GiB; 217G gets clamped
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
mkdir -p slurm/logs

module purge
module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "--- 1. dataset + config sanity (no GPU needed) -------------------------"
srun python3 - <<'PY'
import os.path as osp, torch, h5py
from configs.diffspectra_ters import get_config
c = get_config()
print('data.root        ', c.data.root)
print('ters_key         ', c.data.ters_key)
print('atom_types       ', c.data.atom_types)
print('normalize_factors', c.model.normalize_factors)
print('spatial_cut_off  ', c.model.spatial_cut_off)
for s in ('train', 'val', 'test'):
    h5, pt = osp.join(c.data.root, f'{s}.h5'), osp.join(c.data.graph_root, f'graphs_{s}.pt')
    n_cube = h5py.File(h5, 'r')[c.data.ters_key].shape[0]
    recs = torch.load(pt, weights_only=False)
    worst = max(int(g['h5_index']) for g in recs)
    assert worst < n_cube, f'{s}: sidecar index {worst} >= {n_cube} cubes'
    print(f'  {s:5s} {len(recs):5d} graphs / {n_cube:5d} cubes   max h5_index {worst}  OK')
PY

echo
echo "--- 2. batch-size probe on one GH200 -----------------------------------"
srun python3 slurm/probe_batch_size.py --sizes 16,32,64,128,256 --steps 4

echo
echo "=== finished @ $(date -Is) ==="
