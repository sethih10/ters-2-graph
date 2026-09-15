#!/bin/bash
# The STM test suite (tests/test_stm.py), on a CPU node.
#
#   sbatch slurm/test_stm_cpu.sbatch
#
# Why this file exists: the suite is the preflight that slurm/train_stm_gpumedium.sbatch:21
# and slurm/verify_eval_cond.sbatch:86 run *inside* a GPU job. Running it on its own needed
# either a GPU allocation it has no use for, or the login node -- and tier_config builds
# three DMT models plus a dataset, which is not login-node work. CPU, one node, no GPU.
#
# Why not just `python -m tests.test_stm` at a shell: tier_preprocess imports rdkit, which
# python-pytorch/2.10 does not carry. Without the overlay on PYTHONPATH the suite dies after
# the encoder tier with ModuleNotFoundError and every later check is silently skipped --
# including the ones that exercise config wiring.
#
# --- ROIHU (x86_64) -- same note as slurm/analyze_stm_cpu.sbatch:18: $HOME/.cache/ters-overlay
# is the OLD aarch64 build and its rdkit cannot load here. The x86_64 overlay below is the one.

#SBATCH --job-name=stm-tests
#SBATCH --account=project_2001912
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="/projappl/project_2001912/ters-overlay-x86_64:${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg

echo "=== ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-none}) on $(hostname) @ $(date -Is) ==="
python3 -c "import rdkit, torch, h5py; print('rdkit', rdkit.__version__, '| torch', torch.__version__)"
echo

python3 -m tests.test_stm
