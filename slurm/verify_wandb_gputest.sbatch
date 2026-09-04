#!/bin/bash
# Verify wandb logging end to end on a compute node, and MEASURE whether that node
# can reach the internet at all -- which is what decides offline vs online mode.
#SBATCH --job-name=ters-wandb
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=211G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
set -uo pipefail          # NOT -e: the connectivity probes are expected to fail
mkdir -p slurm/logs
module purge
module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

echo "########## 1. can this compute node reach the internet? ##########"
echo "--- DNS ---"
getent hosts api.wandb.ai && echo "DNS: resolves" || echo "DNS: FAILS"
echo "--- HTTPS to api.wandb.ai (15s timeout) ---"
curl -sS -o /dev/null -m 15 -w 'HTTP %{http_code} in %{time_total}s\n' \
     https://api.wandb.ai/ && echo "HTTPS: reachable" || echo "HTTPS: UNREACHABLE"
echo "--- proxy vars set? ---"
env | grep -i proxy || echo "(no proxy variables)"

echo
echo "########## 2. module self-test on the compute node ##########"
WANDB_SILENT=true python3 ters_wandb.py --self-test

echo
echo "########## 3. real main.py with wandb enabled ##########"
srun python3 main.py \
    --config configs/diffspectra_ters.py \
    --config_original_qm9 configs/base_ters.py \
    --mode train \
    --workdir "exp/wandbtest_${SLURM_JOB_ID}" \
    --config.training.n_iters=20 \
    --config.training.log_freq=5 \
    --config.training.snapshot_freq=100000 \
    --config.training.snapshot_freq_for_preemption=100000

echo
echo "########## 4. what landed on disk ##########"
find "exp/wandbtest_${SLURM_JOB_ID}/wandb" -maxdepth 2 | head -20
echo "=== finished @ $(date -Is) ==="
