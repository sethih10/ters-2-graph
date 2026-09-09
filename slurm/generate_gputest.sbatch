#!/bin/bash
# Verify wandb logging end to end on a compute node, and MEASURE whether that node
# can reach the internet at all -- which is what decides offline vs online mode.
#SBATCH --job-name=ters-gen
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

# Checkpoint under test. Defaults to the E-1 control (the Aug-28 pool_size=1 run) so an
# unset invocation reproduces the numbers already on record; override to point at another
# arm, e.g.  sbatch --export=ALL,TERS_CKPT=/scratch/.../checkpoint_5.pth  docs/experimental.md
CKPT="${TERS_CKPT:-exp/ters_20260828_925597/checkpoints/checkpoint_5.pth}"
echo "ckpt : $CKPT"
# Derive the output dir from the checkpoint. tools/sample_ters.py defaults --out to the
# Aug-28 control's generated/ dir, so running another arm without this SILENTLY OVERWRITES
# the baseline samples the comparison depends on.
OUT="${TERS_OUT:-$(dirname "$(dirname "$CKPT")")/generated}"
mkdir -p "$OUT"
echo "out  : $OUT"


NMOLS="${TERS_NMOLS:-5}"
NSAMP="${TERS_NSAMPLES:-5}"
# TERS_SHUF=1 runs the shuffled-cube NULL CONTROL: same targets and atom counts, another
# molecule's cube. Outputs carry a _shuf suffix, so a null can never overwrite a matched run.
SHUF="${TERS_SHUF:+--shuffle-cubes}"
TAG="${TERS_SHUF:+_shuf}"
echo "n_mols=$NMOLS n_samples=$NSAMP shuffled=${TERS_SHUF:-0}"

for SP in test val; do
  python -m tools.sample_ters --ckpt "$CKPT" --split $SP \
      --n-mols "$NMOLS" --n-samples "$NSAMP" --out "$OUT" $SHUF
done
python -m tools.render_samples --dir "$OUT" --tag "$TAG"
echo "finished : $(date -Is)"
