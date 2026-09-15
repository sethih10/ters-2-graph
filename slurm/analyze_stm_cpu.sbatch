#!/bin/bash
# Corpus analysis of the STM dataset. CPU only -- there is no model here.
#
#   sbatch slurm/analyze_stm_cpu.sbatch                      # all three splits
#   LIMIT=4000 SPLITS="val test" sbatch slurm/analyze_stm_cpu.sbatch     # smoke
#
# What it runs: tools/analyze_stm.py, which walks every record in the graph sidecars
# (composition, bonds, rings, SMARTERS planarity, SMILES), samples the HDF5 cubes for the
# intensity envelope, and writes analysis.json + CSVs + figures.
#
# Why CPU and why one node: the work is an RDKit loop over 230k molecules plus a few
# hundred HDF5 cube reads. No tensor maths, no GPU, and the loop is serial per record --
# so this is a single-core job that mostly needs RAM and time, not cores. Asking for a GPU
# would hold a GH200 idle for the whole run, which is the mistake score_cpu.sbatch was
# written to stop repeating.
#
# --- ROIHU (x86_64) -----------------------------------------------------------------
# This project moved from an aarch64 cluster. Two things do NOT carry over:
#   * $HOME/.cache/ters-overlay is an aarch64 build -- its rdkit fails with
#     "rdBase.so: cannot open shared object file". The x86_64 overlay below replaces it.
#   * /scratch/project_2001912/venv/im2g is an arm64 container and cannot start here at all
#     ("the image's architecture (arm64) could not run on the host's (amd64)").
# The python-pytorch/2.10 module supplies numpy/torch/h5py/matplotlib/pandas/torch_geometric;
# only rdkit is missing, which is all the overlay carries. numpy was deliberately deleted
# from the overlay so the module's own numpy 2.4.4 wins -- the pip-resolved numpy 2.5.3
# would otherwise shadow it and break numba.

#SBATCH --job-name=stm-analyze
#SBATCH --account=project_2001912
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="/projappl/project_2001912/ters-overlay-x86_64:${SLURM_SUBMIT_DIR:-$PWD}"

# Each record is processed serially; letting BLAS thread inside that loop only
# oversubscribes the node. Same reasoning as metric_evaluation/score_cpu.sbatch.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

SPLITS=${SPLITS:-"train val test"}
CUBES=${CUBES:-192}
OUT=${OUT:-/scratch/project_2001912/stm-eval/analysis}
EXTRA=${EXTRA:-}
[ -n "${LIMIT:-}" ] && EXTRA="$EXTRA --limit $LIMIT"

echo "=== ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-none}) on $(hostname) @ $(date -Is) ==="
echo "splits='$SPLITS'  cubes=$CUBES  out=$OUT  extra='$EXTRA'"
python3 -c "import rdkit, numpy, torch, h5py, matplotlib; print('rdkit', rdkit.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"

# Fail before the 40-minute walk if the inputs are not where we think they are.
for s in $SPLITS; do
    [ -s "/scratch/project_2001912/stm-graphs/graphs_${s}.pt" ] \
        || { echo "MISSING graphs_${s}.pt"; exit 1; }
done

# The self-tests need no data and take seconds; running them here means a job that fails
# does so in the first 10 seconds with a named check, not 40 minutes in.
python3 tools/planarity.py --self-test
python3 tools/analyze_stm.py --self-test

# Use srun only when this script IS the batch script. If it was itself launched with srun
# (`srun ... bash slurm/analyze_stm_cpu.sbatch`) we are already inside a job step, and a
# nested srun cannot create another one -- it retries "Requested nodes are busy" until the
# job hits its time limit, which looks like a hang rather than a misuse.
LAUNCH=""
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_STEP_ID:-}${SLURM_STEPID:-}" ]; then
    LAUNCH="srun"
fi

$LAUNCH python3 -u tools/analyze_stm.py \
    --splits $SPLITS \
    --cube-samples "$CUBES" \
    --out-dir "$OUT" \
    $EXTRA

echo "=== finished @ $(date -Is) ==="
ls -la "$OUT"
