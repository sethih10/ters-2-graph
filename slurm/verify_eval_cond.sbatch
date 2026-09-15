#!/bin/bash
# Verify the tools/ evaluation path end to end on a GPU node, and CALIBRATE Metric-3D.
#
#   sbatch slurm/verify_eval_cond.sbatch
#   NGT=2000 sbatch slurm/verify_eval_cond.sbatch          # bigger calibration cell
#
# Three stages, cheapest first, each gating the next. Ordered so that a timeout still leaves
# the earlier answers on disk.
#
#   1. tests/test_stm.py   the synthetic tier, including the two metric regressions this
#                          change fixed (D-8's raise, D-11's fake zero). Seconds.
#   2. sample_cond         generation, small. This is the REGRESSION GATE on the rename and
#                          on the mol_process switch: the SMILES must match a pre-change run
#                          at the same --seed, and processed_test.pkl must now be written.
#   3. eval_cond           the calibration cell plus the generated cell.
#
# WHY A GPU, since this is "just scoring": the stability checks are RDKit and the reference
# walk is CPU, but MOSES is not. `device` is threaded into compute_intermediate_statistics
# and internal_diversity (evaluation/mose_metric.py:69-70, :114) -- FCD embeds every molecule
# through ChemNet, a torch network, over the 23123-molecule reference set, and IntDiv is a
# pairwise-Tanimoto matrix op. Stage 2 needs a GPU outright: 250 sequential denoising steps.
#
# WHAT THE CALIBRATION CELL FOUND -- read this before quoting any Metric-3D number.
# `--ground-truth N` scores the corpus's OWN test structures, which a perfect model would
# reproduce exactly, so their score is the CEILING the metric can report here. Run 2026-09-10
# (job 1228484, 500 structures):
#
#     Metric-3D   mol_stable 0.250   atom_stable 0.823      <- the ceiling, NOT 1.0
#     Metric-2D   mol_stable 0.974   atom_stable 0.999
#
# The falsifier FIRED. Metric-3D calls three quarters of true structures unstable, so on this
# corpus it reports the bond tables rather than the model -- its margins were tuned on QM9
# (H/C/N/O/F only, bond_analyze.py:119-120) and `check_stability` scores against the
# charge-blind `allowed_bonds`. Metric-2D lands where a correct metric should. Re-run this on
# any new corpus before trusting Metric-3D there.
#
# COST -- MEASURED, and the earlier estimate in this header was wrong by ~10x:
#     stage 1 tests            ~60 s
#     stage 2 generation 3x3   ~2 min   (250 sequential denoising steps)
#     reference sets           ~34 s    (23 s train walk + 3 s test + 8 s MOSES/ChemNet)
#     whole job                ~6.5 min
# Which is why the partition is `gputest`: 15:00 is ample, it queues fast, and this job has no
# business sitting in gpumedium. `eval_cond --cache` exists but is OFF by default -- 124 MB of
# a near-quota /scratch to save 34 s is not a trade worth making.

#SBATCH --job-name=stm-verify-eval
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G          # gputest overrides this to the per-GPU share (217086M)
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail          # NOT -e: stage 3 must still run if stage 2 fails
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

# MOSES workers are separate processes; letting BLAS/OMP also thread inside them
# oversubscribes the node and makes it slower, not faster (same as score_cpu.sbatch).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RUN=${RUN:-/scratch/project_2001912/stm-runs/stm_20260908_1134521}
CKPT=${CKPT:-$RUN/checkpoints/checkpoint_3.pth}
OUT=${OUT:-/scratch/project_2001912/stm-eval/verify_eval_cond}
NGT=${NGT:-500}                      # ground-truth molecules in the calibration cell
NMOLS=${NMOLS:-3}
NSAMP=${NSAMP:-3}
SEED=${SEED:-0}
mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ckpt        : $CKPT"
echo "out         : $OUT"
echo "calibration : $NGT ground-truth test structures"

echo
echo "############### 1. synthetic tier (incl. the D-8 and D-11 regressions) ###############"
srun python3 -m tests.test_stm 2>&1 | tee "$OUT/tests.txt"

echo
echo "############### 2. generation -- REGRESSION GATE on the rename + mol_process #########"
echo "### the SMILES below must match a pre-change sample_ters.py run at --seed $SEED ###"
if [ -s "$CKPT" ]; then
  srun python3 -m tools.sample_cond --modality stm \
      --ckpt "$CKPT" --split test \
      --n-mols "$NMOLS" --n-samples "$NSAMP" --seed "$SEED" \
      --no-geometry --out "$OUT/test" 2>&1 | tee "$OUT/generation.txt"
else
  echo "SKIPPED: no checkpoint at $CKPT"
fi

echo
echo "############### 3. distributional metrics + Metric-3D calibration ####################"
PKL="$OUT/test/processed_test.pkl"
srun python3 tools/eval_cond.py \
    --config-qm9 configs/base_stm.py \
    --ground-truth "$NGT" \
    ${PKL:+$([ -s "$PKL" ] && echo "--samples $PKL")} \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --out "$OUT/eval_cond.json" 2>&1 | tee "$OUT/eval_cond.txt"

echo
echo "=== outputs ==="; ls -laR "$OUT" 2>/dev/null | head -40
echo "=== quota ==="; lfs quota -h -p 602001912 /scratch 2>/dev/null | tail -1
echo "=== finished @ $(date -Is) ==="
