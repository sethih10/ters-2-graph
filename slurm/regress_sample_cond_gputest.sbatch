#!/bin/bash
# REGRESSION GATE for the sample_ters.py -> sample_cond.py rename. gputest, ~5 min.
#
#   sbatch slurm/regress_sample_cond_gputest.sbatch
#
# Reproduces the recorded 2026-09-09 baseline EXACTLY -- same checkpoint, same split, same
# 5 x 5, same seed -- and diffs the generated SMILES against it molecule by molecule. That
# baseline (/scratch/project_2001912/stm-eval/generated/samples_test.json, job of 2026-09-09)
# was produced by the PRE-RENAME code, so it is a genuine before/after and not a self-check.
#
# What could plausibly have changed generation, and why neither does:
#   * `--modality` only selects argparse DEFAULTS for --config/--config-qm9/--out. Every
#     explicit invocation, run_metrics.py included, passes all three anyway.
#   * `mol_process_2D` -> `mol_process` happens at sample_cond.py:274, AFTER
#     `sampler.sampling()` has returned at :256. It is a tensor->tuple conversion of an
#     already-finished trajectory, so it cannot move a sample.
# Structural arguments are not evidence, hence this job.
#
# EXPECTED DIFFERENCES, which are not regressions:
#   * `stability` goes from the D-11 hardcoded {0.0, 0.0} to a real measurement. The baseline
#     JSON records `mol_stable 0.0` next to `Validity 0.96` on the same molecules -- that zero
#     was an early return (evaluation/stability.py:143), not a score.
#   * a new processed_test.pkl appears, which the baseline run did not write.
# The SMILES themselves must be IDENTICAL. That is the gate.
#
# Partition: gputest is right for this one -- generation only, no reference sets, well inside
# the 15:00 cap (the same 250-step sampling took ~2 min at 3x3 in job 1228484).

#SBATCH --job-name=stm-regress
#SBATCH --account=project_2001912
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -uo pipefail
mkdir -p slurm/logs

module load python-pytorch/2.10
export PYTHONPATH="$HOME/.cache/ters-overlay:$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

RUN=${RUN:-/scratch/project_2001912/stm-runs/stm_20260908_1134521}
CKPT=${CKPT:-$RUN/checkpoints/checkpoint_3.pth}
BASE=${BASE:-/scratch/project_2001912/stm-eval/generated/samples_test.json}
OUT=${OUT:-/scratch/project_2001912/stm-eval/regress_sample_cond}
mkdir -p "$OUT"

echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID) on $SLURM_JOB_NODELIST @ $(date -Is) ==="
echo "baseline: $BASE"
echo "ckpt    : $CKPT"

# The baseline was made by slurm/eval_stm_gputest.sbatch, which passed neither --seed nor
# --modality, so: seed 0 (the default) and the TERS-shaped config defaults overridden
# explicitly, exactly as that script did.
srun python3 -m tools.sample_cond \
    --config configs/diffspectra_stm.py --config-qm9 configs/base_stm.py \
    --ckpt "$CKPT" --split test \
    --n-mols 5 --n-samples 5 --seed 0 \
    --out "$OUT" 2>&1 | tee "$OUT/generation.txt"

echo
echo "################ DIFF vs the pre-rename baseline ################"
python3 - "$BASE" "$OUT/samples_test.json" <<'PY'
import json, sys
base, new = (json.load(open(p)) for p in sys.argv[1:3])

def smiles(d):
    # keyed by ds_index because `results` is sorted by best_tanimoto, and ties can reorder
    return {r['ds_index']: [s['smiles'] for s in r['samples']] for r in d['results']}

b, n = smiles(base), smiles(new)
if set(b) != set(n):
    print(f'FAIL: different targets\n  baseline {sorted(b)}\n  new      {sorted(n)}')
    sys.exit(1)

bad = [i for i in b if b[i] != n[i]]
for i in bad:
    print(f'  target {i}:\n    baseline {b[i]}\n    new      {n[i]}')
print(f'{len(b)} targets, {sum(len(v) for v in b.values())} samples')
print('SMILES IDENTICAL -- rename and mol_process switch are generation-neutral' if not bad
      else f'FAIL: {len(bad)} target(s) differ')

print(f"\nstability  baseline {base['stability']}  (D-11 early return)")
print(f"stability  new      {new['stability']}  (measured)")
print(f"rdkit      baseline {base['rdkit']}")
print(f"rdkit      new      {new['rdkit']}")
sys.exit(1 if bad else 0)
PY
echo "diff exit: $?"

echo "=== outputs ==="; ls -la "$OUT"
echo "=== finished @ $(date -Is) ==="
