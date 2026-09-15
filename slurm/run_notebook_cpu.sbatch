#!/bin/bash
# Re-execute stm_samples.ipynb in place, so the committed notebook carries fresh outputs.
#
#   sbatch slurm/run_notebook_cpu.sbatch
#   NB=other.ipynb sbatch slurm/run_notebook_cpu.sbatch
#
# Run it AFTER slurm/analyze_stm_cpu.sbatch: the notebook's last section displays the corpus
# figures and reads analysis.json, so executing it first would bake in the previous run's
# numbers (or a "run tools/analyze_stm.py first" message).
#
# On a compute node rather than the login node: this opens the val sidecar (~100 MB), reads
# a few dozen 128x128x14 cubes, and runs a coordinate/image alignment fit over 12 records.
#
# The kernel is the python-pytorch/2.10 module plus the project's x86_64 rdkit overlay --
# see the header of slurm/analyze_stm_cpu.sbatch for why the old aarch64 venv and
# $HOME/.cache/ters-overlay cannot be used on Roihu. The notebook's kernelspec name is
# `python3`, which resolves to whichever python is first on PATH, so loading the module is
# what selects the interpreter.

#SBATCH --job-name=stm-notebook
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NB=${NB:-stm_samples.ipynb}
TIMEOUT=${TIMEOUT:-1800}

echo "=== ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-none}) on $(hostname) @ $(date -Is) ==="
echo "notebook: $NB"
[ -s "$NB" ] || { echo "MISSING $NB"; exit 1; }

# Keep a copy of the pre-run notebook. If execution dies half way, `--inplace` has already
# rewritten the file, and without this the previous outputs are simply gone.
cp "$NB" "/scratch/project_2001912/stm-eval/$(basename "$NB").bak"

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout="$TIMEOUT" \
    "$NB"

python3 - "$NB" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
errs = [(i, o.get('ename'))
        for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'
        for o in c.get('outputs', []) if o.get('output_type') == 'error']
imgs = sum(1 for c in nb['cells'] if c['cell_type'] == 'code'
           for o in c.get('outputs', []) if 'data' in o and 'image/png' in o['data'])
print(f'cells {len(nb["cells"])}   png outputs {imgs}   errors {len(errs)}')
for i, e in errs:
    print(f'  cell {i}: {e}')
raise SystemExit(1 if errs else 0)
PY

echo "=== finished @ $(date -Is) ==="
