"""Run the DiffSpectra paper metric suite on a trained checkpoint.

    python3 metric_evaluation/run_metrics.py --dry-run          # print the commands, do nothing
    python3 metric_evaluation/run_metrics.py                    # STM defaults, full protocol
    python3 metric_evaluation/run_metrics.py --n-mols 3 --n-samples 3 --no-mces   # smoke

**This file computes no chemistry.** It is orchestration only: it invokes the sampler once per
cell and then hands the resulting pickles to the scorer. Every metric definition lives upstream
and is imported there, not here -- see `metric.md` for the provenance and for what each number
means.

    per-pair metrics   compute_metrics.py           (upstream DiffSpectra definitions)
    top-k + null       tools/paper_metrics.py       (aggregation upstream cannot express)
    generation         tools/sample_ters.py         (modality-aware: TERS cubes or STM z-stacks)

Why the protocol is what it is
------------------------------
**20 spectra x 10 samples per cell**, matching `docs/experimental.md` 2d exactly. Two reasons,
and neither is arbitrary:

  * **10 samples** is the floor for a top-10 number. The paper's headline is top-10; scoring
    fewer candidates than k makes the metric undefined, and `compute_metrics.py` cannot express
    it at all -- `evaluate_jsonl_predictions` scores ONE candidate per target
    (compute_metrics.py:207).
  * **the same 20/10 TERS used**, because the whole point of running this is to compare against
    the TERS row, and top-k accuracy on 20 targets moves in steps of 0.05. Changing the target
    count changes the granularity and quietly makes the two tables incomparable.

**Half the cells are nulls, and that is the load-bearing half.** The sampler is handed the true
atom count (sample_ters.py:98) and the chemical space is narrow, so a Tanimoto in isolation
reports mostly the prior. `--shuffle-cubes` pairs each target with another molecule's cube (a
derangement); the matched-minus-shuffled gap is the part attributable to the measurement.
A null that scores well above zero means the protocol leaked, not that the model is good.
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from paths import under                                    # noqa: E402  (needs the path above)

# The last intact checkpoint of STM run 1134521. The run died on a Lustre project quota at step
# 88000 of 150000; checkpoints 1/2/3 survived and checkpoints-meta was truncated.
DEFAULT_CKPT = under('stm-runs/stm_20260908_1134521/checkpoints/checkpoint_3.pth')
DEFAULT_OUT = under('stm-eval/paper_metrics')

CONFIGS = {
    'stm': ('configs/diffspectra_stm.py', 'configs/base_stm.py'),
    'ters': ('configs/diffspectra_ters.py', 'configs/base_ters.py'),
}

# (cell name, split, shuffled). Nulls are not optional -- see the module docstring.
CELLS = [
    ('test', 'test', False),
    ('test_null', 'test', True),
    ('val', 'val', False),
    ('val_null', 'val', True),
]


def sample_cmd(args, split, shuffled, out_dir):
    cfg, cfg_q = CONFIGS[args.modality]
    cmd = [sys.executable, 'tools/sample_ters.py',
           '--ckpt', args.ckpt,
           '--config', cfg, '--config-qm9', cfg_q,
           '--split', split,
           '--n-mols', str(args.n_mols),
           '--n-samples', str(args.n_samples),
           '--seed', str(args.seed),
           '--out', out_dir]
    if args.batch_mols:
        cmd += ['--batch-mols', str(args.batch_mols)]
    if args.no_geometry:
        cmd.append('--no-geometry')
    if shuffled:
        cmd.append('--shuffle-cubes')
    return cmd


def score_cmd(args, cells):
    cmd = [sys.executable, 'tools/paper_metrics.py', '--cells'] + cells
    cmd += ['--ks'] + [str(k) for k in args.ks]
    if args.no_mces:
        cmd.append('--no-mces')
    if args.csv:
        cmd += ['--out', args.csv]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--modality', default='stm', choices=sorted(CONFIGS))
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--n-mols', type=int, default=20,
                    help='spectra per cell (default 20 = the TERS protocol)')
    ap.add_argument('--n-samples', type=int, default=10,
                    help='candidates per spectrum (default 10; must be >= max k)')
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 10])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cells', nargs='+', default=[c[0] for c in CELLS],
                    choices=[c[0] for c in CELLS],
                    help='subset of cells to run (default: all four)')
    ap.add_argument('--batch-mols', type=int, default=0,
                    help='targets per generation batch (0 = all at once). Required above ~20 '
                         'targets or generation OOMs -- see sample_ters --batch-mols.')
    ap.add_argument('--no-geometry', action='store_true',
                    help='skip the per-target geometry payload (only the figures use it)')
    ap.add_argument('--no-mces', action='store_true',
                    help='skip the MCES ILP column (much faster; use for smoke runs)')
    ap.add_argument('--csv', default='', help='also write the table to this CSV')
    ap.add_argument('--skip-sampling', action='store_true',
                    help='score pickles that already exist instead of regenerating them')
    ap.add_argument('--dry-run', action='store_true',
                    help='print every command and exit without running or writing anything')
    args = ap.parse_args()

    if max(args.ks) > args.n_samples:
        # Refuse rather than silently report a top-10 computed from 5 candidates.
        raise SystemExit(
            f'--n-samples {args.n_samples} is below the largest k ({max(args.ks)}). '
            f'top-{max(args.ks)} would be undefined. Raise --n-samples or lower --ks.')

    wanted = [c for c in CELLS if c[0] in args.cells]
    os.chdir(REPO)                      # every command below uses repo-relative config paths

    # --- 1. sample -----------------------------------------------------------------------
    scored = []
    for name, split, shuffled in wanted:
        out_dir = os.path.join(args.out, name)
        # sample_ters names its output mols_<split>[_shuf].pkl; each cell gets its own
        # directory so the matched and shuffled runs of the same split cannot collide.
        pkl = os.path.join(out_dir, f'mols_{split}{"_shuf" if shuffled else ""}.pkl')
        scored.append(f'{args.modality}_{name}={pkl}')

        cmd = sample_cmd(args, split, shuffled, out_dir)
        if args.dry_run:
            print(f'# cell {name}  ({split}, {"SHUFFLED null" if shuffled else "matched"})')
            print('  ' + ' '.join(cmd) + '\n')
            continue
        if args.skip_sampling and os.path.exists(pkl):
            print(f'== cell {name}: reusing {pkl}', flush=True)
            continue
        print(f'\n{"=" * 90}\n== cell {name}  ({split}, '
              f'{"SHUFFLED null" if shuffled else "matched"})\n{"=" * 90}', flush=True)
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(cmd, check=True)

    # --- 2. score ------------------------------------------------------------------------
    cmd = score_cmd(args, scored)
    if args.dry_run:
        print('# score all cells together (one table, nulls beside their matched cell)')
        print('  ' + ' '.join(cmd))
        return

    missing = [s.split('=', 1)[1] for s in scored if not os.path.exists(s.split('=', 1)[1])]
    if missing:
        raise SystemExit(f'these cells were not produced: {missing}')

    print(f'\n{"=" * 90}\n== scoring {len(scored)} cells\n{"=" * 90}', flush=True)
    subprocess.run(cmd, check=True)
    print(f'\noutputs under {args.out}')


if __name__ == '__main__':
    main()
