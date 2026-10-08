"""Run the DiffSpectra paper metric suite on a trained checkpoint.

Samples each cell with sample.py and scores it with
eval_sampled_mols/paper_metrics.py (similarity) and eval_sampled_mols/eval_cond.py (distribution).
Runs matched test samples by default. Select validation and shuffled controls with --cells.
The `*_null` cells pair each target with another molecule's cube, as a chance baseline.

Usage
-----
    python3 eval_sampled_mols/run_metrics.py --ckpt CHECKPOINT --dry-run
    python3 eval_sampled_mols/run_metrics.py --ckpt CHECKPOINT
    python3 eval_sampled_mols/run_metrics.py --ckpt CHECKPOINT --n-mols 3 --n-samples 3 --ks 1 3 --no-mces
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from env import output_root                                             # noqa: E402  (needs the path above)

CONFIGS = {
    'stm': ('configs/diffspectra_stm.py', 'configs/base_stm.py'),
    'ters': ('configs/diffspectra_ters.py', 'configs/base_ters.py'),
}

# (cell name, split, shuffled)
CELLS = [
    ('test', 'test', False),
    ('test_null', 'test', True),
    ('val', 'val', False),
    ('val_null', 'val', True),
]


def sample_cmd(args, split, shuffled, out_dir):
    cfg, cfg_q = CONFIGS[args.modality]
    cmd = [sys.executable, 'sample.py',
           '--modality', args.modality,
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
    cmd = [sys.executable, 'eval_sampled_mols/paper_metrics.py', '--cells'] + cells
    cmd += ['--ks'] + [str(k) for k in args.ks]
    if args.no_mces:
        cmd.append('--no-mces')
    if args.csv:
        cmd += ['--out', args.csv]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--modality', default='stm', choices=sorted(CONFIGS))
    ap.add_argument('--out', default=None, help='default: OUTPUT_ROOT/stm-eval/paper_metrics')
    ap.add_argument('--n-mols', type=int, default=20,
                    help='spectra per cell (default 20 = the TERS protocol)')
    ap.add_argument('--n-samples', type=int, default=10,
                    help='candidates per spectrum (default 10; must be >= max k)')
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 10])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cells', nargs='+', default=['test'],
                    choices=[c[0] for c in CELLS],
                    help='cells to run (default: matched test only). Add val, test_null or val_null explicitly')
    ap.add_argument('--batch-mols', type=int, default=0,
                    help='targets per generation batch (0 = all at once). Needed above ~20 '
                         'targets to stay within GPU memory; forwarded to sample.py --batch-mols.')
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
    args.out = args.out or output_root('stm-eval', 'paper_metrics')

    if max(args.ks) > args.n_samples:
        raise SystemExit(
            f'--n-samples {args.n_samples} is below the largest k ({max(args.ks)}). '
            f'top-{max(args.ks)} would be undefined. Raise --n-samples or lower --ks.')

    wanted = [c for c in CELLS if c[0] in args.cells]
    os.chdir(REPO)                      # every command below uses repo-relative config paths

    # --- 1. sample -----------------------------------------------------------------------
    scored, processed = [], []
    for name, split, shuffled in wanted:
        out_dir = os.path.join(args.out, name)
        # File names must match what sample.py writes.
        pkl = os.path.join(out_dir, f'mols_{split}{"_shuf" if shuffled else ""}.pkl')
        scored.append(f'{args.modality}_{name}={pkl}')
        # Tensor tuples for stage 3.
        processed.append(os.path.join(
            out_dir, f'processed_{split}{"_shuf" if shuffled else ""}.pkl'))

        cmd = sample_cmd(args, split, shuffled, out_dir)
        if args.dry_run:
            print(f'# cell {name}  ({split}, {"shuffled null" if shuffled else "matched"})')
            print('  ' + ' '.join(cmd) + '\n')
            continue
        if args.skip_sampling and os.path.exists(pkl):
            print(f'== cell {name}: reusing {pkl}', flush=True)
            continue
        print(f'\n{"=" * 90}\n== cell {name}  ({split}, '
              f'{"shuffled null" if shuffled else "matched"})\n{"=" * 90}', flush=True)
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(cmd, check=True)

    # --- 2. score: similarity to the target ------------------------------------------------
    cmd = score_cmd(args, scored)
    # One eval_cond call for all cells, so its reference sets are built once.
    eval_c = [sys.executable, 'eval_sampled_mols/eval_cond.py', '--samples'] + processed

    if args.dry_run:
        print('# score selected cells together')
        print('  ' + ' '.join(cmd) + '\n')
        print('# distributional metrics: FCD/SNN/Frag/Scaf/IntDiv + stability, all cells at once')
        print('  ' + ' '.join(eval_c))
        return

    missing = [s.split('=', 1)[1] for s in scored if not os.path.exists(s.split('=', 1)[1])]
    if missing:
        raise SystemExit(f'these cells were not produced: {missing}')

    print(f'\n{"=" * 90}\n== scoring {len(scored)} cells\n{"=" * 90}', flush=True)
    subprocess.run(cmd, check=True)

    # --- 3. score: the sample distribution -------------------------------------------------
    # Non-fatal (check=False): the stage 2 table is already written.
    missing = [p for p in processed if not os.path.exists(p)]
    if missing:
        print(f'\nskipping distributional metrics: no processed pickle at {missing}. '
              f'sample.py writes them next to the mols pickle; '
              f'rerun these cells without --skip-sampling to produce them.')
    else:
        print(f'\n{"=" * 90}\n== distributional metrics, {len(processed)} cells'
              f'\n{"=" * 90}', flush=True)
        subprocess.run(eval_c, check=False)

    print(f'\noutputs under {args.out}')


if __name__ == '__main__':
    main()
