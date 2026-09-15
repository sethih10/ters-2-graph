"""Merge sharded `tools/sample_cond.py --n-shards` output into one scorable cell.

    python -m tools.merge_shards --shards $OUT/shard* --out $OUT/merged

Why sharding exists at all: MEASURED on gputest job 1229202, DataParallel across 4 GPUs is
**4.3x slower** than a single GPU (0.9 vs 3.9 mol/s), because models/utils.py:27 wraps the model
in torch.nn.DataParallel, which re-replicates a 587 MB module to every device on every forward
call -- and sampling makes 250 sequential forward calls per chunk. Independent single-GPU
processes over disjoint target shards replicate nothing.

**This file computes no chemistry and re-derives no metric.** It concatenates, then hands the
result to the same scorers a single-process run would use:

    tools/paper_metrics.py   reads mols_<split>.pkl
    tools/eval_cond.py       reads processed_<split>.pkl

What is NOT simply concatenated, and why
----------------------------------------
`samples_<split>.json` carries per-run aggregates -- `stability` and `rdkit` -- computed by
`get_2D_edm_metric` over that shard's molecules only. Averaging those across shards would be an
average of averages and wrong whenever shards differ in size, which the final shard always does.
So they are NOT merged: the merged JSON records them per shard under `per_shard`, and the
authoritative whole-cell numbers come from re-running the scorers on the merged pickles, which
is what metric_evaluation/score_run.sbatch does anyway.

The per-target `results` list IS concatenated, because each entry is self-contained (it carries
its own ds_index and its own candidates). It is re-sorted by best_tanimoto so the merged file
reads like a single run.

Ordering invariant, which everything downstream depends on
----------------------------------------------------------
`paper_metrics.py` indexes candidates as `generated[mi * n_samples + si]`, so the flat mol list
and the target list must stay in lockstep. Shards are merged in the order given, and each
shard's `truth`/`generated` are appended together, so target `mi` of shard 1 lands at
`len(shard0_truth) + mi` in both lists. Verified by an explicit check below rather than trusted.
"""
import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', nargs='+', required=True,
                    help='shard output directories (globs are fine, e.g. $OUT/shard*)')
    ap.add_argument('--out', required=True, help='directory for the merged cell')
    ap.add_argument('--split', default='test')
    ap.add_argument('--tag', default='', help="'_shuf' for a null-control run")
    args = ap.parse_args()

    # isdir filter: the natural glob for shard output is `$OUT/shard*`, which ALSO matches the
    # per-shard `shard0.log` files the sbatch writes beside them (caught by the smoke run of
    # job 1229349, which reported "merging 8 shards" for a 4-shard job). Filtering here rather
    # than only in the caller, because every future caller would otherwise have to know.
    matched = sorted({d for pat in args.shards for d in glob.glob(pat)})
    dirs = [d for d in matched if os.path.isdir(d)]
    skipped = [d for d in matched if not os.path.isdir(d)]
    if skipped:
        print(f'ignoring {len(skipped)} non-directory match(es): '
              f'{", ".join(os.path.basename(x) for x in skipped[:4])}'
              f'{" ..." if len(skipped) > 4 else ""}')
    if not dirs:
        raise SystemExit(f'no directories matched {args.shards}')
    os.makedirs(args.out, exist_ok=True)
    print(f'merging {len(dirs)} shards:')

    truth, generated, processed, results, per_shard = [], [], [], [], []
    n_samples = None
    for d in dirs:
        mp = os.path.join(d, f'mols_{args.split}{args.tag}.pkl')
        pp = os.path.join(d, f'processed_{args.split}{args.tag}.pkl')
        jp = os.path.join(d, f'samples_{args.split}{args.tag}.json')
        for p in (mp, pp):
            if not os.path.exists(p):
                raise SystemExit(f'{d}: missing {os.path.basename(p)} -- did that shard finish?')
        m, pr = load(mp), load(pp)

        if n_samples is None:
            n_samples = m['n_samples']
        elif m['n_samples'] != n_samples:
            raise SystemExit(f'{d}: n_samples {m["n_samples"]} != {n_samples} from the first '
                             f'shard. These are not shards of one run.')
        # The flat candidate list must be exactly n_samples per target, or the
        # `mi * n_samples + si` indexing every scorer uses silently misaligns at the seam.
        if len(m['generated']) != len(m['truth']) * n_samples:
            raise SystemExit(f'{d}: {len(m["generated"])} candidates for {len(m["truth"])} '
                             f'targets at n_samples={n_samples} -- shard is inconsistent.')

        offset = len(truth)
        for e in pr.get('processed', []):
            processed.append(e)
        for e in m['results']:
            e = dict(e)
            e['mol'] = e['mol'] + offset       # reindex into the merged flat list
            e['shard'] = d
            results.append(e)
        truth.extend(m['truth'])
        generated.extend(m['generated'])

        js = json.load(open(jp)) if os.path.exists(jp) else {}
        per_shard.append({'dir': d, 'n_targets': len(m['truth']),
                          'stability': js.get('stability'), 'rdkit': js.get('rdkit')})
        print(f'  {d}: {len(m["truth"])} targets, {len(m["generated"])} candidates, '
              f'{len(pr.get("processed", []))} processed')

    # Invariants, checked rather than assumed.
    assert len(generated) == len(truth) * n_samples, 'merged candidate count is inconsistent'
    if len(processed) != len(generated):
        print(f'WARNING: {len(processed)} processed tuples vs {len(generated)} candidates -- '
              f'the distributional metrics and the similarity metrics will cover different sets')
    ds_idx = [e['ds_index'] for e in results]
    if len(set(ds_idx)) != len(ds_idx):
        raise SystemExit('duplicate ds_index across shards -- these are not disjoint shards')

    results.sort(key=lambda r: -r['best_tanimoto'])
    first = load(os.path.join(dirs[0], f'processed_{args.split}{args.tag}.pkl'))

    with open(os.path.join(args.out, f'mols_{args.split}{args.tag}.pkl'), 'wb') as f:
        pickle.dump({'truth': truth, 'generated': generated,
                     'n_mols': len(truth), 'n_samples': n_samples,
                     'order': [e['ds_index'] for e in results], 'results': results}, f)
    with open(os.path.join(args.out, f'processed_{args.split}{args.tag}.pkl'), 'wb') as f:
        blob = {k: v for k, v in first.items() if k != 'processed'}
        blob.update({'processed': processed, 'n_mols': len(truth), 'n_samples': n_samples,
                     'merged_from': dirs})
        pickle.dump(blob, f)
    with open(os.path.join(args.out, f'samples_{args.split}{args.tag}.json'), 'w') as f:
        json.dump({'split': args.split, 'merged_from': dirs, 'n_targets': len(truth),
                   'n_samples': n_samples,
                   'note': 'per-shard stability/rdkit below are NOT whole-cell numbers -- '
                           'they are per-shard aggregates and averaging them would be an '
                           'average of averages. Re-run the scorers on the merged pickles.',
                   'per_shard': per_shard, 'results': results}, f, indent=2, default=str)

    print(f'\nmerged {len(truth)} targets x {n_samples} samples = {len(generated)} molecules '
          f'-> {args.out}')
    print(f'score it with:\n  OUT={args.out} sbatch metric_evaluation/score_run.sbatch')


if __name__ == '__main__':
    main()
