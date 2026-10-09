"""Merge the outputs of independent `sample.py --n-shards` jobs into one cell.

Per-shard aggregate metrics are not combined; run the scorers on the merged pickles.
"""
import argparse
import glob
import json
import os
import pickle


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
                raise SystemExit(f'{d}: missing {os.path.basename(p)}; the shard may not have finished')
        m, pr = load(mp), load(pp)

        if n_samples is None:
            n_samples = m['n_samples']
        elif m['n_samples'] != n_samples:
            raise SystemExit(f'{d}: n_samples {m["n_samples"]} != {n_samples} from the first '
                             f'shard. These are not shards of one run.')
        # Scorers index candidates as mi * n_samples + si.
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
                   'note': 'per_shard stability/rdkit values are per-shard aggregates, not '
                           'whole-cell metrics; averaging them gives an average of averages. '
                           'Re-run the scorers on the merged pickles.',
                   'per_shard': per_shard, 'results': results}, f, indent=2, default=str)

    print(f'\nmerged {len(truth)} targets x {n_samples} samples = {len(generated)} molecules '
          f'-> {args.out}')
    print(f'score it with:\n  OUT={args.out} sbatch slurm/score_run.sbatch')


if __name__ == '__main__':
    main()
