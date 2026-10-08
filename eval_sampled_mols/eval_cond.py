"""Distributional and validity metrics for generated TERS/STM samples.

Reads the `processed_*.pkl` files written by `sample.py` and reports Metric-3D, Metric-2D
and predicted-graph geometry validity and MOSES metrics per cell. Metric-3D is not calibrated for STM; use Metric-2D.

Usage
-----
    python -m eval_sampled_mols.eval_cond --samples PROCESSED_PKL
    python -m eval_sampled_mols.eval_cond --samples PKL [PKL ...] --n-jobs 32
    python -m eval_sampled_mols.eval_cond --config-qm9 configs/base_stm.py --ground-truth 500
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import output_root

import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from datasets import get_dataset                                          # noqa: E402
from evaluation import (get_edm_metric, get_2D_edm_metric,                # noqa: E402
                        get_moses_metrics, get_geometry_metrics)

CAVEAT_3D = ('Metric-3D is not calibrated for this corpus: on 500 ground-truth test structures '
             'it reports mol_stable 0.250 (2D reports 0.974 on the same molecules), so its '
             'ceiling here is 0.25, not 1.0. It re-infers bonds from distances with QM9-tuned '
             'margins and a charge-blind valence table. Use Metric-2D, which reads the '
             'predicted bond orders directly (pred_edge=True). Report Metric-3D only together '
             'with this caveat.')


from config_utils import load_config


def _cache_path(cfg_aux_path, config, max_ref):
    key = '|'.join([os.path.abspath(cfg_aux_path),
                    str(getattr(config.data, 'graph_root', config.data.root)),
                    str(config.data.info_name), str(max_ref)])
    return output_root('eval-cache', f'refsets_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl')


def reference_sets(cfg_aux_path, max_ref, use_cache=False):
    """Return ``(train_smiles, test_mols, dataset_info)``, the novelty and MOSES references.

    Splits match ``run_lib.diffspectra_evaluate``. ``cfg_aux_path`` is the graph-only aux config.
    """
    config = load_config(cfg_aux_path)
    cache = _cache_path(cfg_aux_path, config, max_ref)

    _, train_ds, _, test_ds, dataset_info = get_dataset(config)

    if use_cache and os.path.exists(cache):
        print(f'reference sets: reusing {cache}', flush=True)
        with open(cache, 'rb') as f:
            blob = pickle.load(f)
        return blob['train_smiles'], blob['test_mols'], dataset_info

    def collect(ds, what):
        n = len(ds) if max_ref <= 0 else min(len(ds), max_ref)
        t0 = time.time()
        mols = []
        for i in range(n):
            m = ds[i].rdmol
            if m is not None:
                mols.append(m)
        print(f'  {what}: {len(mols)} mols of {len(ds)} in {time.time() - t0:.0f}s', flush=True)
        return mols

    print(f'building reference sets from {cfg_aux_path} '
          f'(max_ref={max_ref if max_ref > 0 else "all"})', flush=True)
    train_mols = collect(train_ds, 'train (novelty)')
    test_mols = collect(test_ds, 'test  (MOSES) ')

    # Release the train Mol objects before any worker pool forks; keeping them exhausts memory.
    from rdkit import Chem
    t0 = time.time()
    train_smiles = [Chem.MolToSmiles(m) for m in train_mols]
    print(f'  train -> {len(train_smiles)} SMILES in {time.time() - t0:.0f}s '
          f'(Mol objects released)', flush=True)
    del train_mols

    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump({'train_smiles': train_smiles, 'test_mols': test_mols,
                         'max_ref': max_ref, 'config': os.path.abspath(cfg_aux_path)}, f)
        print(f'  cached -> {cache}', flush=True)
    return train_smiles, test_mols, dataset_info


def ground_truth_cells(cfg_aux_path, n, split='test', seed=0):
    """Return ``n`` random ground-truth molecules from ``split`` as a calibration cell."""
    import torch
    config = load_config(cfg_aux_path)
    _, train_ds, _, test_ds, _ = get_dataset(config)
    ds = train_ds if split == 'train' else test_ds
    n = len(ds) if n <= 0 else min(n, len(ds))
    sel = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    print(f'building {n} ground-truth molecules from the {split} split', flush=True)

    processed = []
    for i in sel:
        s = ds[i]
        k = int(s.num_atom)
        # Dense (k, k) bond-type matrix, the format the stability metric expects.
        dense = torch.zeros(k, k, dtype=torch.long)
        ei, et = s.edge_index, s.edge_type
        for e in range(ei.shape[1]):
            a, b = int(ei[0, e]), int(ei[1, e])
            if a < k and b < k:
                dense[a, b] = dense[b, a] = int(et[e])
        processed.append((s.pos[:k].float(), s.atom_type[:k].long(),
                          dense, s.fc[:k].long()))
    return {'processed': processed, 'n_mols': len(processed), 'n_samples': 1,
            'split': split, 'shuffled_cubes': False, 'config_qm9': cfg_aux_path,
            'ckpt': f'GROUND-TRUTH-{split.upper()}', 'step': -1}


# Stability runs per chunk in forked workers; set-level metrics need the combined molecules.
_SHARED_PROCESSED = None


def _stability_chunk(job):
    lo, hi, kind, info = job
    chunk = _SHARED_PROCESSED[lo:hi]
    factory = get_edm_metric if kind == '3d' else get_2D_edm_metric
    stab, _, mols = factory(info, None)(chunk)
    n_atoms = sum(len(t[1]) for t in chunk)
    return (round(stab['mol_stable'] * len(chunk)),      # back to counts
            stab['atom_stable'] * n_atoms,
            len(chunk), n_atoms, mols)


def stability_metrics(processed, dataset_info, kind, train_smiles, workers):
    """Return (stability_dict, rdkit_dict, rd_mols); the parallel path matches the serial one."""
    global _SHARED_PROCESSED
    from evaluation.rdkit_metric import eval_rdmol

    factory = get_edm_metric if kind == '3d' else get_2D_edm_metric
    if workers <= 1 or len(processed) < 2 * workers:
        stab, _, mols = factory(dataset_info, None)(processed)
        return stab, eval_rdmol(mols, train_smiles), mols

    import multiprocessing as mp
    n = len(processed)
    # Use several chunks per worker to balance variable molecule costs.
    n_chunks = min(n, workers * 4)
    edges = [round(i * n / n_chunks) for i in range(n_chunks + 1)]
    jobs = [(edges[i], edges[i + 1], kind, dataset_info)
            for i in range(n_chunks) if edges[i] < edges[i + 1]]

    _SHARED_PROCESSED = processed          # inherited by fork; never sent over a pipe
    ctx = mp.get_context('fork')
    t0 = time.time()
    mol_stable = atom_stable = n_mols = n_atoms = 0
    all_mols = []
    with ctx.Pool(workers) as pool:
        for i, (ms, ats, nm, na, mols) in enumerate(pool.imap(_stability_chunk, jobs), 1):
            mol_stable += ms; atom_stable += ats; n_mols += nm; n_atoms += na
            all_mols.extend(mols)
            if i == len(jobs) or i % max(1, len(jobs) // 4) == 0:
                el = time.time() - t0
                print(f'    {kind}: {n_mols}/{n} molecules  {el:.0f}s  '
                      f'eta {el / i * (len(jobs) - i):.0f}s', flush=True)

    stab = {'mol_stable': mol_stable / n_mols, 'atom_stable': atom_stable / n_atoms}
    return stab, eval_rdmol(all_mols, train_smiles), all_mols


def score_cell(name, payload, dataset_info, train_smiles, moses_fn, workers=1):
    """Compute every metric that run_lib.diffspectra_evaluate reports for one cell."""
    processed = payload['processed']
    print(f'\n{"=" * 90}\n== {name}   {len(processed)} molecules   '
          f'{"shuffled null" if payload.get("shuffled_cubes") else "matched"}   '
          f'ckpt step {payload.get("step")}\n{"=" * 90}', flush=True)

    out = {'cell': name, 'n_molecules': len(processed),
           'shuffled_cubes': bool(payload.get('shuffled_cubes')),
           'split': payload.get('split'), 'ckpt': payload.get('ckpt'),
           'step': payload.get('step'),
           'caveat_3d': CAVEAT_3D if dataset_info['name'] == 'STM' else None}

    stab_3d, rdkit_3d, _ = stability_metrics(processed, dataset_info, '3d', train_smiles,
                                             workers)
    print('Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, '
          'complete: %.4f, unique & valid: %.4f, novelty: %.4f'
          % (stab_3d['atom_stable'], stab_3d['mol_stable'], rdkit_3d['Validity'],
             rdkit_3d['Complete'], rdkit_3d['Unique'], rdkit_3d['Novelty']))
    if out['caveat_3d']:
        print(f"   ^ {out['caveat_3d']}")
    out['metric_3d'] = {**stab_3d, **rdkit_3d}

    stab_2d, rdkit_2d, mols_2d = stability_metrics(processed, dataset_info, '2d', train_smiles,
                                                   workers)
    print('Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, '
          'complete: %.4f, unique & valid: %.4f, novelty: %.4f'
          % (stab_2d['atom_stable'], stab_2d['mol_stable'], rdkit_2d['Validity'],
             rdkit_2d['Complete'], rdkit_2d['Unique'], rdkit_2d['Novelty']))
    out['metric_2d'] = {**stab_2d, **rdkit_2d}

    geometry = get_geometry_metrics(processed, dataset_info, workers=workers)
    out['geometry_3d'] = geometry
    print('Geometry-3D || valence: %.4f, bond lengths: %.4f, angle distances: %.4f, '
          'no clashes: %.4f, all checks: %.4f'
          % (geometry['valence_valid'], geometry['bond_lengths_valid'],
             geometry['angle_distances_valid'], geometry['no_clashes'],
             geometry['geometry_valid']))

    if moses_fn is not None:
        moses = moses_fn(mols_2d)
        print('Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f'
              % (moses['FCD'], moses['SNN'], moses['Frag'], moses['Scaf'], moses['IntDiv']))
        out['moses'] = {k: float(v) for k, v in moses.items()}
    else:
        print('MOSES metrics skipped (--no-moses)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', nargs='+', default=[],
                    help='processed_<split>[_shuf].pkl files written by sample.py. '
                         'Pass every cell at once so the reference sets are built once.')
    ap.add_argument('--ground-truth-split', default='test', choices=['test', 'train'],
                    help="which split --ground-truth draws from. 'train' is the null control "
                         "for the distributional metrics: train and test share a distribution "
                         "by construction (one 8:1:1 random split), so training molecules "
                         "scored against the test reference are the baseline for comparing "
                         "generated sets.")
    ap.add_argument('--ground-truth', type=int, default=0, metavar='N',
                    help='also score N ground-truth structures as a calibration cell. '
                         'This is the ceiling Metric-3D can report on this corpus: if it is '
                         'not near 1.0, the QM9-tuned bond margins do not transfer and '
                         'Metric-3D should not be reported. Needs --config-qm9 if --samples '
                         'is empty.')
    ap.add_argument('--config-qm9', default=None,
                    help='aux config for the reference sets (default: the one recorded in the '
                         'sample pickle, i.e. base_stm.py or base_ters.py)')
    ap.add_argument('--max-ref', type=int, default=0,
                    help='cap each reference set at N molecules (0 = all). For smoke runs '
                         'only: a truncated reference changes FCD and inflates novelty, so the '
                         'results are not comparable to a full-reference run. Recorded in the '
                         'output JSON.')
    ap.add_argument('--n-jobs', type=int, default=8,
                    help='MOSES worker processes (run_lib.diffspectra_evaluate uses a fixed 32)')
    ap.add_argument('--workers', type=int, default=0,
                    help='processes for the stability loops (default: os.cpu_count()). Those '
                         'loops are O(n^2) Python per molecule and dominate at 100k+ molecules. '
                         '1 uses the serial path.')
    ap.add_argument('--no-moses', action='store_true',
                    help='skip FCD/SNN/Frag/Scaf/IntDiv -- much faster, for smoke runs')
    ap.add_argument('--cache', action='store_true',
                    help='cache the reference sets to $OUTPUT_ROOT/eval-cache. Off by '
                         'default: the sets take about 34 s to build and 124 MB on disk. '
                         'Useful only when scoring many cells in sequence.')
    ap.add_argument('--out', default='', help='write the JSON here (default: beside --samples[0])')
    args = ap.parse_args()

    if not args.samples and not args.ground_truth:
        raise SystemExit('nothing to score: pass --samples, --ground-truth, or both')
    if args.ground_truth and not args.samples and not args.config_qm9:
        raise SystemExit('--ground-truth without --samples needs --config-qm9')

    cells = []
    for path in args.samples:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if 'processed' not in payload:
            raise SystemExit(
                f'{path} has no "processed" key; it was likely written by an older sampler '
                f'that did not save positions. Re-run sample.py to produce it.')
        name = os.path.basename(os.path.dirname(os.path.abspath(path))) or payload['split']
        cells.append((f'{name}/{os.path.basename(path)}', path, payload))

    # Cells scored against different reference sets are not comparable.
    aux = args.config_qm9 or (cells[0][2].get('config_qm9') if cells else None)
    if aux is None:
        raise SystemExit('no aux config recorded in the sample pickle; pass --config-qm9')
    others = {c[2].get('config_qm9') for c in cells if c[2].get('config_qm9')}
    if args.config_qm9 is None and len(others) > 1:
        raise SystemExit(f'cells disagree on the aux config {sorted(others)}; '
                         f'pass --config-qm9 explicitly if that is intended')

    if args.ground_truth:
        _lbl = ('GROUND-TRUTH train (distributional null)' if args.ground_truth_split == 'train'
                else 'GROUND-TRUTH test (calibration ceiling)')
        cells.insert(0, (_lbl, None,
                         ground_truth_cells(aux, args.ground_truth, args.ground_truth_split)))

    train_smiles, test_mols, dataset_info = reference_sets(aux, args.max_ref, args.cache)

    moses_fn = None
    if not args.no_moses:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'building MOSES reference statistics over {len(test_mols)} test mols '
              f'(n_jobs={args.n_jobs}, device={device})', flush=True)
        t0 = time.time()
        moses_fn = get_moses_metrics(test_mols, n_jobs=args.n_jobs, device=device)
        print(f'  done in {time.time() - t0:.0f}s', flush=True)

    workers = args.workers or (os.cpu_count() or 1)
    print(f'stability loops: {workers} worker process(es)', flush=True)
    results = [score_cell(name, payload, dataset_info, train_smiles, moses_fn, workers)
               for name, _, payload in cells]

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.samples[0])) if args.samples
        else output_root('stm-eval'), 'eval_cond.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'cells': results,
                   'config_qm9': aux,
                   'max_ref': args.max_ref,
                   'reference_truncated': bool(args.max_ref > 0),
                   'n_train_ref': len(train_smiles), 'n_test_ref': len(test_mols),
                   'moses': not args.no_moses,
                   'workers': workers,
                   'caveat_3d': CAVEAT_3D if dataset_info['name'] == 'STM' else None}, f, indent=2, default=str)
    print(f'\nwrote {out_path}')
    if args.max_ref > 0:
        print(f'WARNING: reference sets truncated to {args.max_ref}; FCD and novelty from this '
              f'run are not comparable to a full-reference run.')


if __name__ == '__main__':
    main()
