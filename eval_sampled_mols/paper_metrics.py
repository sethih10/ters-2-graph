"""Score generated molecules against their target molecules.

Reports InChIKey top-k recovery, fingerprint and substructure similarities, and MCES. Score
matched and shuffled-cube cells together to measure the effect of the cube.
"""
import argparse, os, pickle, sys, time, warnings
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Chem.Fraggle import FraggleSim
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_metrics import functional_group_similarity, mol2smiles

try:
    import pulp
    from myopic_mces import MCES
    _SOLVER = pulp.listSolvers(onlyAvailable=True)[0]
except Exception as e:                       # MCES is an ILP; keep the rest usable without it
    MCES, _SOLVER = None, None
    print(f'MCES unavailable ({e}); that column will be blank', file=sys.stderr)

SIMS = ['Tanimoto (Morgan)', 'Cosine (Morgan)', 'Tanimoto (MACCS)', 'Fraggle', 'Functional Group']


def pair_metrics(true_mol, pred_mol, with_mces=True):
    """Per-pair similarity metrics for one target and one candidate.

    Adapted from DiffSpectra (compute_metrics.py, evaluate_jsonl_predictions) with the same
    fingerprint and MCES parameters; scores a single pair instead of a JSONL file.
    """
    out = {k: 0.0 for k in SIMS}
    out['MCES'] = float('nan')
    out['exact'] = False
    try:
        pred_mol.UpdatePropertyCache(); Chem.GetSymmSSSR(pred_mol)
        true_smi, pred_smi = mol2smiles(true_mol), mol2smiles(pred_mol)
        if not true_smi or not pred_smi:
            return out
        try:
            out['exact'] = Chem.MolToInchiKey(true_mol) == Chem.MolToInchiKey(pred_mol)
        except Exception:
            pass
        t_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
        p_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
        out['Tanimoto (Morgan)'] = DataStructs.TanimotoSimilarity(t_fp, p_fp)
        out['Cosine (Morgan)'] = DataStructs.CosineSimilarity(t_fp, p_fp)
        out['Tanimoto (MACCS)'] = DataStructs.TanimotoSimilarity(
            MACCSkeys.GenMACCSKeys(true_mol), MACCSkeys.GenMACCSKeys(pred_mol))
        try:
            fr = FraggleSim.GetFraggleSimilarity(true_mol, pred_mol)
            out['Fraggle'] = fr[0] if isinstance(fr, tuple) else fr
        except Exception:
            out['Fraggle'] = 0.0
        out['Functional Group'] = functional_group_similarity(true_mol, pred_mol)
        if with_mces and MCES is not None:
            out['MCES'] = MCES(true_smi, pred_smi, solver=_SOLVER, threshold=100,
                               always_stronger_bound=False, solver_options=dict(msg=0))[1]
    except Exception:
        pass
    return out


_SHARED = None


def _score_one_target_serial(d, mi, ns, with_mces):
    return [pair_metrics(d['truth'][mi], d['generated'][mi * ns + si], with_mces)
            for si in range(ns)]


def _score_one_target(job):
    """Score one target's candidates. Module-level (not a closure) so Pool can pickle it."""
    mi, ns, with_mces = job
    d = _SHARED
    return [pair_metrics(d['truth'][mi], d['generated'][mi * ns + si], with_mces)
            for si in range(ns)]


def score_cell(path, ks, with_mces=True, workers=1):
    global _SHARED
    with open(path, 'rb') as f:
        d = pickle.load(f)
    nm, ns = d['n_mols'], d['n_samples']
    jobs = [(mi, ns, with_mces) for mi in range(nm)]

    if workers > 1 and nm > 1:
        import multiprocessing as mp
        _SHARED = d                       # inherited by fork; never sent over a pipe
        ctx = mp.get_context('fork')
        chunk = max(1, min(64, nm // (workers * 8) or 1))
        t0 = time.time()
        per_mol = []
        with ctx.Pool(workers) as pool:
            for i, cands in enumerate(pool.imap(_score_one_target, jobs, chunksize=chunk), 1):
                per_mol.append(cands)
                if i % 2000 == 0 or i == nm:
                    el = time.time() - t0
                    print(f'    {i}/{nm} targets  {el:.0f}s elapsed  '
                          f'eta {el / i * (nm - i):.0f}s', file=sys.stderr, flush=True)
    else:
        per_mol = [_score_one_target_serial(d, mi, ns, with_mces) for mi in range(nm)]

    n = len(per_mol)
    row = {'n_targets': n, 'n_samples': ns}
    for k in ks:
        if k <= ns:
            row[f'top-{k}'] = sum(any(c['exact'] for c in cs[:k]) for cs in per_mol) / n
    for m in SIMS:
        row[f'{m} (best)'] = sum(max(c[m] for c in cs) for cs in per_mol) / n
        row[f'{m} (mean)'] = sum(sum(c[m] for c in cs) / ns for cs in per_mol) / n
    if with_mces and MCES is not None:
        valid = [[c['MCES'] for c in cs if c['MCES'] == c['MCES']] for cs in per_mol]
        valid = [v for v in valid if v]
        if valid:
            row['MCES (best)'] = sum(min(v) for v in valid) / len(valid)
            row['MCES (mean)'] = sum(sum(v) / len(v) for v in valid) / len(valid)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', required=True,
                    help='NAME=/path/to/mols_<split>[_shuf].pkl, repeatable')
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 10])
    ap.add_argument('--no-mces', action='store_true', help='skip the ILP metric (much faster)')
    ap.add_argument('--workers', type=int, default=0,
                    help='parallel processes over targets (0 = os.cpu_count(), 1 = serial). '
                         'Results do not depend on this: the per-pair metrics are '
                         'deterministic and Pool.imap preserves order.')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    workers = args.workers or (os.cpu_count() or 1)
    print(f'scoring with {workers} worker process(es)', file=sys.stderr)

    rows = {}
    for spec in args.cells:
        name, path = spec.split('=', 1)
        print(f'scoring {name} ...', file=sys.stderr)
        rows[name] = score_cell(path, args.ks, with_mces=not args.no_mces, workers=workers)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200, 'display.max_rows', 100)
    print('\n' + '=' * 96)
    print('DiffSpectra metrics (compute_metrics.py definitions) -- higher better except MCES')
    print('=' * 96)
    print(df.to_string(float_format=lambda v: f'{v:.4f}'))
    if args.out:
        df.to_csv(args.out)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
