"""DiffSpectra's own evaluation metrics, computed on TERS samples, with top-k and a null control.

REUSE : compute_metrics.py -- the upstream scorer. Its metric set IS the paper's
        (Top-1 by InChIKey, MCES, Tanimoto Morgan, Cosine Morgan, Tanimoto MACCS, Fraggle,
        Functional Group), and the primitives are imported from it rather than reimplemented:
        `functional_group_similarity`, `mol2smiles`, `FUNCTIONAL_GROUPS`.
NEW   : the top-k aggregation and the null-control column.

Why this file has to exist at all. `compute_metrics.evaluate_jsonl_predictions` scores exactly
ONE candidate per target -- `pred_mol = pred_mol_list[0]` (compute_metrics.py:207) -- so it
cannot express the paper's headline "top-10 99.49%". We sample 10 candidates per spectrum, so
the aggregation has to be written; the per-pair chemistry is upstream's, unchanged, including
Morgan radius 2 / 2048 bits and MCES(threshold=100, always_stronger_bound=False).

Reported per cell:
  top-k accuracy (k = 1,3,5,10)  -- fraction of targets whose InChIKey appears in the best k
  best-of-k       for each similarity metric (max; min for MCES, where lower is better)
  mean-over-all   for each similarity metric -- what a single random sample is worth

The null column is the point. A Tanimoto means nothing on its own here: the model is handed the
true atom count and the chemical space is narrow, so the prior alone scores well above zero.
`sample_ters --shuffle-cubes` pairs each target with another molecule's cube; the gap between
matched and shuffled is the part attributable to the TERS measurement.
"""
import argparse, os, pickle, sys, warnings
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Chem.Fraggle import FraggleSim
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_metrics import functional_group_similarity, mol2smiles   # REUSE

try:
    import pulp
    from myopic_mces import MCES
    _SOLVER = pulp.listSolvers(onlyAvailable=True)[0]
except Exception as e:                       # MCES is an ILP; keep the rest usable without it
    MCES, _SOLVER = None, None
    print(f'MCES unavailable ({e}); that column will be blank', file=sys.stderr)

SIMS = ['Tanimoto (Morgan)', 'Cosine (Morgan)', 'Tanimoto (MACCS)', 'Fraggle', 'Functional Group']


def pair_metrics(true_mol, pred_mol, with_mces=True):
    """Upstream's per-pair metrics, same parameters (compute_metrics.py:235-267)."""
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


def score_cell(path, ks, with_mces=True):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    nm, ns = d['n_mols'], d['n_samples']
    per_mol = []
    for mi in range(nm):
        cands = [pair_metrics(d['truth'][mi], d['generated'][mi * ns + si], with_mces)
                 for si in range(ns)]
        per_mol.append(cands)
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
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    rows = {}
    for spec in args.cells:
        name, path = spec.split('=', 1)
        print(f'scoring {name} ...', file=sys.stderr)
        rows[name] = score_cell(path, args.ks, with_mces=not args.no_mces)
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
