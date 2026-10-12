"""Rescore generated samples on heavy atoms (explicit hydrogens removed).

Reports Morgan Tanimoto, formula match and canonical SMILES match per sample, and writes
`rescored<tag>.json` and `summary<tag>.pkl` (the latter with SVG depictions).
"""
import argparse, base64, io, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')


def heavy(mol):
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return Chem.RemoveHs(m)
    except Exception:
        return None


def fp(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def svg(mol, size=(260, 210), label=''):
    if mol is None:
        return '<div class="bad">unparseable</div>'
    m = Chem.Mol(mol)
    try:
        Chem.rdDepictor.Compute2DCoords(m)
    except Exception:
        return '<div class="bad">no coords</div>'
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    d.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, legend=label)
    d.FinishDrawing()
    return d.GetDrawingText()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='a <workdir>/generated directory')
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    ap.add_argument('--tag', default='',
                    help="Suffix of the sample files to score, e.g. '_shuf' for the "
                         "shuffled-cube null control written by sample.py --shuffle-cubes. "
                         "Outputs get the same suffix, so a null-control run does not overwrite "
                         "a matched run.")
    args = ap.parse_args()

    summary = {}
    for split in args.splits:
        with open(os.path.join(args.dir, f'mols_{split}{args.tag}.pkl'), 'rb') as f:
            d = pickle.load(f)
        n_mols, n_samp = d['n_mols'], d['n_samples']
        rows = []
        for e in d['results']:
            mi = e['mol']
            gt = heavy(d['truth'][mi])
            gt_smi = Chem.MolToSmiles(gt) if gt else None
            gt_fp, gt_formula = fp(gt), (CalcMolFormula(gt) if gt else None)
            samples = []
            for si in range(n_samp):
                g = heavy(d['generated'][mi * n_samp + si])
                smi = Chem.MolToSmiles(g) if g else None
                f_ = fp(g)
                tan = float(DataStructs.TanimotoSimilarity(gt_fp, f_)) if (f_ and gt_fp) else 0.0
                formula = CalcMolFormula(g) if g else None
                samples.append({'i': si, 'smiles': smi, 'tanimoto': tan, 'formula': formula,
                                'valid': g is not None,
                                'formula_match': bool(formula and formula == gt_formula),
                                'exact': bool(smi and smi == gt_smi),
                                'svg': svg(g, label=f'#{si}  Tan {tan:.2f}')})
            rows.append({'mol': mi, 'ds_index': e['ds_index'], 'n_atoms': e['n_atoms'],
                         'truth_smiles': gt_smi, 'truth_formula': gt_formula,
                         'truth_svg': svg(gt, size=(300, 240), label='ground truth'),
                         'samples': samples,
                         'best': max(s['tanimoto'] for s in samples),
                         'mean': float(np.mean([s['tanimoto'] for s in samples])),
                         'n_valid': sum(s['valid'] for s in samples),
                         'n_formula': sum(s['formula_match'] for s in samples),
                         'n_exact': sum(s['exact'] for s in samples)})
        rows.sort(key=lambda r: -r['best'])
        summary[split] = rows

        print(f'\n{"="*94}\n{split.upper()} -- rescored on heavy atoms (explicit H removed)\n{"="*94}')
        print(f"{'idx':>6}{'atoms':>7}{'valid':>8}{'formula':>9}{'exact':>7}{'best Tan':>10}{'mean Tan':>10}  formula")
        for r in rows:
            print(f"{r['ds_index']:>6}{r['n_atoms']:>7}{r['n_valid']:>5}/{len(r['samples'])}"
                  f"{r['n_formula']:>6}/{len(r['samples'])}{r['n_exact']:>4}/{len(r['samples'])}"
                  f"{r['best']:>10.3f}{r['mean']:>10.3f}  {r['truth_formula']}")
        allt = [s['tanimoto'] for r in rows for s in r['samples']]
        print(f"  overall: mean Tan {np.mean(allt):.3f}, max {np.max(allt):.3f}, "
              f"formula matches {sum(r['n_formula'] for r in rows)}/{len(allt)}, "
              f"exact {sum(r['n_exact'] for r in rows)}/{len(allt)}")

    with open(os.path.join(args.dir, f'rescored{args.tag}.json'), 'w') as f:
        json.dump({k: [{kk: vv for kk, vv in r.items() if kk not in ('truth_svg',)
                        and kk != 'samples'} | {'samples': [{k2: v2 for k2, v2 in s.items()
                                                             if k2 != 'svg'} for s in r['samples']]}
                       for r in v] for k, v in summary.items()}, f, indent=2)
    with open(os.path.join(args.dir, f'summary{args.tag}.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    print(f'\nwrote {args.dir}/rescored{args.tag}.json and summary{args.tag}.pkl')


if __name__ == '__main__':
    main()
