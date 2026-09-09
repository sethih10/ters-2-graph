"""Draw the generated molecules to PNG files on disk.

REUSE : rdkit.Chem.Draw.MolsToGridImage -- the same primitive visualize.py:29 uses.
ADAPT : visualize.py:7-32. That helper lays molecules out as an undifferentiated grid
        capped at config.sampling.vis_row * vis_col, with no legends, so it cannot show
        which sample came from which spectrum or how good any of them is. Here each ROW is
        one spectrum: column 0 is the ground truth, columns 1-5 are that spectrum's five
        samples, and the legend carries the score. Rows run best to worst.
        (Two bugs in that helper, left alone since this file does not modify it: `Chem` is
        never imported, so check_valid=True raises NameError on a misspelled
        `Chem.SantizeMol`; and the bare `except: pass` at line 31 hides every failure.)
NEW   : the truth/sample pairing and the legend strings.

Writes into <dir>/figures/ as PNG, with SVG alongside as a vector fallback.
"""
import argparse, os, pickle

# --- ADDED (stm): output paths resolve to /scratch, never inside the repo ($HOME is a
# 15 GB quota and one run is 3.7 GB). Both import forms, because these are run BOTH as
# `python3 tools/x.py` (sibling import) and as `python3 -m tools.x` (package import).
try:
    from paths import generated
except ImportError:
    from tools.paths import generated
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')


def prep(mol):
    """Heavy-atom depiction with 2D coords, matching how the scores were computed."""
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        m = Chem.RemoveHs(m)
        AllChem.Compute2DCoords(m)
        return m
    except Exception:
        return None


def grid_for_split(rows, path_png, path_svg, sub=(380, 300)):
    mols, legends = [], []
    for rank, r in enumerate(rows):
        t = prep(r['truth'])
        mols.append(t)
        legends.append(f"#{r['ds_index']}  TRUTH  {r['truth_formula']}")
        for s in r['samples']:
            g = prep(s['mol'])
            mols.append(g)
            if g is None:
                legends.append(f"sample {s['i']}  unparseable")
            else:
                tag = ' *EXACT*' if s['exact'] else (' [formula]' if s['formula_match'] else '')
                legends.append(f"sample {s['i']}  Tan {s['tanimoto']:.2f}{tag}")
    ncol = 1 + len(rows[0]['samples'])

    # REUSE: identical call shape to visualize.py:29, plus legends and an explicit column count.
    img = Draw.MolsToGridImage(mols, molsPerRow=ncol, subImgSize=sub,
                               legends=legends, useSVG=False, returnPNG=False)
    with open(path_png, 'wb') as f:
        if hasattr(img, 'save'):
            img.save(f)                      # PIL image
        else:
            f.write(img.data if hasattr(img, 'data') else img)
    svg = Draw.MolsToGridImage(mols, molsPerRow=ncol, subImgSize=sub,
                               legends=legends, useSVG=True)
    with open(path_svg, 'w') as f:
        f.write(svg.data if hasattr(svg, 'data') else svg)
    return len(mols), ncol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=generated('ters-runs/ters_20260828_925597'))
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    args = ap.parse_args()
    figdir = os.path.join(args.dir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    for split in args.splits:
        with open(os.path.join(args.dir, f'mols_{split}.pkl'), 'rb') as f:
            d = pickle.load(f)
        n_samp = d['n_samples']
        rows = []
        for e in d['results']:                     # already sorted best -> worst
            mi = e['mol']
            gt = prep(d['truth'][mi])
            samples = []
            for si in range(n_samp):
                g = d['generated'][mi * n_samp + si]
                gp = prep(g)
                gt_smi = Chem.MolToSmiles(gt) if gt else None
                smi = Chem.MolToSmiles(gp) if gp else None
                tan = 0.0
                if gp is not None and gt is not None:
                    fa = AllChem.GetMorganFingerprintAsBitVect(gt, 2, nBits=2048)
                    fb = AllChem.GetMorganFingerprintAsBitVect(gp, 2, nBits=2048)
                    from rdkit import DataStructs
                    tan = float(DataStructs.TanimotoSimilarity(fa, fb))
                from rdkit.Chem.rdMolDescriptors import CalcMolFormula
                samples.append({'i': si, 'mol': g, 'tanimoto': tan,
                                'exact': bool(smi and smi == gt_smi),
                                'formula_match': bool(gp is not None and gt is not None and
                                                      CalcMolFormula(gp) == CalcMolFormula(gt))})
            from rdkit.Chem.rdMolDescriptors import CalcMolFormula
            rows.append({'ds_index': e['ds_index'], 'truth': d['truth'][mi],
                         'truth_formula': CalcMolFormula(gt) if gt else '?',
                         'samples': samples})
        png = os.path.join(figdir, f'{split}_truth_vs_samples.png')
        svg = os.path.join(figdir, f'{split}_truth_vs_samples.svg')
        n, ncol = grid_for_split(rows, png, svg)
        print(f'{split}: {len(rows)} rows x {ncol} cols ({n} depictions) -> {png}')

    print(f'\nfigures in {figdir}:')
    for f in sorted(os.listdir(figdir)):
        print(f'  {f}  ({os.path.getsize(os.path.join(figdir, f))/1024:.0f} KB)')


if __name__ == '__main__':
    main()
