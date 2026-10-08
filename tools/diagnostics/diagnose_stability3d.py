"""3D and 2D stability metrics on ground-truth STM test molecules.

Prints molecule- and atom-level stability from ``check_stability`` (3D) and
``check_2D_stability`` (2D), grouped by element set and by whether any atom is charged.

Usage
-----
    python -m tools.diagnostics.diagnose_stability3d [-n N]
"""
import argparse, importlib.util, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from datasets import get_dataset
from datasets.datasets_config import get_dataset_info
from evaluation.stability import check_stability, check_2D_stability

QM9_ELEMENTS = {'H', 'C', 'N', 'O', 'F'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-qm9', default='configs/base_stm.py')
    ap.add_argument('--info', default='stm_10elem')
    ap.add_argument('-n', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location('cfg', a.config_qm9)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    _, _, _, test_ds, _ = get_dataset(m.get_config())
    info = get_dataset_info(a.info)
    dec = info['atom_decoder']

    n = min(a.n, len(test_ds))
    sel = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(a.seed))[:n].tolist()
    print(f'{n} ground-truth molecules from the test split (correct by construction)\n')

    strata = collections.defaultdict(lambda: dict(mols=0, stable=0, atoms=0, atoms_stable=0,
                                                  stable2d=0, atoms_stable2d=0))
    for i in sel:
        s = test_ds[i]
        k = int(s.num_atom)
        at = s.atom_type[:k].long()
        pos = s.pos[:k].float()
        fc = s.fc[:k].long()
        dense = torch.zeros(k, k, dtype=torch.long)
        ei, et = s.edge_index, s.edge_type
        for e in range(ei.shape[1]):
            x, y = int(ei[0, e]), int(ei[1, e])
            if x < k and y < k:
                dense[x, y] = dense[y, x] = int(et[e])

        syms = {dec[int(t)] for t in at}
        heavy = syms - QM9_ELEMENTS
        charged = bool((fc != 0).any())
        key = ('non-QM9 elem' if heavy else 'H/C/N/O/F only',
               'charged' if charged else 'neutral')

        ms, nsb, na, _ = check_stability(pos, at, info)
        ms2, nsb2, na2, _ = check_2D_stability(pos, at, fc, dense, info)
        d = strata[key]
        d['mols'] += 1; d['stable'] += int(ms); d['atoms'] += na
        d['atoms_stable'] += nsb; d['stable2d'] += int(ms2); d['atoms_stable2d'] += nsb2

    print(f"{'elements':16s} {'charge':9s} {'n':>6s} "
          f"{'3D mol':>8s} {'3D atom':>8s} {'2D mol':>8s} {'2D atom':>8s}")
    tot = collections.Counter()
    for key in sorted(strata):
        d = strata[key]
        print(f'{key[0]:16s} {key[1]:9s} {d["mols"]:6d} '
              f'{d["stable"]/d["mols"]:8.4f} {d["atoms_stable"]/d["atoms"]:8.4f} '
              f'{d["stable2d"]/d["mols"]:8.4f} {d["atoms_stable2d"]/d["atoms"]:8.4f}')
        for f in ('mols', 'stable', 'atoms', 'atoms_stable', 'stable2d', 'atoms_stable2d'):
            tot[f] += d[f]
    print(f'{"ALL":16s} {"":9s} {tot["mols"]:6d} '
          f'{tot["stable"]/tot["mols"]:8.4f} {tot["atoms_stable"]/tot["atoms"]:8.4f} '
          f'{tot["stable2d"]/tot["mols"]:8.4f} {tot["atoms_stable2d"]/tot["atoms"]:8.4f}')
    print('\nCompare the 3D and 2D columns on the same molecules: 2D uses the stored '
          '\nbond orders and a charge-aware valence table, 3D re-infers bonds from distances '
          '\nwith a charge-blind one. Any gap comes from the metric, not the molecules.')


if __name__ == '__main__':
    main()
