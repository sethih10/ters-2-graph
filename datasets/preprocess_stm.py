"""Build graph sidecars for the STM corpus at ``$DATA_ROOT/stm-data``.

Uses the bond inference and encoding of :mod:`datasets.preprocess_ters` with a ten-element
type map. Each split group's ``xyz`` rows are ``(x, y, z, Z, partial_charge)``, zero-padded;
the partial charge is not a formal charge and is not used as ``fc``.

Usage
-----
    python -m datasets.preprocess_stm --self-test
    python -m datasets.preprocess_stm --inspect $DATA_ROOT/stm-data/val.h5
    python -m datasets.preprocess_stm --census  $DATA_ROOT/stm-data/train.h5
    python -m datasets.preprocess_stm --data-dir $DATA_ROOT/stm-data \\
        --out-dir datasets/stm --stats
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import torch

from .preprocess_ters import (
    ATOMIC_SYMBOLS,
    check_record,
    encode_graph,
    geometry_stats,
    mol_from_geometry,
)

# Atomic number -> node index; order must match the ``stm_10elem`` atom_encoder.
STM_TYPES = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9}
STM_SYMBOLS = [ATOMIC_SYMBOLS[z] for z in sorted(STM_TYPES, key=STM_TYPES.get)]

XYZ_KEY = 'xyz'
CUBE_KEY = 'X'


def _split_group(f, split=None):
    """Return ``(name, group)`` for the split group; inferred if the file has only one."""
    keys = [k for k in f.keys() if hasattr(f[k], 'keys')]
    if split is not None:
        if split not in f:
            raise KeyError(f'no group {split!r}; file has {list(f.keys())}')
        return split, f[split]
    if len(keys) != 1:
        raise KeyError(
            f'expected exactly one group to pick the split from, found {keys}. '
            f'Pass --split explicitly.')
    return keys[0], f[keys[0]]


def iter_molecules(group, limit=None):
    """Yield ``(index, Z, pos, cid)`` per molecule with padding removed.

    Padding is detected by ``Z == 0``, not by zero coordinates: a real atom can sit at the origin.
    """
    xyz = group[XYZ_KEY]
    cids = group['cid'] if 'cid' in group else None
    n = xyz.shape[0] if limit is None else min(limit, xyz.shape[0])

    # Read in slabs rather than one HDF5 read per molecule.
    slab = 4096
    for start in range(0, n, slab):
        stop = min(start + slab, n)
        block = np.asarray(xyz[start:stop])
        for k in range(stop - start):
            rows = block[k]
            z = rows[:, 3].astype(int)
            keep = z > 0
            idx = start + k
            cid = None
            if cids is not None:
                raw = cids[idx]
                cid = raw.decode() if isinstance(raw, bytes) else str(raw)
            yield idx, z[keep], rows[keep, :3].astype(np.float64), cid


def census(h5_path, split=None, limit=None):
    """Count elements, molecule sizes and the partial-charge range in the raw ``xyz`` array."""
    import h5py

    elements, sizes = Counter(), Counter()
    n_empty = 0
    charge_min, charge_max = np.inf, -np.inf
    with h5py.File(h5_path, 'r') as f:
        name, g = _split_group(f, split)
        total = g[XYZ_KEY].shape[0]
        raw_charges = g[XYZ_KEY]
        for idx, z, pos, cid in iter_molecules(g, limit=limit):
            if len(z) == 0:
                n_empty += 1
                continue
            elements.update(int(v) for v in z)
            sizes[len(z)] += 1
        probe = np.asarray(raw_charges[:min(2048, total)])
        mask = probe[..., 3] > 0
        charge_min = float(probe[..., 4][mask].min())
        charge_max = float(probe[..., 4][mask].max())

    unknown = sorted(set(elements) - set(STM_TYPES))
    report = dict(
        split=name,
        n_molecules=total,
        n_scanned=sum(sizes.values()) + n_empty,
        n_empty=n_empty,
        elements={ATOMIC_SYMBOLS.get(z, f'Z{z}'): c for z, c in sorted(elements.items())},
        elements_z=dict(sorted(elements.items())),
        unknown_elements=unknown,
        n_atoms=dict(min=min(sizes) if sizes else 0, max=max(sizes) if sizes else 0),
        size_histogram=dict(sorted(sizes.items())),
        partial_charge_range=[charge_min, charge_max],
        suggested_atom_types=len(STM_TYPES),
        suggested_max_node=max(sizes) if sizes else 0,
    )
    if unknown:
        report['WARNING'] = (
            f'elements {unknown} are not in STM_TYPES -- every molecule containing one '
            f'will fail encode_graph. Extend STM_TYPES and data.atom_types together.')
    return report


def inspect(h5_path):
    """Print the HDF5 layout, including group attributes."""
    import h5py

    with h5py.File(h5_path, 'r') as f:
        print(f'{h5_path}')
        print(f'  root keys: {list(f.keys())}')
        for k in f.keys():
            node = f[k]
            if not hasattr(node, 'keys'):
                print(f'  dset {k:10s} {str(node.shape):24s} {node.dtype}')
                continue
            print(f'  group {k!r}')
            for ak, av in node.attrs.items():
                print(f'    attr {ak:18s} = {av}')
            for dk in node.keys():
                d = node[dk]
                print(f'    dset {dk:18s} {str(d.shape):24s} {d.dtype}')


def process_split(h5_path, out_path, split=None, charge=0, allow_fallback=True,
                  want_stats=False, check=True, limit=None):
    """Build one sidecar with the ten-element map and return a summary dict.

    Also writes ``<out>.failures.json`` and, with ``want_stats``, ``<out>.stats.json``.
    """
    import h5py
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    with h5py.File(h5_path, 'r') as f:
        name, g = _split_group(f, split)
        n_cubes = g[CUBE_KEY].shape[0] if CUBE_KEY in g else None
        print(f'reading {h5_path}  (split={name}, xyz={XYZ_KEY}, cubes={n_cubes})')

        records, failures = [], []
        n_fallback = n_aromatic_mols = total = 0
        for idx, z, pos, cid in iter_molecules(g, limit=limit):
            total += 1
            try:
                if len(z) == 0:
                    raise ValueError('molecule has no atoms after unpadding')
                mol, used_fallback, n_aromatic = mol_from_geometry(
                    z, pos, allow_fallback=allow_fallback, charge=charge)
                rec = encode_graph(mol, h5_index=idx, types=STM_TYPES)
                if check:
                    check_record(rec, types=STM_TYPES)
            except Exception as exc:
                failures.append(dict(h5_index=idx, cid=cid, n_atoms=int(len(z)),
                                     error=f'{type(exc).__name__}: {exc}'))
                continue
            rec['used_fallback'] = used_fallback
            rec['cid'] = cid                       # PubChem CID
            n_fallback += used_fallback
            n_aromatic_mols += n_aromatic > 0
            records.append(rec)

    # h5_index addresses the cube array, not the filtered record list.
    if n_cubes is not None and records:
        worst = max(int(r['h5_index']) for r in records)
        if worst >= n_cubes:
            raise IndexError(
                f'h5_index {worst} but {CUBE_KEY} holds {n_cubes} cubes -- '
                f'xyz and X are not the same length in {h5_path}')

    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    torch.save(records, out_path)

    survival = len(records) / total if total else 0.0
    print(f'  survived           {len(records)} / {total}  ({survival:.1%})')
    print(f'  covalent fallback  {n_fallback}')
    print(f'  aromatic (pre-kek) {n_aromatic_mols}')
    print(f'  wrote              {out_path}')

    if failures:
        fail_path = out_path + '.failures.json'
        with open(fail_path, 'w') as fh:
            json.dump(failures[:5000], fh, indent=2)
        reasons = Counter(f['error'].split(':')[0] for f in failures)
        print(f'  failures           {fail_path}  ({dict(reasons)})')

    if survival < 0.90:
        print('  WARNING: survival below 90%. Bond orders are being guessed for a large '
              'fraction of the corpus -- check --charge and the fallback rate before '
              'trusting any SMILES-exact metric.')

    summary = dict(split=name, h5=osp.basename(h5_path), n_total=total,
                   n_kept=len(records), survival_rate=survival, n_fallback=n_fallback,
                   n_molecules_with_aromatics=n_aromatic_mols)

    if want_stats and records:
        stats = geometry_stats(records)
        # Histograms over the encoded molecules only, unlike --census.
        kept_elements = Counter()
        node_hist = Counter()
        for r in records:
            kept_elements.update(int(v) for v in r['charge'].tolist())
            node_hist[int(r['num_atom'])] += 1
        stats['kept_elements'] = {ATOMIC_SYMBOLS.get(z, f'Z{z}'): c
                                  for z, c in sorted(kept_elements.items())}
        stats['train_n_nodes'] = dict(sorted(node_hist.items()))
        stats['suggested_atom_types'] = len(STM_TYPES)
        stats['suggested_max_node'] = max(node_hist)
        summary['geometry'] = stats
        stats_path = out_path + '.stats.json'
        with open(stats_path, 'w') as fh:
            json.dump(dict(summary, geometry=stats), fh, indent=2)
        print(f'  stats              {stats_path}')
        print(f'  elements kept      {stats["kept_elements"]}')
        print(f'  max_node           {stats["suggested_max_node"]}')

    return summary


def geometry_symbols(sidecar_path, top_k=8, limit=None):
    """Return the ``top_bond_sym`` / ``top_angle_sym`` / ``top_dihedral_sym`` config lists.

    Built with ``evaluation.cal_geometry``'s symbol builders; a symbol in any other format
    matches nothing there.
    """
    from evaluation.cal_geometry import (
        get_bond_symbol, get_bond_pair_symbol, get_triple_bond_symbol,
    )

    records = torch.load(sidecar_path, weights_only=False)
    if limit:
        records = records[:limit]

    bonds, angles, dihedrals = Counter(), Counter(), Counter()
    for r in records:
        mol = r['rdmol']
        for b in mol.GetBonds():
            bonds[get_bond_symbol(b)[0]] += 1
        for atom in mol.GetAtoms():
            nbr_bonds = list(atom.GetBonds())
            for i in range(len(nbr_bonds)):
                for j in range(i + 1, len(nbr_bonds)):
                    sym, _ = get_bond_pair_symbol([nbr_bonds[i], nbr_bonds[j]])
                    angles[sym] += 1
        for b in mol.GetBonds():
            a0, a1 = b.GetBeginAtom(), b.GetEndAtom()
            for left in a0.GetBonds():
                if left.GetIdx() == b.GetIdx():
                    continue
                for right in a1.GetBonds():
                    if right.GetIdx() == b.GetIdx():
                        continue
                    try:
                        sym, _ = get_triple_bond_symbol([left, b, right])
                    except Exception:
                        continue
                    dihedrals[sym] += 1

    return dict(
        n_molecules=len(records),
        top_bond_sym=[k for k, _ in bonds.most_common(top_k)],
        top_angle_sym=[k for k, _ in angles.most_common(top_k)],
        top_dihedral_sym=[k for k, _ in dihedrals.most_common(top_k)],
        bond_counts=dict(bonds.most_common(top_k)),
        angle_counts=dict(angles.most_common(top_k)),
        dihedral_counts=dict(dihedrals.most_common(top_k)),
    )


def atom_fc_census(sidecar_path):
    """Return ``atom_fc_num``: counts of each non-zero (element, formal charge) pair."""
    records = torch.load(sidecar_path, weights_only=False)
    counts = Counter()
    for r in records:
        for z, fc in zip(r['charge'].tolist(), r['fc'].tolist()):
            if int(fc) != 0:
                counts[f'{ATOMIC_SYMBOLS.get(int(z), int(z))}{int(fc)}'] += 1
    return dict(counts.most_common())


def self_test():
    """Test unpadding and the ten-element map on synthetic molecules."""
    ok = 0

    # 1. Unpadding must keep an atom at the origin.
    block = np.zeros((1, 54, 5), dtype=np.float32)
    block[0, 0] = [0.0, 0.0, 0.0, 6, -0.1]
    block[0, 1] = [1.09, 0.0, 0.0, 1, 0.03]
    block[0, 2] = [-0.36, 1.03, 0.0, 1, 0.03]
    block[0, 3] = [-0.36, -0.51, 0.89, 1, 0.03]
    block[0, 4] = [-0.36, -0.51, -0.89, 1, 0.03]

    class _FakeGroup(dict):
        pass
    g = _FakeGroup(xyz=block)
    idx, z, pos, cid = next(iter(iter_molecules(g)))
    assert len(z) == 5, f'unpadding kept {len(z)} atoms, expected 5 (methane)'
    assert (pos[0] == 0).all(), 'the origin atom was dropped -- padding test used coords'
    print(f'  ok  unpadding keeps an atom at the origin  ({len(z)} atoms)')
    ok += 1

    # 2. Elements outside the TERS map encode under STM_TYPES.
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    cases = [
        ('methane',       [6, 1, 1, 1, 1],
         [(0, 0, 0), (1.09, 0, 0), (-.36, 1.03, 0), (-.36, -.51, .89), (-.36, -.51, -.89)]),
        ('chloromethane', [6, 17, 1, 1, 1],
         [(0, 0, 0), (1.78, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]),
        ('bromomethane',  [6, 35, 1, 1, 1],
         [(0, 0, 0), (1.93, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]),
        ('fluoromethane', [6, 9, 1, 1, 1],
         [(0, 0, 0), (1.38, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]),
        ('methanethiol',  [6, 16, 1, 1, 1, 1],
         [(0, 0, 0), (1.82, 0, 0), (-.38, 1.03, 0), (-.38, -.51, .89), (-.38, -.51, -.89),
          (2.16, 1.23, 0)]),
    ]
    for name, z, pos in cases:
        mol, fallback, n_arom = mol_from_geometry(np.array(z), np.array(pos, dtype=float))
        rec = encode_graph(mol, h5_index=0, types=STM_TYPES)
        check_record(rec, types=STM_TYPES)
        assert int(rec['atom_type'].max()) < len(STM_TYPES)
        print(f'  ok  {name:14s} {rec["num_atom"]} atoms  '
              f'{rec["edge_index"].shape[1] // 2} bonds  fallback={fallback}')
        ok += 1

    # 3. An element outside the map must raise, not silently mis-index.
    try:
        mol, _, _ = mol_from_geometry(np.array([6, 1, 1, 1, 1]),
                                      np.array([(0, 0, 0), (1.09, 0, 0), (-.36, 1.03, 0),
                                                (-.36, -.51, .89), (-.36, -.51, -.89)],
                                               dtype=float))
        encode_graph(mol, h5_index=0, types={1: 0})      # C missing on purpose
    except ValueError as exc:
        assert 'outside the configured set' in str(exc), str(exc)
        print('  ok  unmapped element raises instead of mis-indexing')
        ok += 1
    else:
        raise AssertionError('an unmapped element did not raise')

    # 4. encode_graph's default element map must stay H/C/N/O.
    from .preprocess_ters import TYPES as TERS_TYPES
    assert TERS_TYPES == {1: 0, 6: 1, 7: 2, 8: 3}, 'the TERS element map changed'
    mol, _, _ = mol_from_geometry(np.array([6, 1, 1, 1, 1]),
                                  np.array([(0, 0, 0), (1.09, 0, 0), (-.36, 1.03, 0),
                                            (-.36, -.51, .89), (-.36, -.51, -.89)],
                                           dtype=float))
    rec_default = encode_graph(mol, h5_index=0)
    assert rec_default['atom_type'].tolist() == [1, 0, 0, 0, 0], 'TERS default drifted'
    print('  ok  encode_graph() with no types= is still the H/C/N/O map')
    ok += 1

    print(f'self-test: {ok}/{len(cases) + 3} passed')
    return ok == len(cases) + 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true', help='no data required')
    ap.add_argument('--inspect', metavar='H5', help='print the HDF5 layout and exit')
    ap.add_argument('--census', metavar='H5', help='element/size census over xyz, then exit')
    ap.add_argument('--geometry-symbols', metavar='SIDECAR',
                    help='top bond/angle/dihedral symbols + atom_fc_num from a written '
                         'sidecar -- the datasets_config.py values. Then exit.')
    ap.add_argument('--h5', help='path to one split HDF5')
    ap.add_argument('--out', help='output sidecar .pt (single-split mode)')
    ap.add_argument('--data-dir', help='directory holding {train,val,test}.h5')
    ap.add_argument('--out-dir', default='datasets/stm', help='where graphs_{split}.pt go')
    ap.add_argument('--split', default=None, help='group name inside the HDF5')
    ap.add_argument('--charge', type=int, default=0, help='net charge for DetermineBonds')
    ap.add_argument('--no-fallback', action='store_true',
                    help='fail instead of falling back to covalent-radius connectivity')
    ap.add_argument('--stats', action='store_true', help='write geometry stats JSON')
    ap.add_argument('--no-check', action='store_true', help='skip per-record validation')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only the first N molecules (smoke runs)')
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if a.inspect:
        inspect(a.inspect)
        return
    if a.census:
        print(json.dumps(census(a.census, a.split, a.limit), indent=2))
        return
    if a.geometry_symbols:
        out = geometry_symbols(a.geometry_symbols, limit=a.limit)
        out['atom_fc_num'] = atom_fc_census(a.geometry_symbols)
        print(json.dumps(out, indent=2))
        return

    kw = dict(split=a.split, charge=a.charge, allow_fallback=not a.no_fallback,
              want_stats=a.stats, check=not a.no_check, limit=a.limit)

    if a.data_dir:
        summaries = []
        for name in ('train', 'val', 'test'):
            h5 = osp.join(a.data_dir, f'{name}.h5')
            if not osp.exists(h5):
                print(f'skipping {h5} (missing)')
                continue
            summaries.append(process_split(h5, osp.join(a.out_dir, f'graphs_{name}.pt'),
                                           **kw))
        print(json.dumps(summaries, indent=2))
    elif a.h5 and a.out:
        print(json.dumps(process_split(a.h5, a.out, **kw), indent=2))
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
