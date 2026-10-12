"""Build graph sidecars from the atom geometry in TERS HDF5 files.

Infers bonds with RDKit, kekulizes, and writes one ``graphs_<split>.pt`` per split: a list of
QM9S-style graph records, each with an ``h5_index`` pointing at its image cube.

Usage
-----
    python -m datasets.preprocess_ters --self-test
    python -m datasets.preprocess_ters --h5 .../train.h5 --inspect
    python -m datasets.preprocess_ters \\
        --data-dir $DATA_ROOT/planar_hdf5_0.5 \\
        --out-dir datasets/ters --stats
    python -m datasets.preprocess_ters --h5 .../train.h5 \\
        --out datasets/ters/graphs_train.pt --stats
    python -m datasets.preprocess_ters --elements hcnof --h5 .../qm9s/train.h5 \\
        --out <graph_root>/graphs_train.pt --stats
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import torch


# Must cover every element in preprocess_stm.STM_TYPES.
ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I',
}

# Atomic number -> node index; order must match the ``ters_hcno`` atom_encoder.
TYPES = {1: 0, 6: 1, 7: 2, 8: 3}

# H/C/N/O/F for QM9S; order must match the ``qm9_second_half`` atom_encoder.
TYPES_HCNOF = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
ELEMENT_SETS = {'hcno': TYPES, 'hcnof': TYPES_HCNOF}

# Candidate HDF5 dataset names, tried in order.
ATOM_Z_KEYS = ('atomic_numbers', 'atomic_number', 'atom_types', 'Z', 'species')
ATOM_POS_KEYS = ('atom_positions', 'atom_pos', 'positions', 'coordinates', 'coords', 'xyz')


def build_xyz_block(atomic_numbers, atom_pos):
    """Return an XYZ-format string for ``Chem.MolFromXYZBlock``."""
    t = list(zip(atomic_numbers, atom_pos))
    text = f"{len(t)}\nComment\n"
    for atom, pos in t:
        pos_str = "\t".join(f"{coord:.6f}" for coord in pos)
        text += ATOMIC_SYMBOLS.get(int(atom), "X") + "\t" + pos_str + "\n"
    return text


# Cordero (2008) covalent radii in Angstrom, as in ``ase.data.covalent_radii``; used without ase.
_COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,                       # H, C, N, O
    9: 0.57, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20,  # F, Si, P, S, Cl, Br
}
_COVALENT_RADII_HCNO = _COVALENT_RADII      # backward-compatible alias


def _covalent_radius(z):
    try:
        from ase.data import covalent_radii
        return float(covalent_radii[int(z)])
    except ImportError:
        try:
            return _COVALENT_RADII[int(z)]
        except KeyError:
            raise ImportError(
                f'covalent radius for Z={int(z)} needs `pip install ase` -- only '
                f'{sorted(_COVALENT_RADII)} are vendored') from None


def covalent_bond_fallback(mol, atomic_numbers, atom_pos, tolerance=1.15):
    """Bond atoms closer than ``tolerance`` times the sum of their covalent radii.

    Every bond is single, so bond orders from this path are not meaningful.
    """
    from rdkit import Chem

    pos = np.asarray(atom_pos, dtype=np.float64)
    z = np.asarray(atomic_numbers, dtype=int)
    n = len(z)

    radii = np.array([_covalent_radius(zi) for zi in z])
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cutoff = (radii[:, None] + radii[None, :]) * tolerance
    adj = (d <= cutoff) & ~np.eye(n, dtype=bool)

    rw = Chem.RWMol(mol)
    for i, j in zip(*np.where(np.triu(adj))):
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    return rw.GetMol()


def mol_from_geometry(atomic_numbers, atom_pos, allow_fallback=True, charge=0):
    """Infer bonds from geometry. Returns ``(mol, used_fallback, n_aromatic)``.

    ``mol`` is kekulized; ``n_aromatic`` counts aromatic bonds before kekulization.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    xyz = build_xyz_block(atomic_numbers, atom_pos)
    raw = Chem.MolFromXYZBlock(xyz)
    if raw is None:
        raise ValueError('MolFromXYZBlock returned None (malformed xyz block)')

    used_fallback = False
    try:
        mol = Chem.Mol(raw)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception:
        if not allow_fallback:
            raise
        mol = covalent_bond_fallback(raw, atomic_numbers, atom_pos)
        used_fallback = True

    Chem.SanitizeMol(mol)
    n_aromatic = sum(1 for b in mol.GetBonds()
                     if b.GetBondType() == Chem.BondType.AROMATIC)

    # EdgeComSpectraTransform(include_aromatic=False) erases aromatic bonds; kekulize first.
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol, used_fallback, n_aromatic


def encode_graph(mol, h5_index, types=None):
    """Encode a kekulized Mol as a graph record in the ``QM9S.process`` format.

    ``types`` maps atomic number to node index (default ``TYPES``); other elements raise
    ``ValueError``.
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT

    types = TYPES if types is None else types

    bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> no edge

    n = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

    type_idx, charges, formal_charges = [], [], []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in types:
            raise ValueError(
                f'element Z={z} outside the configured set {sorted(types)}')
        type_idx.append(types[z])
        charges.append(z)
        formal_charges.append(atom.GetFormalCharge())

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long).reshape(2, -1)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    perm = (edge_index[0] * n + edge_index[1]).argsort()

    return dict(
        atom_type=torch.tensor(type_idx),
        pos=pos,
        charge=torch.tensor(charges),
        fc=torch.tensor(formal_charges),
        edge_index=edge_index[:, perm],
        edge_type=edge_type[perm],
        num_atom=n,
        rdmol=Chem.Mol(mol),
        h5_index=h5_index,
    )


# --- HDF5 access ------------------------------------------------------------------------

def _pick_key(h5, candidates, what):
    for k in candidates:
        if k in h5:
            return k
    raise KeyError(
        f'could not find the {what} dataset. Tried {candidates}. '
        f'Available top-level keys: {list(h5.keys())}. '
        f'Pass the right name explicitly with --z-key / --pos-key.'
    )


def inspect(h5_path):
    """Print the HDF5 layout."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        print(f'{h5_path}')

        def show(name, obj):
            import h5py as _h
            if isinstance(obj, _h.Dataset):
                print(f'  {name:32s} shape={obj.shape} dtype={obj.dtype}')
            else:
                print(f'  {name}/')
        f.visititems(show)


def _numeric_group_accessor(node, what):
    """Return ``(length, getter)`` for a per-molecule HDF5 node.

    Supports a dense ``(N, max_atoms)`` dataset padded with Z=0, or a group of datasets keyed
    ``'0'..'N-1'`` and indexed by ``str(i)``. Do not enumerate ``.keys()``: h5py returns them
    in lexicographic order, which pairs graphs with the wrong cubes.
    """
    import h5py
    if isinstance(node, h5py.Group):
        keys = set(node.keys())
        n = len(keys)
        expected = {str(i) for i in range(n)}
        if keys != expected:
            missing = sorted(expected - keys)[:5]
            extra = sorted(keys - expected)[:5]
            raise KeyError(
                f'{what} is a group whose keys are not exactly 0..{n - 1}. '
                f'missing={missing} unexpected={extra}. This module indexes such a group by '
                f'str(i) so that graph i pairs with cube i; it cannot use .keys() order, '
                f'which h5py returns lexicographically.')
        return n, lambda i: node[str(i)]
    return len(node), lambda i: node[i]


def iter_molecules(h5, z_key, pos_key):
    """Yield ``(index, atomic_numbers, positions)`` with zero padding removed.

    ``index`` is the cube index: molecule ``i`` is conditioned on ``channels_*[i]``.
    """
    nz, get_z = _numeric_group_accessor(h5[z_key], z_key)
    npos, get_p = _numeric_group_accessor(h5[pos_key], pos_key)
    if nz != npos:
        raise ValueError(f'{z_key} holds {nz} molecules but {pos_key} holds {npos}')

    for i in range(nz):
        z = np.asarray(get_z(i)).reshape(-1).astype(int)
        raw = np.asarray(get_p(i), dtype=np.float64)
        p = raw.reshape(len(z), 3) if raw.size == len(z) * 3 else raw.reshape(-1, 3)
        if len(p) != len(z):
            raise ValueError(
                f'molecule {i}: {z_key} has {len(z)} entries but {pos_key} has {len(p)} rows')
        keep = z > 0
        yield i, z[keep], p[keep]


# --- Geometry statistics ----------------------------------------------------------------

def geometry_stats(records):
    """Summarise molecule geometry: size, element counts, and nearest-neighbour and bond lengths."""
    radii, nn_dists, z_abs, n_atoms, bond_lens = [], [], [], [], []
    elements = Counter()

    for r in records:
        pos = r['pos'].numpy().astype(np.float64)
        n_atoms.append(int(r['num_atom']))
        elements.update(int(z) for z in r['charge'].tolist())

        centred = pos - pos.mean(axis=0, keepdims=True)
        radii.append(np.linalg.norm(centred, axis=-1).max())
        z_abs.append(np.abs(centred[:, 2]).max())

        if len(pos) > 1:
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            nn_dists.extend(d.min(axis=1).tolist())

        ei = r['edge_index'].numpy()
        if ei.size:
            bond_lens.extend(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=-1).tolist())

    radii, nn_dists = np.array(radii), np.array(nn_dists)
    p95_radius = float(np.percentile(radii, 95))
    p95_nn = float(np.percentile(nn_dists, 95))

    return dict(
        n_molecules=len(records),
        elements=dict(sorted(elements.items())),
        n_atoms=dict(min=int(min(n_atoms)), median=float(np.median(n_atoms)),
                     p95=float(np.percentile(n_atoms, 95)), max=int(max(n_atoms))),
        centred_radius=dict(max=float(radii.max()), p95=p95_radius),
        max_abs_z=float(np.max(z_abs)),
        nn_distance=dict(min=float(nn_dists.min()), median=float(np.median(nn_dists)),
                         p95=p95_nn),
        bond_length=dict(n=len(bond_lens),
                         median=float(np.median(bond_lens)) if bond_lens else None,
                         max=float(np.max(bond_lens)) if bond_lens else None),
    )


def check_record(rec, expect_aromatic_free=True, types=None):
    """Assert that an encoded record is well formed and every edge survives the transform.

    Returns a note saying whether the ``EdgeComSpectraTransform`` check ran.
    """
    types = TYPES if types is None else types
    n = int(rec['num_atom'])
    ei, et = rec['edge_index'], rec['edge_type']

    assert rec['pos'].shape == (n, 3), f"pos {tuple(rec['pos'].shape)} != ({n}, 3)"
    assert rec['atom_type'].shape == (n,), 'atom_type length != num_atom'
    assert rec['fc'].shape == (n,), 'fc length != num_atom'
    assert int(rec['atom_type'].max()) < len(types), \
        f'atom_type index outside the configured {len(types)}-element set'
    assert ei.shape[1] == et.shape[0], 'edge_index and edge_type disagree'
    assert ei.shape[1] % 2 == 0, 'odd edge count -- edges must be stored both directions'
    assert int(ei.max()) < n if ei.numel() else True, 'edge_index refers to a missing atom'

    if expect_aromatic_free:
        assert int(et.max()) <= 3 if et.numel() else True, \
            'bond type 4 (aromatic) survived kekulization -- it would be erased by ' \
            'EdgeComSpectraTransform'

    keys =(ei[0] * n + ei[1]).tolist()
    assert keys == sorted(keys), 'edges are not sorted by (row * N + col)'

    try:
        from torch_geometric.data import Data
        from datasets.build_dataset import EdgeComSpectraTransform
    except ImportError:
        return 'transform check skipped (torch_geometric unavailable)'

    data = Data(atom_type=rec['atom_type'], pos=rec['pos'], fc=rec['fc'],
                edge_index=ei, edge_type=et)
    # atom_type holds indices, so pass the encoder's values, as get_dataset does
    data = EdgeComSpectraTransform(list(range(len(types))), include_aromatic=False)(data)

    n_exist = int(data.edge_one_hot[..., 0].sum())
    assert n_exist == ei.shape[1], (
        f'transform kept {n_exist} of {ei.shape[1]} directed edges -- bonds are being '
        f'silently erased')
    assert int(data.atom_one_hot.sum()) == n, 'atom one-hot does not cover every atom'
    return 'transform check passed'


def self_test():
    """Run the RDKit pipeline end to end on synthetic molecules; needs no data."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')

    cases = [('methane', 'C'), ('ethanol', 'CCO'), ('benzene', 'c1ccccc1'),
             ('pyridine', 'c1ccncc1'), ('nitromethane', 'C[N+](=O)[O-]'),
             ('acetic acid', 'CC(=O)O')]

    failed, note = 0, 'no cases ran'
    for name, smiles in cases:
        try:
            ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(ref, randomSeed=0xC0FFEE)
            AllChem.MMFFOptimizeMolecule(ref)
            z = [a.GetAtomicNum() for a in ref.GetAtoms()]
            pos = ref.GetConformer().GetPositions()

            mol, used_fallback, n_aromatic = mol_from_geometry(z, pos)
            rec = encode_graph(mol, h5_index=-1)
            note = check_record(rec)

            got = Chem.MolToSmiles(Chem.RemoveHs(mol))
            want = Chem.MolToSmiles(Chem.RemoveHs(ref))
            match = 'exact' if got == want else f'DIFFERS (got {got}, want {want})'
            print(f'  {name:14s} {rec["num_atom"]:3d} atoms  '
                  f'{rec["edge_index"].shape[1] // 2:2d} bonds  '
                  f'aromatic-in={n_aromatic:2d}  '
                  f'fallback={used_fallback}  {match}')
            if got != want:
                failed += 1
        except Exception as exc:
            print(f'  {name:14s} FAILED  {type(exc).__name__}: {exc}')
            failed += 1

    # Fluorine must encode under TYPES_HCNOF and raise under the default TYPES.
    for name, smiles in [('fluoromethane', 'CF'), ('trifluoroethanol', 'OCC(F)(F)F')]:
        try:
            ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(ref, randomSeed=0xC0FFEE)
            AllChem.MMFFOptimizeMolecule(ref)
            z = [a.GetAtomicNum() for a in ref.GetAtoms()]
            mol, _, _ = mol_from_geometry(z, ref.GetConformer().GetPositions())
            rec = encode_graph(mol, h5_index=-1, types=TYPES_HCNOF)
            check_record(rec, types=TYPES_HCNOF)
            n_f = int((rec['atom_type'] == TYPES_HCNOF[9]).sum())
            try:
                encode_graph(mol, h5_index=-1)
                rejected = False
            except ValueError:
                rejected = True
            ok = n_f == smiles.count('F') and rejected
            print(f'  {name:14s} {rec["num_atom"]:3d} atoms  F={n_f}  '
                  f'hcnof=ok  hcno={"rejected" if rejected else "ACCEPTED"}'
                  f'{"" if ok else "  FAILED"}')
            failed += not ok
        except Exception as exc:
            print(f'  {name:14s} FAILED  {type(exc).__name__}: {exc}')
            failed += 1

    print(f'\n{note}')
    print('self-test: ' + ('all cases passed' if not failed else f'{failed} case(s) failed'))
    return failed


def process_split(h5_path, out_path, z_key=None, pos_key=None, charge=0,
                  allow_fallback=True, want_stats=False, check=True, types=None):
    """Build one sidecar and return a summary dict; ``types`` defaults to ``TYPES``."""
    import h5py
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    with h5py.File(h5_path, 'r') as f:
        z_key = z_key or _pick_key(f, ATOM_Z_KEYS, 'atomic-number')
        pos_key = pos_key or _pick_key(f, ATOM_POS_KEYS, 'coordinate')
        print(f'reading {h5_path}  (Z={z_key}, pos={pos_key})')

        records, failures = [], []
        n_fallback = n_aromatic_mols = total = 0
        for idx, z, pos in iter_molecules(f, z_key, pos_key):
            total += 1
            try:
                mol, used_fallback, n_aromatic = mol_from_geometry(
                    z, pos, allow_fallback=allow_fallback, charge=charge)
                rec = encode_graph(mol, h5_index=idx, types=types)
                if check:
                    check_record(rec, types=types)
            except Exception as exc:
                failures.append(dict(h5_index=idx, n_atoms=int(len(z)),
                                     error=f'{type(exc).__name__}: {exc}'))
                continue
            rec['used_fallback'] = used_fallback
            n_fallback += used_fallback
            n_aromatic_mols += n_aromatic > 0
            records.append(rec)

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
            json.dump(failures, fh, indent=2)
        print(f'  failures           {fail_path}')

    if survival < 0.90:
        print('  WARNING: survival below 90%. Check --charge and the covalent-radius '
              'fallback rate before training.')

    summary = dict(split=osp.basename(h5_path), n_total=total, n_kept=len(records),
                   survival_rate=survival, n_fallback=n_fallback,
                   n_molecules_with_aromatics=n_aromatic_mols)

    if want_stats and records:
        stats = geometry_stats(records)
        summary['geometry'] = stats
        stats_path = out_path + '.stats.json'
        with open(stats_path, 'w') as fh:
            json.dump(dict(summary, geometry=stats), fh, indent=2)
        print(f'  stats              {stats_path}')

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', help='path to one split HDF5')
    ap.add_argument('--out', help='output sidecar .pt (single-split mode)')
    ap.add_argument('--data-dir', help='directory holding {train,val,test}.h5 -- '
                                       'processes all three')
    ap.add_argument('--out-dir', default='datasets/ters',
                    help='where graphs_<split>.pt go in --data-dir mode')
    ap.add_argument('--splits', default='train,val,test',
                    help='comma-separated split names for --data-dir mode')
    ap.add_argument('--z-key', help='override the atomic-number dataset name')
    ap.add_argument('--pos-key', help='override the coordinate dataset name')
    ap.add_argument('--charge', type=int, default=0, help='total charge for DetermineBonds')
    ap.add_argument('--elements', choices=sorted(ELEMENT_SETS), default='hcno',
                    help='element set for the node types: hcno (planar TERS, default) or '
                         'hcnof (QM9S). Must match data.atom_types / data.info_name.')
    ap.add_argument('--no-fallback', action='store_true',
                    help='drop molecules DetermineBonds fails on instead of using '
                         'covalent-radius connectivity')
    ap.add_argument('--no-check', action='store_true',
                    help='skip the per-record well-formedness check (faster)')
    ap.add_argument('--stats', action='store_true',
                    help='also write geometry statistics as JSON and print suggested '
                         'config values')
    ap.add_argument('--inspect', action='store_true', help='print the HDF5 layout and exit')
    ap.add_argument('--self-test', action='store_true',
                    help='run the pipeline on synthetic molecules; needs no data')
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(1 if self_test() else 0)
    if args.inspect:
        if not args.h5:
            ap.error('--inspect needs --h5')
        inspect(args.h5)
        return

    common = dict(z_key=args.z_key, pos_key=args.pos_key, charge=args.charge,
                  allow_fallback=not args.no_fallback, want_stats=args.stats,
                  check=not args.no_check, types=ELEMENT_SETS[args.elements])

    if args.data_dir:
        summaries = []
        for split in args.splits.split(','):
            split = split.strip()
            h5_path = osp.join(args.data_dir, f'{split}.h5')
            if not osp.exists(h5_path):
                print(f'skipping {split}: {h5_path} not found')
                continue
            summaries.append(process_split(
                h5_path, osp.join(args.out_dir, f'graphs_{split}.pt'), **common))
            print()
    elif args.h5:
        if not args.out:
            ap.error('--out is required with --h5')
        summaries = [process_split(args.h5, args.out, **common)]
    else:
        ap.error('give either --h5/--out, --data-dir, --inspect, or --self-test')

    if not summaries:
        return

    print('--- Per-split summary ---')
    print(f'{"split":12s} {"kept/total":>14s} {"survival":>9s} {"fallback":>9s} '
          f'{"aromatic":>9s}')
    for s in summaries:
        print(f'{s["split"]:12s} {s["n_kept"]:6d}/{s["n_total"]:<7d} '
              f'{s["survival_rate"]:8.1%} {s["n_fallback"]:9d} '
              f'{s["n_molecules_with_aromatics"]:9d}')

    if args.stats:
        geo = [s['geometry'] for s in summaries if 'geometry' in s]
        if geo:
            print('\nSuggested config values (take the max across splits):')
            print(f'  data.max_node               = '
                  f'{max(g["n_atoms"]["max"] for g in geo)}')
            elements = sorted({z for g in geo for z in map(int, g['elements'])})
            print(f'  data.atom_types             = {len(elements)}   (Z = {elements})')


if __name__ == '__main__':
    main()
