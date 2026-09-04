"""Phase 1 -- derive molecular graph targets from TERS HDF5 geometry.

The TERS HDF5 stores atomic numbers and Cartesian coordinates but no connectivity.
DMT's targets need a bond-order matrix and every metric in ``evaluation/`` needs an
RDKit ``Mol``.  This script builds both offline into a small sidecar per split; the
105 MB image cubes stay in the HDF5 and are never copied.

Provenance (see docs/02-reuse-map.md):
  * ``build_xyz_block``        -- ADAPT of Multimodal-ters/scripts/npz_to_hdf5_converter.py:90-95
                                  (same producer, handed to Chem.MolFromXYZBlock instead of
                                  the rasteriser)
  * ``ATOMIC_SYMBOLS``         -- REUSE of Multimodal-ters/src/datasets/ters_image_to_image_sh.py:35-46
  * ``covalent_bond_fallback`` -- ADAPT of Multimodal-ters/src/datasets/ters_image_to_image_sh.py:160-174
                                  (the working copy; NOT src/utils/dataset_utils.py:5-30, which
                                  raises NameError on an undefined ``covalent_radii_new``)
  * ``encode_graph``           -- REUSE of the encoding convention in
                                  datasets/qm9s_dataset.py:206-260 (types/bonds dicts, both edge
                                  directions, ``(row * N + col)`` sort)
  * orchestration + stats      -- NEW

Usage
-----
    # 0. prove the environment and the port work -- needs no data at all
    python -m datasets.preprocess_ters --self-test

    # 1. confirm the HDF5 layout (key names are probed, not assumed)
    python -m datasets.preprocess_ters --h5 .../train.h5 --inspect

    # 2. build all three sidecars + the Phase 0 geometry rows
    python -m datasets.preprocess_ters \
        --data-dir /scratch/phys/sin/sethih1/Multimodal_TERS/planar_hdf5_0.05 \
        --out-dir datasets/ters --stats

    # ...or one split at a time
    python -m datasets.preprocess_ters --h5 .../train.h5 \
        --out datasets/ters/graphs_train.pt --stats

If the probe picks the wrong HDF5 datasets, pass ``--z-key`` / ``--pos-key`` rather than
editing this file.  Run once; requires ``rdkit``.  ``ase`` is optional -- it is used only
by the covalent-radius fallback, which falls back to vendored H/C/N/O radii without it.
"""

import argparse
import math
import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import torch


# --- REUSE: Multimodal-ters/src/datasets/ters_image_to_image_sh.py:35-46 -----------------
# Only H/C/N/O are expected in this dataset (docs/04-phase0-measurements.md), but keep the
# wider map so an unexpected element surfaces as a real symbol rather than "X".
ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I',
}

# --- REUSE: datasets/qm9s_dataset.py:206-207 (minus F, which TERS does not contain) -------
TYPES = {1: 0, 6: 1, 7: 2, 8: 3}          # atomic number -> node feature index

# Candidate HDF5 key names, tried in order.  The Multimodal-ters converter is not mounted
# where this was written, so the exact names are probed rather than assumed; --inspect
# prints the real ones.
ATOM_Z_KEYS = ('atomic_numbers', 'atomic_number', 'atom_types', 'Z', 'species')
ATOM_POS_KEYS = ('atom_positions', 'atom_pos', 'positions', 'coordinates', 'coords', 'xyz')


def build_xyz_block(atomic_numbers, atom_pos):
    """ADAPT of Multimodal-ters/scripts/npz_to_hdf5_converter.py:90-95.

    Unchanged except for the consumer: in MT this feeds ``molecule_circular_image``;
    here it feeds ``Chem.MolFromXYZBlock``.
    """
    t = list(zip(atomic_numbers, atom_pos))
    text = f"{len(t)}\nComment\n"
    for atom, pos in t:
        pos_str = "\t".join(f"{coord:.6f}" for coord in pos)
        text += ATOMIC_SYMBOLS.get(int(atom), "X") + "\t" + pos_str + "\n"
    return text


# Cordero (2008) covalent radii in Angstrom -- the same values ``ase.data.covalent_radii``
# carries, vendored for H/C/N/O so this script runs in environments without ``ase`` (the im2g
# venv has rdkit but not ase). ASE is still preferred when importable so that a dataset
# containing an unexpected element is handled rather than rejected.
_COVALENT_RADII_HCNO = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}


def _covalent_radius(z):
    try:
        from ase.data import covalent_radii
        return float(covalent_radii[int(z)])
    except ImportError:
        try:
            return _COVALENT_RADII_HCNO[int(z)]
        except KeyError:
            raise ImportError(
                f'covalent radius for Z={int(z)} needs `pip install ase` -- only '
                f'{sorted(_COVALENT_RADII_HCNO)} are vendored') from None


def covalent_bond_fallback(mol, atomic_numbers, atom_pos, tolerance=1.15):
    """ADAPT of Multimodal-ters/src/datasets/ters_image_to_image_sh.py:160-174.

    MT's loop collects a ``set`` of ``(Z_i, Z_j)`` pairs; we need the loop body, not its
    return value, so this returns index pairs instead and writes them onto ``mol``.

    Connectivity only -- every bond is assigned order 1.  Molecules that go down this path
    have meaningless bond orders, so SMILES exact-match is not a valid metric for them.
    That is why ``used_fallback`` is recorded per molecule.
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
    """Geometry -> kekulized RDKit Mol.  Returns ``(mol, used_fallback, n_aromatic)``.

    ``n_aromatic`` is counted *before* kekulization -- it feeds the
    ``include_aromatic`` decision in docs/04-phase0-measurements.md.
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

    # Kekulize, or aromatic rings vanish silently: EdgeComSpectraTransform maps bond type
    # 4 -> 0 (datasets/build_dataset.py:120-122) when include_aromatic=False, which zeroes
    # the feature, which zeroes the *existence* channel at line 138.  An aromatic bond
    # becomes no bond at all, with no error.  QM9's SDF is pre-kekulized; DetermineBonds
    # output is not.
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol, used_fallback, n_aromatic


def encode_graph(mol, h5_index):
    """REUSE of the encoding convention at datasets/qm9s_dataset.py:226-260.

    Deviation from docs/03-execution-plan.md: ``fc`` carries RDKit's assigned formal
    charges rather than zeros.  The HDF5 has none, but DetermineBonds assigns them (nitro
    groups, N-oxides), and ``fc`` is a real feature channel consumed by
    CollateSpectra.__call__ (datasets/build_dataset.py:430) -- zeroing it would train the
    model to predict a charge state the target molecule does not have.
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT

    bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> no edge

    n = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

    type_idx, charges, formal_charges = [], [], []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in TYPES:
            raise ValueError(f'element Z={z} outside the H/C/N/O set')
        type_idx.append(TYPES[z])
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
    """Print the file layout.  Run this before the first real pass."""
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

    Two layouts occur in the wild and they index differently:

    * **Dense dataset** ``(N, max_atoms)`` -- ``node[i]`` works, padded with Z=0.
    * **Group of per-molecule datasets** keyed ``'0'..'N-1'`` (what the real planar TERS
      files use) -- ``node[i]`` raises ``TypeError``; the key must be a *string*.

    The group case is where a silent-corruption bug lives.  ``h5py`` returns group keys in
    **lexicographic** order -- ``'0', '1', '10', '100', '101', ...`` -- so pairing
    ``enumerate(node.keys())`` with a cube index binds nearly every graph to the wrong
    cube, with no error anywhere.  Verified against the real file: keys sort as
    ``0,1,10,100`` but the cube at dense index ``i`` belongs to group key ``str(i)``
    (confirmed by matching each cube's live frequency channels to that molecule's
    vibrational modes, and each ``targets`` disc to its atom).  So this indexes by
    ``str(i)`` and never iterates ``.keys()``.
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
    """Yield ``(index, atomic_numbers, positions)``, stripping zero padding.

    ``index`` is the dense cube index: the molecule yielded as ``i`` is the one whose
    conditioning cube is ``channels_*[i]``.  See ``_numeric_group_accessor`` for why that
    is not the same as enumerating group keys.

    Handles both a ragged group layout and a dense ``(N, max_atoms)`` layout padded with
    zeros -- Z=0 is not a real element, so it is an unambiguous padding marker.
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


# --- Phase 0 geometry statistics (folded in -- same data pass, no extra cost) -------------

def geometry_stats(records):
    """Produce the geometry rows of docs/04-phase0-measurements.md.

    Centres each molecule the way losses.py:463 does at train time, so the radii reported
    here are the ones ``model.normalize_factors[0]`` actually has to normalise.
    """
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

        # The actual bonded distances, not a proxy for them.  The cutoff below has to keep
        # every one of these, and the nearest-neighbour distance systematically understates
        # the longest bond: an atom's nearest neighbour is usually an H at ~1.0 A, while the
        # longest bond in the same molecule is a C-C at ~1.6 A.
        ei = r['edge_index'].numpy()
        if ei.size:
            bond_lens.extend(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=-1).tolist())

    radii, nn_dists = np.array(radii), np.array(nn_dists)
    p95_radius = float(np.percentile(radii, 95))
    p95_nn = float(np.percentile(nn_dists, 95))
    # Round the longest real bond UP to the next 0.1 A.  Measured on the real planar file the
    # longest bond is 1.601 A, and `p95_nn + 0.2` gives 1.600 -- short by a thousandth, which
    # would silently drop the single longest bond in the corpus.  Bonds are what this
    # adjacency is for, so measure them.
    max_bond = float(np.max(bond_lens)) if bond_lens else float(p95_nn + 0.2)
    desired_cutoff = round(math.ceil(max_bond * 10 + 1e-9) / 10, 2)

    # normalize_factors[0]: the divisor that lands the p95 radius near 1.0.  Floor of 2
    # per docs/04-phase0-measurements.md -- TERS molecules sit in an ~18 A field of view
    # and are larger than the QM9 set pos_norm=1 was tuned on.
    _pos_norm = max(2, int(round(p95_radius)))

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
        # normalize_factors[0]: pick the divisor that lands the p95 radius near 1.0.
        # Floor of 2 per the decision in docs/04-phase0-measurements.md -- TERS molecules
        # sit in an 18 A box and are larger than the QM9 set pos_norm=1 was tuned on.
        suggested_pos_norm=_pos_norm,
        # spatial_cut_off is compared against a squared distance (models/utils.py:118-126)
        # computed on the *normalised* coordinates -- utils.py:56 does `pos = pos / pos_norm`
        # before the model ever sees them.  So the threshold is in units of
        # (angstrom / pos_norm)**2, NOT angstrom**2.  Upstream QM9 hides this because its
        # pos_norm is 1, where the two coincide; at pos_norm=6 they differ by 36x.
        # Getting this wrong does not crash -- it silently makes `adj_spatial` all-ones,
        # i.e. a constant feature.  Measured on the real planar file: 2.56 (the angstrom**2
        # answer) marks 97.3% of all atom pairs adjacent; the value below marks 9.9% and
        # still keeps 100% of RDKit bonds.
        suggested_spatial_cut_off=round((desired_cutoff / _pos_norm) ** 2, 4),
        suggested_spatial_cut_off_angstrom=desired_cutoff,
        suggested_spatial_cut_off_note=(
            f'squared, in units of (A / pos_norm={_pos_norm}); '
            f'equals {desired_cutoff:.2f} A in real space'),
    )


def check_record(rec, expect_aromatic_free=True):
    """Assert an encoded record is well formed. Raises AssertionError with a reason.

    The last check is the important one: it pushes the record through the *real*
    ``EdgeComSpectraTransform`` and confirms the number of existing edges survives. That is
    the direct test for the aromatic trap (docs/05-progress-log.md D-4) -- a bond type 4
    reaching the transform is silently erased, so an aromatic molecule would come out of
    this check with fewer edges than it has bonds.
    """
    n = int(rec['num_atom'])
    ei, et = rec['edge_index'], rec['edge_type']

    assert rec['pos'].shape == (n, 3), f"pos {tuple(rec['pos'].shape)} != ({n}, 3)"
    assert rec['atom_type'].shape == (n,), 'atom_type length != num_atom'
    assert rec['fc'].shape == (n,), 'fc length != num_atom'
    assert int(rec['atom_type'].max()) < len(TYPES), 'atom_type index outside H/C/N/O'
    assert ei.shape[1] == et.shape[0], 'edge_index and edge_type disagree'
    assert ei.shape[1] % 2 == 0, 'odd edge count -- edges must be stored both directions'
    assert int(ei.max()) < n if ei.numel() else True, 'edge_index refers to a missing atom'

    if expect_aromatic_free:
        assert int(et.max()) <= 3 if et.numel() else True, \
            'bond type 4 (aromatic) survived kekulization -- it would be erased by ' \
            'EdgeComSpectraTransform (datasets/build_dataset.py:120-122)'

    # (row * N + col) ordering, per datasets/qm9s_dataset.py:258
    keys = (ei[0] * n + ei[1]).tolist()
    assert keys == sorted(keys), 'edges are not sorted by (row * N + col)'

    try:
        from torch_geometric.data import Data
        from datasets.build_dataset import EdgeComSpectraTransform
    except ImportError:
        return 'transform check skipped (torch_geometric unavailable)'

    data = Data(atom_type=rec['atom_type'], pos=rec['pos'], fc=rec['fc'],
                edge_index=ei, edge_type=et)
    # atom_type stores indices, so the list is the encoder's *values* --
    # matches the call site at datasets/build_dataset.py:24
    data = EdgeComSpectraTransform(list(range(len(TYPES))), include_aromatic=False)(data)

    n_exist = int(data.edge_one_hot[..., 0].sum())
    assert n_exist == ei.shape[1], (
        f'transform kept {n_exist} of {ei.shape[1]} directed edges -- bonds are being '
        f'silently erased')
    assert int(data.atom_one_hot.sum()) == n, 'atom one-hot does not cover every atom'
    return 'transform check passed'


def self_test():
    """Exercise the RDKit half end-to-end with no HDF5 and no cluster.

    Exists because the TERS data is not mounted where this was written
    (docs/05-progress-log.md D-1, D-2). Run it first on the cluster: if it fails, the
    problem is the environment or the port, not the data.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')

    # Benzene is the case that matters: it is the molecule the aromatic trap destroys.
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

    print(f'\n{note}')
    print('self-test: ' + ('all cases passed' if not failed else f'{failed} case(s) failed'))
    return failed


def process_split(h5_path, out_path, z_key=None, pos_key=None, charge=0,
                  allow_fallback=True, want_stats=False, check=True):
    """Build one sidecar. Returns a summary dict for the aggregate report."""
    import h5py
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')   # mirrors datasets/qm9s_dataset.py:198

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
                rec = encode_graph(mol, h5_index=idx)
                if check:
                    check_record(rec)
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

    # Decision rule from docs/04-phase0-measurements.md: below ~90% survival, the fallback
    # is doing too much work to trust bond orders across the corpus.
    if survival < 0.90:
        print('  WARNING: survival below 90%. Revisit the fallback / --charge before '
              'training -- see docs/04-phase0-measurements.md.')

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
    ap.add_argument('--no-fallback', action='store_true',
                    help='drop molecules DetermineBonds fails on instead of using '
                         'covalent-radius connectivity')
    ap.add_argument('--no-check', action='store_true',
                    help='skip the per-record well-formedness check (faster)')
    ap.add_argument('--stats', action='store_true',
                    help='also write the Phase 0 geometry rows as JSON')
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
                  check=not args.no_check)

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

    # The follow-on table in docs/04-phase0-measurements.md, ready to paste.
    print('--- Phase 0 follow-on (docs/04-phase0-measurements.md) ---')
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
            print(f'  model.normalize_factors[0]  = '
                  f'{max(g["suggested_pos_norm"] for g in geo)}')
            pn = max(g['suggested_pos_norm'] for g in geo)
            cut = max(g['suggested_spatial_cut_off_angstrom'] for g in geo)
            print(f'  model.spatial_cut_off       = {round((cut / pn) ** 2, 4)}'
                  f'   ({cut} A, SQUARED -- models/utils.py:118-126)')
            print(f'  data.max_node               = '
                  f'{max(g["n_atoms"]["max"] for g in geo)}')
            elements = sorted({z for g in geo for z in map(int, g['elements'])})
            print(f'  data.atom_types             = {len(elements)}   (Z = {elements})')


if __name__ == '__main__':
    main()
