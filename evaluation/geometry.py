"""Check predicted molecular graphs against their coordinates in angstroms.

Adapted from maabuu/posebusters, modules/distance_geometry.py (check_geometry).
https://github.com/maabuu/posebusters/tree/5fa994f92ffe50a2bcdd2b9665057c8b70576834
Uses RDKit directly, includes explicit hydrogens, and reuses the existing valence check.
"""

from functools import partial
import math

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom
import torch

from .stability import check_2D_stability


CHECKS = ('valence_valid', 'chemical_valid', 'bond_lengths_valid',
          'angle_distances_valid', 'no_clashes', 'geometry_valid')
BOUND_PARAMETERS = dict(set15bounds=True, scaleVDW=True,
                        doTriangleSmoothing=True, useMacrocycle14config=False)
DEFAULT_TOLERANCES = dict(bond_tolerance=0.25, angle_tolerance=0.25,
                          clash_tolerance=0.30)


def _validate_tolerances(tolerances):
    for name, value in tolerances.items():
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'{name} must be finite and in [0, 1), got {value}')


def _terminal_bounds(mol, lower, upper):
    # Resonance-equivalent terminal N/O bonds share a range, as in PoseBusters.
    query = Chem.MolFromSmarts(
        '[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]')
    groups = {}
    for a, b in mol.GetSubstructMatches(query):
        pair = tuple(sorted((a, b)))
        elements = tuple(sorted((mol.GetAtomWithIdx(a).GetSymbol(),
                                 mol.GetAtomWithIdx(b).GetSymbol())))
        groups.setdefault(elements, set()).add(pair)
    for pairs in groups.values():
        lo = min(lower[a, b] for a, b in pairs)
        hi = max(upper[a, b] for a, b in pairs)
        for a, b in pairs:
            lower[a, b] = lower[b, a] = lo
            upper[a, b] = upper[b, a] = hi


def check_geometry(mol, bond_tolerance=0.25, angle_tolerance=0.25,
                   clash_tolerance=0.30):
    """Check bond, angle-endpoint and nonbonded distances on a sanitized graph.

    Tolerances are fractions of RDKit bounds. Coordinates and bounds are angstroms.
    """
    tolerances = dict(bond_tolerance=bond_tolerance, angle_tolerance=angle_tolerance,
                      clash_tolerance=clash_tolerance)
    _validate_tolerances(tolerances)
    result = dict(chemical_valid=False, bond_lengths_valid=False,
                  angle_distances_valid=False, no_clashes=False,
                  geometry_valid=False, error=None)
    mol = Chem.Mol(mol)
    if not mol.GetNumAtoms() or not mol.GetNumConformers():
        result['error'] = 'missing_atoms_or_coordinates'
        return result
    positions = np.asarray(mol.GetConformer().GetPositions())
    if not np.isfinite(positions).all():
        result['error'] = 'nonfinite_coordinates'
        return result
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        result['error'] = 'sanitization_failed'
        return result
    result['chemical_valid'] = True
    try:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol, **BOUND_PARAMETERS)
    except (ValueError, RuntimeError):
        result['error'] = 'bounds_failed'
        return result
    # RDKit stores lower bounds below the diagonal and upper bounds above it.
    lower = np.tril(bounds, -1) + np.tril(bounds, -1).T
    upper = np.triu(bounds, 1) + np.triu(bounds, 1).T
    _terminal_bounds(mol, lower, upper)
    bonded = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
              for b in mol.GetBonds()}
    angles = set()
    for atom in mol.GetAtoms():
        neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
        for i, a in enumerate(neighbors):
            for b in neighbors[i + 1:]:
                angles.add(tuple(sorted((a, b))))
    counts = dict(n_bonds=0, n_bad_bonds=0, n_angles=0, n_bad_angles=0,
                  n_nonbonded=0, n_clashes=0)
    for a in range(mol.GetNumAtoms()):
        for b in range(a + 1, mol.GetNumAtoms()):
            distance = float(np.linalg.norm(positions[a] - positions[b]))
            lo, hi = lower[a, b], upper[a, b]
            if not np.isfinite([distance, lo, hi]).all() or lo < 0 or hi < lo:
                result['error'] = 'invalid_distance_bounds'
                return result
            if (a, b) in bonded:
                counts['n_bonds'] += 1
                counts['n_bad_bonds'] += int(
                    distance < (1 - bond_tolerance) * lo or
                    distance > (1 + bond_tolerance) * hi)
            elif (a, b) in angles:
                counts['n_angles'] += 1
                counts['n_bad_angles'] += int(
                    distance < (1 - angle_tolerance) * lo or
                    distance > (1 + angle_tolerance) * hi)
            else:
                counts['n_nonbonded'] += 1
                counts['n_clashes'] += int(distance < (1 - clash_tolerance) * lo)
    result.update(counts)
    result.update(bond_lengths_valid=counts['n_bad_bonds'] == 0,
                  angle_distances_valid=counts['n_bad_angles'] == 0,
                  no_clashes=counts['n_clashes'] == 0)
    result['geometry_valid'] = all(result[k] for k in
                                   ('chemical_valid', 'bond_lengths_valid',
                                    'angle_distances_valid', 'no_clashes'))
    return result


def check_3d_geometry(processed, dataset_info, **tolerances):
    """Check one (positions, atom types, predicted bonds, formal charges) tuple."""
    _validate_tolerances(tolerances)
    pos, atoms, edges, charges = [torch.as_tensor(x).detach().cpu() for x in processed]
    n = atoms.numel()
    if atoms.shape != (n,) or pos.shape != (n, 3) or edges.shape != (n, n):
        raise ValueError('expected positions (N, 3), atom types (N,), bonds (N, N)')
    if charges.numel() == 0:
        charges = torch.zeros(n, dtype=torch.long)
    if charges.shape != (n,):
        raise ValueError('expected formal charges (N,) or an empty tensor')
    decoder = dataset_info['atom_decoder']
    for values, name in ((atoms, 'atom types'), (edges, 'bond types'), (charges, 'charges')):
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError(f'{name} must contain finite integers')
    if (atoms < 0).any() or (atoms >= len(decoder)).any():
        raise ValueError('atom type outside atom_decoder')
    if (edges < 0).any() or (edges > 4).any():
        raise ValueError('bond types must be 0, 1, 2, 3 or 4 (aromatic)')
    if not torch.equal(edges, edges.T) or torch.any(edges.diag() != 0):
        raise ValueError('bond matrix must be symmetric with a zero diagonal')
    if n == 0 or not torch.isfinite(pos).all():
        return dict.fromkeys(CHECKS, False) | {'error': 'empty_or_nonfinite_coordinates'}
    valence, _, _, mol = check_2D_stability(
        pos.double(), atoms.long(), charges.long(), edges.long(), dataset_info)
    # Preserve every predicted charge, including states omitted by the legacy allow-list.
    for atom, charge in zip(mol.GetAtoms(), charges):
        atom.SetFormalCharge(int(charge))
        atom.SetNoImplicit(True)
    result = check_geometry(mol, **tolerances)
    result['valence_valid'] = bool(valence)
    result['geometry_valid'] = result['geometry_valid'] and result['valence_valid']
    return result


def get_geometry_metrics(processed, dataset_info, workers=1, **tolerances):
    """Return molecule pass fractions; failures remain in every denominator."""
    parameters = DEFAULT_TOLERANCES | tolerances
    _validate_tolerances(parameters)
    if workers < 1:
        raise ValueError('workers must be positive')
    if not len(processed):
        raise ValueError('cannot score an empty molecule collection')
    check = partial(check_3d_geometry, dataset_info=dataset_info, **parameters)
    counts = dict.fromkeys(CHECKS, 0)
    failures = {}
    pair_counts = dict.fromkeys(('n_bonds', 'n_bad_bonds', 'n_angles', 'n_bad_angles',
                                'n_nonbonded', 'n_clashes'), 0)

    def collect(results):
        for result in results:
            for key in counts:
                counts[key] += int(result[key])
            for key in pair_counts:
                pair_counts[key] += result.get(key, 0)
            if result['error']:
                failures[result['error']] = failures.get(result['error'], 0) + 1

    if workers == 1 or len(processed) < 2 * workers:
        collect(map(check, processed))
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            collect(pool.imap(check, processed, chunksize=max(1, len(processed) // (workers * 4))))
    n = len(processed)
    return {**{key: value / n for key, value in counts.items()},
            'n_molecules': n, 'pass_counts': counts, 'pair_counts': pair_counts,
            'failures': failures, 'parameters': parameters | {'include_hydrogens': True,
                                          'bounds': BOUND_PARAMETERS.copy()},
            'rdkit_version': rdBase.rdkitVersion}
