import numpy as np
from rdkit import Chem
from .bond_analyze import get_bond_order, geom_predictor, allowed_bonds, allowed_fc_bonds
from .rdkit_metric import eval_rdmol
from rdkit.Geometry import Point3D
import copy
import torch


bond_list = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
stability_bonds = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3,
                   Chem.rdchem.BondType.AROMATIC: 1.5}


# Stability and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    """Look up for bond types and construct a Rdkit Mol"""
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    # atoms
    # convert to RDKit Mol, add atom first
    mol = Chem.RWMol()
    for atom in atom_type:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if 'QM9' in dataset_info['name']:
                order = get_bond_order(atom1, atom2, dist)
            elif 'Geom' in dataset_info['name']:
                order = geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            # --- ADDED (eval): the image-cube corpora. Purely additive -- the two branches
            # above and the raise below are untouched, so the QM9S path is unaffected.
            #
            # `bonds1` already tabulates every element these corpora contain: ters_hcno is
            # H/C/N/O, stm_10elem is H,C,N,O,F,Si,P,S,Cl,Br (bond_analyze.py:5-35).
            # check_exists=True returns 0 for a pair with no tabulated length instead of
            # raising KeyError, so an unexpected element degrades to "no bond" rather than
            # killing an eval run that already cost a full ancestral sampling pass.
            #
            # CAVEAT, and it must travel with every number this produces: margin1/2/3 were
            # tuned to maximise the stability of *QM9 true samples* (bond_analyze.py:119-120),
            # i.e. on H/C/N/O/F only. For Si, P, S, Cl and Br the thresholds are uncalibrated,
            # so 3D stability here is a RELATIVE signal across checkpoints of the same corpus
            # and is NOT comparable to a published QM9 stability figure. The 2D metric stays
            # authoritative: these models set pred_edge=True, so bond orders are predicted by
            # the network and check_2D_stability reads them directly rather than re-deriving
            # them from geometry. Falsifier for the caveat, and the way to retire it: run this
            # over GROUND-TRUTH structures from the train split; if their 3D stability is not
            # near 1.0, the margins are wrong for this corpus.
            elif dataset_info['name'] in ('TERS', 'STM'):
                order = get_bond_order(atom1, atom2, dist, check_exists=True)
            else:
                raise ValueError('Fail to get dataset bond info.')
            nr_bonds[i] += order
            nr_bonds[j] += order
            # add bond to RDKIT Mol
            if order > 0:
                mol.AddBond(i, j, bond_list[order])

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x), mol


def check_2D_stability(positions, atom_types, formal_charges, edge_types, dataset_info):
    """Convert the generated tensors to rdkit mols and check stability"""
    dataset_name = dataset_info['name']
    atom_decoder = dataset_info['atom_decoder']
    if 'atom_fc_num' in dataset_info:
        atom_fcs = dataset_info['atom_fc_num']
    else:
        atom_fcs = {}
    atom_num = atom_types.size(0)

    # convert to rdkit mol
    # atoms
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    # add formal charges to Mol
    if formal_charges.shape[-1] == 0:
        formal_charges = torch.zeros_like(atom_types)

    for atom_id, fc in enumerate(formal_charges):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_str = atom.GetSymbol()
        if fc != 0:
            atom_fc = atom_str + str(fc.item())
            # print('formal charges: ' + atom_fc)
            if atom_fc in atom_fcs:
                # print('add formal charges, ' + atom_fc)
                atom.SetFormalCharge(fc.item())

    # add positions to Mol
    if positions is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)

    # add bonds
    edge_index = torch.nonzero(edge_types)
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i]
        if src < dst:
            order = edge_types[src, dst]
            mol.AddBond(src.item(), dst.item(), bond_list[int(order)])

    # --- ADDED (eval): closes D-11 (docs/05-progress-log.md:173-183). 'TERS' and 'STM' fell
    # through to this early return, so every generation log to date printed
    # `mol_stable 0.0, atom_stable 0.0` -- an early return, NOT a score, which is why those
    # zeros coexisted with RDKit `Validity 0.96` on the same molecules. Verified on two
    # hand-built stable molecules (CH3Cl, CH3Br) through mol_process: (0, 0, 5) before this
    # line changed, (True, 5, 5) after.
    #
    # Nothing else is needed to make the block below correct for these corpora. It reads
    # `allowed_fc_bonds` (bond_analyze.py:97-107), which already covers every element of
    # stm_10elem -- H, C, N, O, F, Si, P, S, Cl, Br -- including the S and Br formal-charge
    # states that stm_10elem['atom_fc_num'] records. ters_hcno is H/C/N/O, a subset.
    #
    # This is the allow-list fix docs/05-progress-log.md:181 proposes, taken in preference to
    # the other option there (setting `'name': 'QM9'` in ters_hcno / stm_10elem). That
    # alternative would also silently route check_stability's 3D dispatch above into the QM9
    # branch, making the TERS/STM branch unreachable and re-hiding the margin caveat.
    #
    # Additive: 'GeomDrug' and 'QM9' behaviour is unchanged, and anything else still returns
    # the zeros rather than a wrong number.
    if dataset_name not in ['GeomDrug', 'QM9', 'TERS', 'STM']:
        return 0, 0, atom_num, mol

    # kekulize mol and iterate bonds for stability
    new_mol = copy.deepcopy(mol)
    try:
        Chem.Kekulize(new_mol)
    except:
        # print('Can\'t Kekulize mol.')
        pass

    nr_bonds = np.zeros(atom_num, dtype='int')
    for bond in new_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # if bond_type in stability_bonds:
        #     order = stability_bonds[bond_type]
        # else:
        #     order = 0
        order = stability_bonds[bond_type]
        nr_bonds[start] += order
        nr_bonds[end] += order

    # stability
    nr_stable_bonds = 0
    atom_types_str = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for atom_type_i, nr_bonds_i, fc_i in zip(atom_types_str, nr_bonds, formal_charges):
        fc_i = fc_i.item()
        possible_bonds = allowed_fc_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[fc_i] if fc_i in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == nr_bonds_i if type(expected_bonds) == int else nr_bonds_i in expected_bonds
        else:
            is_stable = nr_bonds_i in possible_bonds
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == atom_num
    return molecule_stable, nr_stable_bonds, atom_num, mol


def get_edm_metric(dataset_info, train_mols=None):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    def edm_metric(processed_list):
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0

        rd_mols = []
        for mol in processed_list:
            pos, atom_type = mol[:2]
            validity_res = check_stability(pos, atom_type, dataset_info)

            molecule_stable += int(validity_res[0])
            nr_stable_bonds += int(validity_res[1])
            n_atoms += int(validity_res[2])
            rd_mols.append(validity_res[3])

        # Stability
        fraction_mol_stable = molecule_stable / float(len(processed_list))
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {
            'mol_stable': fraction_mol_stable,
            'atom_stable': fraction_atm_stable,
        }

        # Basic rdkit metric result (Validity, Fragment, Unique)
        rdkit_dict = eval_rdmol(rd_mols, train_smiles)
        return stability_dict, rdkit_dict, rd_mols

    return edm_metric


def get_2D_edm_metric(dataset_info, train_mols=None):
    train_smiles = None
    if train_mols is not None:
        train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    def edm_metric_2D(processed_list):
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0

        rd_mols = []
        for mol in processed_list:
            pos, atom_types, edge_types, fc = mol
            validity_res = check_2D_stability(pos, atom_types, fc, edge_types, dataset_info)
            molecule_stable += int(validity_res[0])
            nr_stable_bonds += int(validity_res[1])
            n_atoms += int(validity_res[2])
            rd_mols.append(validity_res[3])

        # Stability
        fraction_mol_stable = molecule_stable / float(len(processed_list))
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {
            'mol_stable': fraction_mol_stable,
            'atom_stable': fraction_atm_stable,
        }

        # Basic rdkit metric result (Validity, Fragment, Unique)
        rdkit_dict = eval_rdmol(rd_mols, train_smiles)
        return stability_dict, rdkit_dict, rd_mols

    return edm_metric_2D
