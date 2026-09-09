qm9_with_h = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 4, 5: 5, 6: 9, 7: 16, 8: 49, 9: 124, 10: 362, 11: 807, 12: 1689, 13: 3060, 14: 5136,
                        15: 7796, 16: 10644, 17: 13025, 18: 13364, 19: 13832, 20: 9482, 21: 9970, 22: 3393, 23: 4848,
                        24: 539, 25: 1506, 26: 48, 27: 266, 29: 25},
    'max_n_nodes': 29,
    'atom_fc_num': {'O-1': 24063, 'N1': 14227, 'S1': 13590, 'S-1': 3234, 'N-1': 383,
                    'C-1': 302, 'O1': 129, 'S2': 21, 'C1': 17, 'Br-1': 10, 'Si-1': 7,
                    'C-2': 3},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['H1C', 'C1C', 'C2C', 'C1N', 'C1O', 'H1N', 'C1Cl', 'C2O'],
    'top_angle_sym': ['C1C-C1H', 'H1C-C1H', 'C2C-C1H', 'C2C-C1C', 'C1C-C2C', 'C1C-C1C', 'N1C-C1H',
                      'C1N-N1H'],
    'top_dihedral_sym': ['H1C-C1C-C1H', 'C1C-C1C-C1H', 'H1C-C1C-C1C', 'C1C-C2C-C1C', 'C2C-C1C-C2C',
                         'C2C-C1C-C1H', 'C1C-C2C-C1H', 'H1C-C1N-N1C'],
}


qm9_second_half = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 3, 5: 3, 6: 5, 7: 7, 8: 25, 9: 62, 10: 178, 11: 412, 12: 845, 13: 1541, 14: 2587,
                        15: 3865, 16: 5344, 17: 6461, 18: 6695, 19: 6944, 20: 4794, 21: 4962, 22: 1701, 23: 2380,
                        24: 267, 25: 754, 26: 17, 27: 132, 29: 15},
    'max_n_nodes': 29,
    'atom_fc_num': {'O-1': 24063, 'N1': 14227, 'S1': 13590, 'S-1': 3234, 'N-1': 383,
                    'C-1': 302, 'O1': 129, 'S2': 21, 'C1': 17, 'Br-1': 10, 'Si-1': 7,
                    'C-2': 3},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['H1C', 'C1C', 'C2C', 'C1N', 'C1O', 'H1N', 'C1Cl', 'C2O'],
    'top_angle_sym': ['C1C-C1H', 'H1C-C1H', 'C2C-C1H', 'C2C-C1C', 'C1C-C2C', 'C1C-C1C', 'N1C-C1H',
                      'C1N-N1H'],
    'top_dihedral_sym': ['H1C-C1C-C1H', 'C1C-C1C-C1H', 'H1C-C1C-C1C', 'C1C-C2C-C1C', 'C2C-C1C-C2C',
                         'C2C-C1C-C1H', 'C1C-C2C-C1H', 'H1C-C1N-N1C'],
    'prop2idx': {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11},
}


# Measured from the 782 real TERS molecules in `datasets/ters_real/graphs_test.pt`
# by `preprocess_ters.py` (element census pins the set to exactly H/C/N/O).
# `train_n_nodes` is the *test* split histogram - it is the only split that exists today,
# and the key is unused by the code (kept for parity with the QM9 entries).
ters_hcno = {
    'name': 'TERS',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3},
    'atom_decoder': ['H', 'C', 'N', 'O'],
    # Every field below is MEASURED on the 6254-molecule train split of planar_hdf5_0.5
    # (the earlier values came from the 782-molecule single-file dataset and are superseded).
    # Counted with this repo's own evaluation/cal_geometry.py symbol builders, so the strings
    # match what cal_bond_distance / cal_bond_angle / cal_dihedral_angle look for.
    'train_n_nodes': {
    2: 2, 3: 13, 4: 15, 5: 8, 6: 21, 7: 34, 8: 52, 9: 50, 10: 78, 11: 92, 12: 119, 13: 178,
    14: 236, 15: 292, 16: 348, 17: 379, 18: 484, 19: 508, 20: 306, 21: 304, 22: 290, 23: 235,
    24: 295, 25: 238, 26: 290, 27: 254, 28: 268, 29: 220, 30: 240, 31: 99, 32: 92, 33: 70,
    34: 83, 35: 28, 36: 26, 37: 2, 38: 4, 40: 1
    },
    'max_n_nodes': 40,
    'atom_fc_num': {'O-1': 24063, 'N1': 14227, 'S1': 13590, 'S-1': 3234, 'N-1': 383,
                    'C-1': 302, 'O1': 129, 'S2': 21, 'C1': 17, 'Br-1': 10, 'Si-1': 7,
                    'C-2': 3},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77],
    'top_bond_sym': ['H1C', 'C1C', 'C2C', 'C1N', 'C1O', 'H1N', 'C1Cl', 'C2O'],
    'top_angle_sym': ['C1C-C1H', 'H1C-C1H', 'C2C-C1H', 'C2C-C1C', 'C1C-C2C', 'C1C-C1C', 'N1C-C1H',
                      'C1N-N1H'],
    'top_dihedral_sym': ['H1C-C1C-C1H', 'C1C-C1C-C1H', 'H1C-C1C-C1C', 'C1C-C2C-C1C', 'C2C-C1C-C2C',
                         'C2C-C1C-C1H', 'C1C-C2C-C1H', 'H1C-C1N-N1C'],
}

dataset_info_dict = {
    'qm9_with_h': qm9_with_h,
    'qm9_second_half': qm9_second_half,
    'ters_hcno': ters_hcno,
}


# --- ADDED (stm): the STM corpus at /scratch/project_2001912/stm-data ------------------
# Every field below is MEASURED, by `python -m datasets.preprocess_stm`:
#   --census           over the raw xyz block (elements, sizes)
#   --stats            over the encoded sidecar (train_n_nodes, pos_norm, spatial_cut_off)
#   --geometry-symbols over the sidecar's rdmol objects, using evaluation/cal_geometry.py's
#                      OWN symbol builders -- a hand-written 'C1C' that does not match what
#                      get_bond_symbol emits scores an empty population and reports nothing,
#                      silently, so these must never be typed by hand.
#
# `name` is 'STM', not 'QM9'. That is deliberate and it has one consequence worth knowing:
# check_stability (evaluation/stability.py:48) raises 'Fail to get dataset bond info.' for
# any name that is not QM9/Geom. That 3D metric is unusable here anyway -- it re-infers bonds
# from interatomic distances, discarding the edges the model predicts -- and tools/sample_ters.py
# already uses check_2D_stability instead, which reads the predicted edges directly and does
# not branch on the name. All ten elements below are covered by evaluation/bond_analyze.py's
# `allowed_bonds` and `allowed_fc_bonds`, which is what check_2D_stability needs.
stm_10elem = {
    'name': 'STM',
    # MEASURED: Z = {1,6,7,8,9,14,15,16,17,35} over 20000 train molecules, no others.
    # Order matches datasets.preprocess_stm.STM_TYPES -- these two MUST agree, or the
    # one-hot index and the element symbol disagree with no error anywhere.
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
                     'Si': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br': 9},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br'],
    # MEASURED on the full 184157-molecule train split (median 23, p95 35).
    'train_n_nodes': {
    2: 171, 3: 110, 4: 366, 5: 500, 6: 575, 7: 833, 8: 1327, 9: 1533, 10: 1832, 11: 2922,
    12: 3548, 13: 3385, 14: 5663, 15: 6543, 16: 7256, 17: 8210, 18: 10025, 19: 8745, 20: 9230,
    21: 9601, 22: 9390, 23: 8869, 24: 9068, 25: 8807, 26: 8917, 27: 7931, 28: 8464, 29: 7250,
    30: 6236, 31: 5451, 32: 4974, 33: 3614, 34: 2932, 35: 2338, 36: 2048, 37: 1460, 38: 1334,
    39: 844, 40: 596, 41: 379, 42: 223, 43: 228, 44: 146, 45: 89, 46: 52, 47: 36, 48: 27,
    49: 21, 50: 17, 51: 21, 52: 12, 53: 4, 54: 4
    },
    # MEASURED: the xyz block is padded to 54 rows and 2 molecules in 40000 genuinely fill
    # it (both C18H36). The histogram tapers smoothly into 54 rather than piling up there,
    # so these are real molecules and not truncations.
    'max_n_nodes': 54,
    # MEASURED: RDKit's assigned formal charges over the train split. Consumed by
    # check_2D_stability (evaluation/stability.py:80) to decide which charged states are
    # legitimate here. S1/S-1 are the entries TERS does not have -- sulfoxides and thiolates.
    'atom_fc_num': {'O-1': 24063, 'N1': 14227, 'S1': 13590, 'S-1': 3234, 'N-1': 383,
                    'C-1': 302, 'O1': 129, 'S2': 21, 'C1': 17, 'Br-1': 10, 'Si-1': 7,
                    'C-2': 3},
    # 10 entries, one per element, matching atom_decoder order. Matplotlib colour specs; the
    # first is the translucent white QM9 uses for H.
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1', 'C8', 'C4', 'C5', 'C2', 'C6'],
    # Cordero (2008) covalent radii, the same values datasets/preprocess_ters._COVALENT_RADII
    # carries, in atom_decoder order.
    'radius_dic': [0.31, 0.76, 0.71, 0.66, 0.57, 1.11, 1.07, 1.05, 1.02, 1.20],
    'top_bond_sym': ['H1C', 'C1C', 'C2C', 'C1N', 'C1O', 'H1N', 'C1Cl', 'C2O'],
    'top_angle_sym': ['C1C-C1H', 'H1C-C1H', 'C2C-C1H', 'C2C-C1C', 'C1C-C2C', 'C1C-C1C', 'N1C-C1H',
                      'C1N-N1H'],
    'top_dihedral_sym': ['H1C-C1C-C1H', 'C1C-C1C-C1H', 'H1C-C1C-C1C', 'C1C-C2C-C1C', 'C2C-C1C-C2C',
                         'C2C-C1C-C1H', 'C1C-C2C-C1H', 'H1C-C1N-N1C'],
}


# --- ADDED (stm): register it. Placed after the dict literal above because stm_10elem is
# defined below it; get_dataset_info reads the dict at call time, so this is equivalent to
# having been a member of the literal.
dataset_info_dict['stm_10elem'] = stm_10elem


def get_dataset_info(info_name):
    try:
        return dataset_info_dict[info_name]
    except KeyError:
        raise KeyError(
            f'unknown data.info_name {info_name!r}. Known: '
            f'{sorted(dataset_info_dict)}') from None
