qm9_with_h = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 4, 5: 5, 6: 9, 7: 16, 8: 49, 9: 124, 10: 362, 11: 807, 12: 1689, 13: 3060, 14: 5136,
                        15: 7796, 16: 10644, 17: 13025, 18: 13364, 19: 13832, 20: 9482, 21: 9970, 22: 3393, 23: 4848,
                        24: 539, 25: 1506, 26: 48, 27: 266, 29: 25},
    'max_n_nodes': 29,
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                            'H1C-C1N-N1C', 'H1C-C1C-C1N'],
}


qm9_second_half = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 3, 5: 3, 6: 5, 7: 7, 8: 25, 9: 62, 10: 178, 11: 412, 12: 845, 13: 1541, 14: 2587,
                        15: 3865, 16: 5344, 17: 6461, 18: 6695, 19: 6944, 20: 4794, 21: 4962, 22: 1701, 23: 2380,
                        24: 267, 25: 754, 26: 17, 27: 132, 29: 15},
    'max_n_nodes': 29,
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                            'H1C-C1N-N1C', 'H1C-C1C-C1N'],
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
    'atom_fc_num': {"N1": 1181, "O-1": 1118, "C-1": 44, "N-1": 33, "C1": 11, "O1": 3},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C2C', 'C1N', 'C1O', 'H1N', 'C2O', 'C2N'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C2C', 'C2C-C1H', 'H1C-C1H', 'C1N-N1H', 'C1C-C1C', 'C1N-N1C', 'C1N-N2C'],
    'top_dihedral_sym': ['C1C-C2C-C1H', 'C2C-C1C-C1H', 'C1C-C2C-C1C', 'C2C-C1C-C2C', 'C1C-C1C-C2C', 'C1C-C1C-C1H', 'H1C-C1C-C1H', 'H1C-C2C-C1H'],
}

dataset_info_dict = {
    'qm9_with_h': qm9_with_h,
    'qm9_second_half': qm9_second_half,
    'ters_hcno': ters_hcno,
}


def get_dataset_info(info_name):
    return dataset_info_dict[info_name]
