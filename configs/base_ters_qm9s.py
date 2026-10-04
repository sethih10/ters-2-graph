"""Auxiliary QM9S-TERS config for `--config_original_qm9`.

configs/base_ters.py with the data paths and element set of configs/diffspectra_ters_qm9s.py.
"""

from configs.base_ters import get_config as _base_ters_config
from configs.diffspectra_ters_qm9s import QM9S_GRAPHS, QM9S_ROOT


def get_config():
    config = _base_ters_config()
    data = config.data

    data.root = QM9S_ROOT
    data.graph_root = QM9S_GRAPHS
    data.info_name = 'qm9_second_half'   # H/C/N/O/F
    data.atom_types = 5                  # must match the main config
    data.max_node = 29                   # max atoms per molecule; not read
    config.model.normalize_factors = '5, 4, 4, 1'   # not read on this path
    config.eval.num_samples = 6279

    return config
