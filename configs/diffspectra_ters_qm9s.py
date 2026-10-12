"""DiffSpectra conditioned on QM9S TERS cubes of shape (64, 64, 100).

configs/diffspectra_ters.py with data-dependent overrides; pair with configs/base_ters_qm9s.py.
"""

from configs.diffspectra_ters import get_config as _ters_config
from env import data_root


QM9S_ROOT = data_root('QM9S-dataset', 'qm9s_hdf5_64px_5A')
QM9S_GRAPHS = data_root('qm9s-ters', 'graphs')


def get_config():
    config = _ters_config()
    data, model, training, evaluate = config.data, config.model, config.training, config.eval

    # {train,val,test}.h5; channels_100 has shape (N, 64, 64, 100).
    data.root = QM9S_ROOT
    # graphs_{split}.pt built from QM9S_ROOT; change the two together.
    data.graph_root = QM9S_GRAPHS
    # Upstream QM9 entry (H/C/N/O/F); its name also selects the QM9 stability checks.
    data.info_name = 'qm9_second_half'
    data.atom_types = 5                  # H/C/N/O/F; must match data.info_name
    data.max_node = 29                   # max atoms per molecule; not read

    # Same as upstream DiffSpectra; spatial_cut_off is a squared distance in A^2.
    model.normalize_factors = '1, 4, 4, 1'
    model.spatial_cut_off = 2.0

    # Keep n_iters a multiple of snapshot_freq and begin/end_ckpt = n_iters // snapshot_freq.
    training.n_iters = 150000 // training.num_gpus
    training.snapshot_freq = 25000 // training.num_gpus
    evaluate.begin_ckpt = 6
    evaluate.end_ckpt = 6

    evaluate.num_samples = 6279          # test molecules in graphs_test.pt; must not exceed it

    training.wandb.group = 'qm9s-ters'

    return config
