"""Source QM9S setup: 64px cubes, selective pooling, published frozen denoiser."""
from configs.diffspectra_ters_frozen import get_config as _frozen_config
from env import data_root

QM9S_ROOT = data_root('QM9S-dataset', 'qm9s_hdf5_64px_5A')
# The rebuilt HDF5 row order requires v3; the original graphs/ is stale.
QM9S_GRAPHS = data_root('qm9s-ters', 'graphs_v3')


def get_config():
    config = _frozen_config()
    config.data.root = QM9S_ROOT
    config.data.graph_root = QM9S_GRAPHS
    config.data.info_name = 'qm9_second_half'
    config.data.atom_types = 5
    config.data.max_node = 29
    config.data.num_workers = 16
    config.model.ters_downsample = (False, True, True, False)
    config.model.ters_pool_size = 4
    config.training.n_iters = 60000 // config.training.num_gpus
    config.eval.begin_ckpt = config.eval.end_ckpt = 12
    config.eval.num_samples = 782
    config.training.wandb.group = 'qm9s-frozen'
    return config
