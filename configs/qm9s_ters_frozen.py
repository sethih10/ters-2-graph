"""QM9S TERS fine-tuning from the published DiffSpectra checkpoint; run with main_finetune.py."""
from configs.diffspectra_ters_frozen import get_config as _frozen_config
from configs.diffspectra_ters_qm9s import QM9S_GRAPHS, QM9S_ROOT


def get_config():
    config = _frozen_config()
    config.data.root = QM9S_ROOT
    config.data.graph_root = QM9S_GRAPHS
    config.data.info_name = 'qm9_second_half'
    config.data.atom_types = 5
    config.data.max_node = 29
    config.data.num_workers = 16
    config.training.n_iters = 60000 // config.training.num_gpus
    config.eval.begin_ckpt = config.eval.end_ckpt = 12
    config.eval.num_samples = 782
    config.training.wandb.group = 'qm9s-frozen'
    return config
