"""DiffSpectra conditioned on QM9S TERS mode maps through the channel-fusion ViT encoder.

Pair with ``--config_original_qm9 configs/base_ters_qm9s.py``.
"""

from configs.diffspectra_ters_qm9s import get_config as _ters_config


def get_config():
    config = _ters_config()
    config.data.bin_modes = False             # one map per normal mode, with its wavenumber
    config.data.normalize = 'log_relative'
    config.model.ters_encoder = 'mode_fusion'
    config.training.wandb.group = 'qm9s-fusion'
    return config
