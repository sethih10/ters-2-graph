"""DiffSpectra conditioned on QM9S STM images: 11 constant-current setpoints of 64 x 64 pixels.

The QM9S-TERS config with the STM images; pair with ``--config_original_qm9 configs/base_stm_qm9.py``.
Training uses one random setpoint per sample and evaluation setpoint 0 (tip closest), so the encoder
has one input channel.
"""

from configs.diffspectra_ters_qm9s import get_config as _ters_config


def get_config():
    config = _ters_config()
    config.data.modality = 'stm'
    config.data.normalize = 'corpus'          # one mean/std for every setpoint keeps their differences
    config.data.transforms.single_slice = True
    config.model.ters_in_channels = 1
    config.training.wandb.group = 'qm9-stm'
    return config
