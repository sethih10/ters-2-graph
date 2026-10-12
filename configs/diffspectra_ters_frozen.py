"""Planar TERS fine-tuning from the published DiffSpectra checkpoint; run with main_finetune.py."""
from configs.diffspectra_ters import get_config as _ters_config
from env import data_root


def get_config():
    config = _ters_config()
    model = config.model
    model.pretrained_denoiser_path = data_root(
        'DiffSpectra', 'exp', 'allspectra_pretrained_specformer',
        'checkpoints', 'checkpoint_40.pth')
    model.load_pretrained_cond_encoder = False   # True continues the checkpoint's TERS encoder
    model.dropout = 0.0                          # dropout only acts inside the frozen denoiser
    return config
