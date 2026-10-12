"""Published DiffSpectra denoiser, fresh trainable TERS encoder (HCNO)."""
from configs.diffspectra_ters import get_config as _ters_config
from env import data_root


def get_config():
    config = _ters_config()
    model = config.model
    # Sentinel: main_finetune maps this to DMT after installing its setup factory.
    # Ordinary main.py must not silently train an unfrozen model with this config.
    model.name = 'DMT_Finetune'
    model.frozen_denoiser = True
    model.pretrained_denoiser_path = data_root(
        'DiffSpectra', 'exp', 'allspectra_pretrained_specformer',
        'checkpoints', 'checkpoint_40.pth')
    model.pretrained_denoiser_use_ema = True
    model.src_atom_decoder = ('H', 'C', 'N', 'O', 'F')
    model.load_pretrained_cond_encoder = False
    model.load_pretrained_cond_lin = True
    model.trainable_patterns = ('cond_encoder.', 'cond_lin.')
    model.src_normalize_factors = model.normalize_factors = '1, 4, 4, 1'
    model.src_spatial_cut_off = model.spatial_cut_off = 2.0
    model.allow_input_scale_mismatch = False
    model.pretrained_ters_encoder_path = ''
    model.ters_encoder = 'conv2d'
    model.ters_kernel_size = (3, 3, 3, 3)
    model.ters_n_groups = 8
    model.ters_downsample = (True, True, True, True)
    return config
