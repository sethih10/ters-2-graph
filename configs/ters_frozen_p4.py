"""Continue an end-to-end planar TERS run with its denoiser frozen; run with main_finetune.py."""
from configs.diffspectra_ters_frozen import get_config as _frozen_config
from env import output_root


def get_config():
    config = _frozen_config()
    model = config.model
    # Do not reuse the source run's work directory.
    model.pretrained_denoiser_path = output_root(
        'ters-runs', 'ters_20260904_1046657', 'checkpoints', 'checkpoint_5.pth')
    model.load_pretrained_cond_encoder = True
    # The source run was trained with these; a frozen denoiser needs the same scale.
    model.normalize_factors = '7, 4, 4, 1'
    model.spatial_cut_off = 0.0737
    config.optim.lr = 5e-5
    config.optim.warmup = 500
    return config
