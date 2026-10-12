"""Continue a compatible end-to-end TERS p4 encoder with its denoiser frozen."""
from configs.diffspectra_ters_frozen import get_config as _frozen_config
from env import output_root


def get_config():
    config = _frozen_config()
    model = config.model
    # Override for your own compatible source run; do not reuse its work directory.
    model.pretrained_denoiser_path = output_root(
        'ters-runs', 'ters_20260904_1046657', 'checkpoints', 'checkpoint_5.pth')
    model.src_atom_decoder = ('H', 'C', 'N', 'O')
    model.load_pretrained_cond_encoder = True
    model.src_normalize_factors = model.normalize_factors = '7, 4, 4, 1'
    model.src_spatial_cut_off = model.spatial_cut_off = 0.0737
    config.optim.lr = 5e-5
    config.optim.warmup = 500
    return config
