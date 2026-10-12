"""Fine-tune a pretrained DMT on TERS cubes with its denoiser frozen.

Loads the checkpoint's EMA weights (except cond_encoder, unless
model.load_pretrained_cond_encoder) and trains only cond_encoder and cond_lin.
Use this script for evaluation too, so the EMA covers the same parameters.

Usage:
    python main_finetune.py --config configs/diffspectra_ters_frozen.py --workdir RUN --mode train
    python main_finetune.py --config configs/diffspectra_ters_frozen.py --workdir RUN --mode eval
"""

import torch
from absl import app

import main
from models import utils as model_utils
from models.dmt import DMT

TRAINABLE = ('cond_encoder.', 'cond_lin.')
# Buffers in a DiffSpectra checkpoint; they have no EMA copy.
BUFFERS = ('running_mean', 'running_var', 'num_batches_tracked', 'sdp_attn.scale')


def load_ema_weights(path):
    """Return the checkpoint's EMA weights keyed by parameter name."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('module.'): v for k, v in ckpt['model'].items()}
    names = [k for k in weights if not k.endswith(BUFFERS)]
    shadows = ckpt['ema']['shadow_params']
    if len(names) != len(shadows) or any(weights[n].shape != s.shape
                                         for n, s in zip(names, shadows)):
        raise ValueError(f'{path}: EMA tensors do not match the model parameters')
    weights.update(zip(names, shadows))
    return weights


def drop_atom_types(weights, n_keep, fc):
    """Keep the first n_keep atom types (e.g. HCNOF -> HCNO) in the atom-sized tensors."""
    n_src = weights['node_pred_mlp.4.bias'].shape[0] - fc
    if n_keep > n_src:
        raise ValueError(f'checkpoint has {n_src} atom types, config needs {n_keep}')
    keep = list(range(n_keep)) + [n_src] * fc    # the formal-charge channel follows the atom types
    block = n_src + fc                           # node_emb input: [atoms, self-cond atoms]
    weights['node_emb.weight'] = weights['node_emb.weight'][:, keep + [block + i for i in keep]]
    weights['node_pred_mlp.4.weight'] = weights['node_pred_mlp.4.weight'][keep]
    weights['node_pred_mlp.4.bias'] = weights['node_pred_mlp.4.bias'][keep]


def build_model(config):
    model = DMT(config)
    weights = load_ema_weights(config.model.pretrained_denoiser_path)
    drop_atom_types(weights, config.data.atom_types, int(config.model.include_fc_charge))

    skip = () if config.model.load_pretrained_cond_encoder else ('cond_encoder.',)
    state = model.state_dict()
    state.update({k: weights[k] for k in state if not k.startswith(skip)})
    model.load_state_dict(state)

    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE))
    return model


def run(argv):
    model_utils._MODELS['DMT'] = build_model
    main.main(argv)


if __name__ == '__main__':
    app.run(run)
