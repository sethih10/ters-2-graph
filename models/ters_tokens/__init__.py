"""TERS token encoders, selected with config.model.ters_encoder. See README.md in this folder."""

import torch
from ml_collections import ConfigDict
from torch import nn

from .adapters import TersTokenXAttn, attach
from .encoder import TersTokenEncoder

ENCODERS = ('token', 'token_xattn')


def load_trunk(trunk: TersTokenEncoder, path: str) -> None:
    """Copy the encoder weights of an earlier `token` or `token_xattn` run into `trunk`."""
    saved = torch.load(path, map_location='cpu', weights_only=False)['model']
    saved = {k.removeprefix('module.'): v for k, v in saved.items()}
    is_xattn = any(k.startswith('cond_encoder.trunk.') for k in saved)
    prefix = 'cond_encoder.trunk.' if is_xattn else 'cond_encoder.'
    trunk.load_state_dict({k: saved[prefix + k] for k in trunk.state_dict()})


def build_encoder(config: ConfigDict, dmt: nn.Module) -> nn.Module:
    """Build the encoder named by config.model.ters_encoder; `token_xattn` also hooks into `dmt`."""
    m = config.model
    if m.ters_encoder not in ENCODERS:
        raise ValueError(f"model.ters_encoder must be 'v1' or one of {ENCODERS}, got {m.ters_encoder!r}")
    trunk = TersTokenEncoder(m.ters_in_channels, m.nf, m.ters_width, m.ters_spec_dim,
                             m.ters_tok_layers, m.ters_tok_heads, m.ters_cond_drop)
    if m.ters_encoder_init:
        load_trunk(trunk, m.ters_encoder_init)
    if m.ters_encoder == 'token':
        return trunk
    block_ids = list(range(0, m.n_layers, m.ters_adapter_every_k))
    attach(dmt, block_ids)
    return TersTokenXAttn(trunk, m.nf, block_ids, m.ters_adapter_heads)
