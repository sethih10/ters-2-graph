"""Config keys of the token encoders (models/ters_tokens), shared by the *_token*.py configs."""
from ml_collections import ConfigDict


def add_token_encoder(config: ConfigDict) -> ConfigDict:
    model = config.model
    model.ters_encoder = 'token'
    model.ters_width = 32                # tokens have 8 * width dims
    model.ters_spec_dim = 128            # per-pixel spectral MLP width
    model.ters_tok_layers = 2
    model.ters_tok_heads = 4
    model.ters_cond_drop = 0.1           # dropout of the conditioning vector
    model.ters_encoder_init = ''         # checkpoint of an earlier token / token_xattn run
    return config


def add_token_xattn(config: ConfigDict) -> ConfigDict:
    model = add_token_encoder(config).model
    model.ters_encoder = 'token_xattn'
    model.ters_adapter_heads = 4
    model.ters_adapter_every_k = 1       # 1 = every DMT block, 2 = every other block
    return config
