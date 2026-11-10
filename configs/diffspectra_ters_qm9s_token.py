"""QM9S TERS from scratch with the token encoder (models/ters_tokens); run with main.py."""
from configs.diffspectra_ters_qm9s import get_config as _base_config
from configs.ters_tokens import add_token_encoder


def get_config():
    config = add_token_encoder(_base_config())
    config.training.wandb.group = 'qm9s-ters-token'
    return config
