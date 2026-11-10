"""QM9S TERS from scratch with the token encoder and cross-attention adapters; run with main.py."""
from configs.diffspectra_ters_qm9s import get_config as _base_config
from configs.ters_tokens import add_token_xattn


def get_config():
    config = add_token_xattn(_base_config())
    config.training.wandb.group = 'qm9s-ters-token-xattn'
    return config
