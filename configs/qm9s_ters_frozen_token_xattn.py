"""QM9S TERS fine-tuning with the token encoder and cross-attention adapters; run with main_finetune.py."""
from configs.qm9s_ters_frozen import get_config as _base_config
from configs.ters_tokens import add_token_xattn


def get_config():
    config = add_token_xattn(_base_config())
    config.model.freeze_denoiser = True  # False also trains the pretrained denoiser
    config.training.wandb.group = 'qm9s-frozen-token-xattn'
    return config
