"""Reconstruct the frozen model and training EMA layout for sampling/evaluation.

Use the same model settings and trainable_patterns as the training run.
The fine-tuning script installs a factory that handles frozen configs only.
"""
from configs.diffspectra_ters_frozen import get_config as _training_config


def get_config():
    config = _training_config()
    from main_finetune import configure_finetune_runtime
    configure_finetune_runtime(config)
    return config
