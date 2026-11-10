"""Turn a frozen-run checkpoint into a standard one: EMA weights merged in, EMA list covering
every parameter, optimizer state dropped. The result works with sample.py and as
model.pretrained_denoiser_path.

    python -m models.ters_tokens.export RUN/checkpoints/checkpoint_12.pth OUT.pth
"""

import argparse

import torch

# Same values as in main_finetune.py, which cannot be imported without the full training stack.
TRAINABLE = ('cond_encoder.', 'cond_lin.')
BUFFERS = ('running_mean', 'running_var', 'num_batches_tracked', 'sdp_attn.scale')


def export(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    weights, shadow = dict(ckpt['model']), ckpt['ema']['shadow_params']
    params = [k for k in weights if not k.endswith(BUFFERS)]
    trained = [k for k in params if k.removeprefix('module.').startswith(TRAINABLE)]
    if len(shadow) != len(trained) or any(weights[k].shape != s.shape for k, s in zip(trained, shadow)):
        raise ValueError(f'{src} is not a frozen-run checkpoint: its EMA does not match cond_encoder + cond_lin')
    weights.update(zip(trained, shadow))
    ema = {**ckpt['ema'], 'shadow_params': [weights[k].clone() for k in params]}
    torch.save({'model': weights, 'ema': ema, 'step': ckpt['step']}, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()
    export(args.src, args.dst)
