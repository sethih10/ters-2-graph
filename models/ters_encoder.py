"""Convolutional encoder for TERS image cubes.

Maps a cube ``(B, C, H, W)`` to ``(B, output_dim)``; used as ``DMT.cond_encoder``. Submodule
names and shapes follow the encoder path of ``AttentionUNet`` so that
``DMT.load_pretrained_ters_encoder`` can load its weights.
"""

import torch
from torch import nn


def _groups_for(channels, preferred=8):
    """Return the largest group count <= ``preferred`` that divides ``channels``."""
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Residual block: conv-norm-ReLU, conv-norm, skip connection, ReLU.

    GroupNorm rather than BatchNorm because ``nn.DataParallel`` splits small batches across
    replicas. Parameter names are checkpoint keys; do not rename them.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class TersEncoder(nn.Module):
    """Encode a TERS cube ``(B, C, H, W)`` into ``(B, output_dim)``.

    A 1x1 stem, residual stages with max pooling and a pooled linear head.

    Parameters
    ----------
    in_channels : int
        Number of spectral channels; must match ``TersGraphDataset.in_channels``.
    output_dim : int
        Embedding size; must equal ``config.model.nf``.
    pool_size : int
        Spatial grid size before the head; 1 is global average pooling.
    """

    def __init__(self, in_channels=100, filters=(64, 128, 256, 512),
                 kernel_size=(3, 3, 3, 3), output_dim=256, n_groups=8, pool_size=1):
        super().__init__()
        if len(kernel_size) != len(filters):
            raise ValueError(
                f'filters and kernel_size must be the same length, '
                f'got {len(filters)} and {len(kernel_size)}')

        if pool_size < 1:
            raise ValueError(f'pool_size must be >= 1, got {pool_size}')

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.pool_size = pool_size

        self.stem = nn.Conv2d(in_channels, filters[0], kernel_size=1)

        widths = [filters[0]] + list(filters)
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernel_size[i], n_groups=n_groups)
            for i in range(len(filters))
        ])
        self.pool = nn.MaxPool2d(2, 2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(filters[-1] * pool_size * pool_size, output_dim),
        )
        self.out_norm = nn.LayerNorm(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'TersEncoder expects (B, C, H, W), got {tuple(x.shape)}. The cube is '
                f'permuted to channels-first in TersGraphDataset.__getitem__.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'TersEncoder built for in_channels={self.in_channels} but got '
                f'{x.size(1)}. Set config.model.ters_in_channels to match '
                f'TersGraphDataset.in_channels.')

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            # Stop pooling once a side is below 2, so small inputs do not shrink to size 0.
            if min(x.shape[-2:]) >= 2:
                x = self.pool(x)
        return self.out_norm(self.head(x))