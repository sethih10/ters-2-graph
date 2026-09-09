"""Phase 3 -- conditioning encoder for TERS image cubes.

Drop-in replacement for ``SpecFormer`` as ``DMT.cond_encoder``.  Same contract, so nothing
in the denoiser moves: ``(B, C, H, W) -> (B, output_dim)``, consumed by ``cond_lin`` at
models/dmt.py:348-349.

Provenance (see docs/02-reuse-map.md 4, components 11 and 12):
  * encoder trunk  -- ADAPT of Multimodal-ters/src/models/models.py:84-133 (AttentionUNet):
                      keep the 1x1 stem and the ResBlock+MaxPool ladder, drop the decoder
                      ladder, the attention gates, the final resblock and the output conv
  * ``ResBlock``   -- ADAPT of Multimodal-ters/src/models/layers.py:6-34, with
                      ``BatchNorm2d`` -> ``GroupNorm`` (see the note on that below)
  * ``out_norm`` + ``reset_parameters()`` -- REUSE of the interface SpecFormer exposes at
                      models/specformer.py:67 and :70-73, so swapping the two encoders stays
                      a one-line change
  * pooling head, small-input guard -- NEW

.. warning::
   **The MT repository is not mounted here (docs/05-progress-log.md D-3), so the two ADAPT
   rows above were reconstructed from the specification recorded in docs/02-reuse-map.md 4 --
   filter widths, kernel sizes, stem width and ladder shape -- not by reading MT's source.**
   The shapes and the training behaviour are verified; exact layer-for-layer correspondence
   with MT is not.  When MT is available, diff this against ``models.py:84-133`` and
   ``layers.py:6-34`` before loading any MT checkpoint into it.

Checked first and rejected: ``models/layers.py`` in this repo has no 2D convolution at all --
it is entirely graph message passing plus Gaussian-basis and sinusoidal embeddings.
``Flatten_Head`` (models/specformer.py:428) is a flatten-plus-linear for token sequences, not
a spatial pooling head.  The conv stack has to come from MT.
"""

import torch
from torch import nn


def _groups_for(channels, preferred=8):
    """Largest group count <= ``preferred`` that divides ``channels``.

    ``GroupNorm`` requires ``channels % num_groups == 0``.  The default filter widths
    (64/128/256/512) are all divisible by 8, but the stem width is configurable, so pick a
    legal value rather than crashing on an odd channel count.
    """
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """ADAPT of Multimodal-ters/src/models/layers.py:6-34 (see the module warning).

    **``BatchNorm2d`` -> ``GroupNorm`` is a correctness fix, not tuning.** ``create_model``
    wraps the model in ``nn.DataParallel`` (models/utils.py:27), which computes BatchNorm
    statistics *per replica*.  At the planned ``batch_size=4`` across two GPUs that is
    BatchNorm over two samples, whose running statistics are noise.  GroupNorm does not
    depend on the batch axis at all, so it is unaffected by both the small batch and the
    replica split.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2                      # 'same' -- the ladder downsamples
                                                        # via MaxPool, not via the convs
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels)
        self.act = nn.ReLU(inplace=True)
        # Projection on the skip path only when the width changes, so a same-width block
        # stays a true identity shortcut.
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class TersEncoder(nn.Module):
    """TERS cube -> a single conditioning vector.

    Parameters
    ----------
    in_channels : int
        Spectral channels in the cube.  Comes from ``TersGraphDataset.in_channels``; with
        the ``channels_100`` decision (docs/05-progress-log.md) that is 100, not 400.
    output_dim : int
        Must equal ``config.model.nf`` -- ``cond_lin`` is ``Linear(hidden_dim, time_dim)``
        (models/dmt.py:262).
    pool_size : int
        Spatial grid kept by the head before flattening.  ``1`` is the original global
        average pool.  ``k > 1`` keeps a ``k x k`` map, so coarse spatial layout survives
        into the conditioning vector instead of being averaged away -- see
        docs/experimental.md (E-1).  Output width is unchanged either way, so nothing
        downstream of ``cond_lin`` moves; only the head's Linear grows, by
        ``filters[-1] * (pool_size^2 - 1) * output_dim`` parameters.
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

        # 1x1 stem: mixes the spectral channels at full spatial resolution before any
        # spatial convolution.  With 100 input channels this is where most of the input
        # dimensionality is absorbed.
        self.stem = nn.Conv2d(in_channels, filters[0], kernel_size=1)

        widths = [filters[0]] + list(filters)
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernel_size[i], n_groups=n_groups)
            for i in range(len(filters))
        ])
        self.pool = nn.MaxPool2d(2, 2)

        # AdaptiveAvgPool2d fixes the OUTPUT grid, so any H,W still works and the cube size
        # need not be known -- that property is what pool_size inherits.  At pool_size=1 this
        # is the original global average pool, which is spatially invariant: it reduces the
        # 256x256 map to one number per channel, so *where* a mode lights up cannot reach the
        # denoiser.  pool_size=k keeps a k x k grid instead.  See docs/experimental.md (E-1).
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(filters[-1] * pool_size * pool_size, output_dim),
        )
        # Mirrors SpecFormer.out_norm (models/specformer.py:67).  Keep it: cond_lin was
        # initialised for a LayerNorm-scaled input, and handing it raw pooled activations
        # instead changes the scale of the conditioning signal added to time_emb at
        # models/dmt.py:354.
        self.out_norm = nn.LayerNorm(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Matches the interface SpecFormer exposes at models/specformer.py:70-73."""
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
            # The real cube size is not known until the HDF5 is read (D-1), and four
            # unconditional 2x2 pools would take anything below 16x16 down to a 0-sized
            # dimension.  Stop pooling once there is nothing left to halve; AdaptiveAvgPool2d
            # in the head handles whatever spatial size survives.
            if min(x.shape[-2:]) >= 2:
                x = self.pool(x)
        return self.out_norm(self.head(x))