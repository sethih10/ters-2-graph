"""3D convolutional encoder for STM z-stacks.

Maps a z-stack ``(B, C, H, W)``, C = z-slices, to ``(B, output_dim)``. Selected with
``config.model.stm_encoder = 'v1'`` (the default). Inputs must be normalised with
``NormalizeStmCube``.

3D stem adapted from ml-spm ``ASDAFMNet`` (mlspm/image/models.py; Alldritt et al., Sci. Adv. 6,
eaay6913, 2020); its decoder is replaced by a 2D residual trunk and pooled head.

Usage
-----
    python -m models.stm_encoder --self-test
    python -m models.stm_encoder --inspect H5
    python -m models.stm_encoder --check H5
"""

import torch
import torch.nn.functional as F
from torch import nn

from .ters_encoder import ResBlock, _groups_for


class NormalizeStmCube:
    """Normalise one STM z-stack: divide, asinh-compress, then standardise over the whole cube.

    Parameters
    ----------
    uint16_max : float
        Divisor applied first; ``1.0`` if the cube is already float.
    asinh_scale : float or None
        ``s`` in ``asinh(x / s)``; ``None`` disables the compression.
    eps : float
        If the cube's standard deviation is below this, it is only mean-centred.
    """

    def __init__(self, uint16_max=65535.0, asinh_scale=0.01, eps=1e-6):
        self.uint16_max = uint16_max
        self.asinh_scale = asinh_scale
        self.eps = eps

    def __call__(self, cube):
        x = cube.float()
        if self.uint16_max and self.uint16_max != 1.0:
            x = x / self.uint16_max
        if self.asinh_scale:
            x = torch.asinh(x / self.asinh_scale)
        mean = x.mean()
        std = x.std()
        if std < self.eps:
            return x - mean
        return (x - mean) / std


class StmEncoder(nn.Module):
    """Encode an STM z-stack ``(B, C, H, W)`` into ``(B, output_dim)``.

    The C z-slices form the depth of a single-channel volume: a 3D convolutional stem, z folded
    into channels, a 2D residual trunk and a pooled linear head.

    Parameters
    ----------
    in_channels : int
        Number of z-slices.
    conv3d_channels : tuple of int
        Widths of the 3D stem layers.
    filters : tuple of int
        Widths of the 2D trunk.
    output_dim : int
        Embedding size; must equal ``config.model.nf``.
    pool_size : int
        Spatial grid size before the head; 1 is global average pooling.
    """

    def __init__(self, in_channels=14, conv3d_channels=(4, 8, 16), filters=(64, 128, 256),
                 kernel_size=None, output_dim=256, n_groups=8, pool_size=1,
                 padding_mode='replicate', negative_slope=0.1):
        super().__init__()
        if kernel_size is None:
            kernel_size = (3,) * len(filters)
        if len(kernel_size) != len(filters):
            raise ValueError(
                f'filters and kernel_size must be the same length, '
                f'got {len(filters)} and {len(kernel_size)}')
        if pool_size < 1:
            raise ValueError(f'pool_size must be >= 1, got {pool_size}')
        if in_channels < 1:
            raise ValueError(f'in_channels (= number of z-slices) must be >= 1, '
                             f'got {in_channels}')

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.negative_slope = negative_slope

        # 3D stem
        widths_3d = [1] + list(conv3d_channels)
        self.conv3d = nn.ModuleList([
            nn.Conv3d(widths_3d[i], widths_3d[i + 1], kernel_size=3, padding=1,
                      padding_mode=padding_mode)
            for i in range(len(conv3d_channels))
        ])

        n_stages = len(conv3d_channels)
        d = in_channels
        z_kernels = []
        for i in range(n_stages):
            k = 2 if (i < n_stages - 1 and d >= 2) else 1
            z_kernels.append(k)
            d //= k
        self.z_kernels = tuple(z_kernels)
        self.z_out = d

        # Fold z into channels, then the 2D trunk.
        self.flat_channels = conv3d_channels[-1] * self.z_out
        self.proj = nn.Conv2d(self.flat_channels, filters[0], kernel_size=1)

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

        # Non-persistent, so it is not part of state_dict.
        self.register_buffer('_scale_checked', torch.zeros(1, dtype=torch.bool),
                             persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=self.negative_slope,
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _check_scale(self, x):
        """Warn once if the input looks unnormalised (raw uint16 values)."""
        if bool(self._scale_checked):
            return
        self._scale_checked.fill_(True)
        absmax = float(x.detach().abs().max())
        if absmax > 100.0:
            print(f'[StmEncoder] WARNING: input |max| = {absmax:.4g}. The raw HDF5 is '
                  f'uint16 (0..65535); this looks unnormalised. Use '
                  f'models.stm_encoder.NormalizeStmCube in the dataset transform, not '
                  f'datasets.ters_dataset.NormalizeVectorized, which standardises each '
                  f'z-slice separately and removes the decay envelope this encoder reads.')

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'StmEncoder expects (B, C, H, W) with C = z-slices, got {tuple(x.shape)}. '
                f'The stack is stored (N, H, W, C) in the HDF5, so it must be permuted to '
                f'channels-first before this point.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'StmEncoder built for in_channels={self.in_channels} z-slices but got '
                f'{x.size(1)}. Set config.model.stm_in_channels to match the corpus '
                f"(len(f[split].attrs['z_slices'])).")
        self._check_scale(x)

        h = x.unsqueeze(1)                                    # (B, 1, Z, H, W)
        for i, conv in enumerate(self.conv3d):
            h = F.leaky_relu(conv(h), negative_slope=self.negative_slope)
            # H, W pool only while >= 2; z follows the fixed schedule that sets flat_channels.
            sk = 2 if min(h.shape[-2:]) >= 2 else 1
            zk = self.z_kernels[i]
            if sk > 1 or zk > 1:
                h = F.avg_pool3d(h, kernel_size=(zk, sk, sk), stride=(zk, sk, sk))

        b, c, d, hh, ww = h.shape
        h = h.reshape(b, c * d, hh, ww)                       # (B, C * Z, H, W)

        h = self.proj(h)
        for block in self.blocks:
            h = block(h)
            # Stop pooling once a side is below 2, so small inputs do not shrink to size 0.
            if min(h.shape[-2:]) >= 2:
                h = self.pool(h)
        return self.out_norm(self.head(h))


# ---------------------------------------------------------------------------------------
# Command-line checks
# ---------------------------------------------------------------------------------------

def _self_test():
    """Check shapes, gradients and the normaliser on random input."""
    torch.manual_seed(0)
    cases = [(14, 128, 128), (10, 64, 64), (1, 128, 128), (14, 16, 16)]
    ok = 0
    for c, hh, ww in cases:
        enc = StmEncoder(in_channels=c, output_dim=256)
        x = torch.randn(2, c, hh, ww)
        y = enc(x)
        assert y.shape == (2, 256), f'{y.shape} != (2, 256)'
        y.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.requires_grad]
        assert all(g is not None for g in grads), 'a parameter received no gradient'
        assert torch.isfinite(y).all(), 'non-finite output'
        n = sum(p.numel() for p in enc.parameters())
        print(f'  ok  C={c:3d} {hh:4d}x{ww:<4d} -> {tuple(y.shape)}  '
              f'z_kernels={enc.z_kernels} z_out={enc.z_out} '
              f'flat={enc.flat_channels:4d}  params={n/1e6:.2f}M')
        ok += 1

    # Slice means must stay monotone in z; the test cube is one pattern scaled per slice.
    base = torch.rand(1, 32, 32) ** 3 * 65535.0
    envelope = torch.linspace(1.0, 0.22, 14).view(14, 1, 1)
    norm = NormalizeStmCube()(base * envelope)
    means = norm.mean(dim=(1, 2))
    assert torch.all(means[:-1] > means[1:]), 'NormalizeStmCube destroyed the z envelope'
    assert abs(float(norm.mean())) < 1e-4 and abs(float(norm.std()) - 1) < 1e-2
    print(f'  ok  NormalizeStmCube keeps the envelope monotone: '
          f'{float(means[0]):+.3f} -> {float(means[-1]):+.3f}')
    ok += 1

    # NormalizeVectorized must flatten every slice mean of the same cube.
    from datasets.ters_dataset import NormalizeVectorized
    ters_means = NormalizeVectorized()(base * envelope).mean(dim=(1, 2))
    assert float(ters_means.abs().max()) < 1e-4, (
        'NormalizeVectorized no longer flattens each slice -- re-check whether StmEncoder '
        'still needs its own normaliser')
    print(f'  ok  NormalizeVectorized (TERS) flattens the same envelope to '
          f'{float(ters_means[0]):+.3f} -> {float(ters_means[-1]):+.3f}')
    ok += 1

    # A constant cube must not yield NaN.
    flat = NormalizeStmCube()(torch.full((14, 8, 8), 1234.0))
    assert torch.isfinite(flat).all(), 'constant cube produced non-finite values'
    print('  ok  constant cube -> finite (zero-std guard)')
    ok += 1

    print(f'self-test: {ok}/{len(cases) + 3} passed')
    return ok == len(cases) + 3


def _inspect(h5_path):
    """Print the groups, attributes and dataset shapes of an HDF5 file."""
    import h5py

    with h5py.File(h5_path, 'r') as f:
        print(f'{h5_path}')
        print(f'  root keys: {list(f.keys())}')
        for split in f.keys():
            g = f[split]
            if not hasattr(g, 'keys'):
                continue
            print(f'  group {split!r}')
            for k, v in g.attrs.items():
                print(f'    attr {k:18s} = {v}')
            for k in g.keys():
                d = g[k]
                print(f'    dset {k:18s} {str(d.shape):24s} {d.dtype}')


def _check(h5_path, split=None, n=8):
    """Run the first ``n`` cubes of a split through ``NormalizeStmCube`` and ``StmEncoder``."""
    import h5py
    import numpy as np

    with h5py.File(h5_path, 'r') as f:
        split = split or list(f.keys())[0]
        g = f[split]
        z_slices = g.attrs.get('z_slices')
        X = g['X']
        print(f'{h5_path}[{split}]  X={X.shape} {X.dtype}  z_slices={z_slices}')

        n = min(n, X.shape[0])
        cubes = np.asarray(X[:n]).astype(np.float32)            # (n, H, W, C)

    stack = torch.from_numpy(cubes).permute(0, 3, 1, 2).contiguous()   # (n, C, H, W)
    in_channels = stack.shape[1]
    if z_slices is not None and len(z_slices) != in_channels:
        print(f'  WARNING: z_slices has {len(z_slices)} entries but X has {in_channels} '
              f'channels')

    norm = NormalizeStmCube()
    normed = torch.stack([norm(c) for c in stack])
    per_slice = normed.mean(dim=(0, 2, 3))
    monotone = bool(torch.all(per_slice[:-1] > per_slice[1:]))
    print(f'  normalised: mean {normed.mean():+.4f}  std {normed.std():.4f}')
    print(f'  per-slice mean {per_slice[0]:+.3f} -> {per_slice[-1]:+.3f}  '
          f'monotone={monotone}')

    enc = StmEncoder(in_channels=in_channels, output_dim=256)
    enc.eval()
    with torch.no_grad():
        y = enc(normed)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f'  encoder: z_kernels={enc.z_kernels} z_out={enc.z_out} '
          f'flat={enc.flat_channels}  params={n_params/1e6:.2f}M')
    print(f'  {tuple(normed.shape)} -> {tuple(y.shape)}  '
          f'mean {y.mean():+.4f}  std {y.std():.4f}  finite={bool(torch.isfinite(y).all())}')
    return y


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true',
                    help='shape/gradient/normaliser checks, no data required')
    ap.add_argument('--inspect', metavar='H5', help='print the HDF5 layout and exit')
    ap.add_argument('--check', metavar='H5', help='run real cubes through the encoder')
    ap.add_argument('--split', default=None, help='split name inside the HDF5')
    a = ap.parse_args()

    if a.self_test:
        _self_test()
    if a.inspect:
        _inspect(a.inspect)
    if a.check:
        _check(a.check, a.split)
    if not (a.self_test or a.inspect or a.check):
        ap.print_help()
