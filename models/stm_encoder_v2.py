"""Residual 3D convolutional STM encoder.

Maps an STM z-stack ``(B, C, H, W)`` to ``(B, output_dim)``. Selected with
``config.model.stm_encoder = 'v2'``. Inputs must be normalised with ``NormalizeStmCube``.

Adapted from SINGROUP/ml-spm (Kurki et al., ACS Nano 18, 11130, 2024): ``ResBlock3d`` is
``Conv3dBlock`` (mlspm/modules.py) with GroupNorm added; widths and z pooling follow ``EDAFMNet``.

Usage
-----
    python -m models.stm_encoder_v2 --self-test
    python -m models.stm_encoder_v2 --bench
    python -m models.stm_encoder_v2 --compare
    python -m models.stm_encoder_v2 --cone H5
"""

import torch
import torch.nn.functional as F
from torch import nn

from .ters_encoder import ResBlock, _groups_for
from .stm_encoder import NormalizeStmCube  # noqa: F401  (re-exported)


class ResBlock3d(nn.Module):
    """3D residual block: ``depth`` conv-norm-LeakyReLU layers and a skip connection.

    The skip is identity when the width is unchanged and a 1x1 conv otherwise.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, depth=2, n_groups=8,
                 padding_mode='replicate', negative_slope=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError(f'depth must be >= 1, got {depth}')
        padding = kernel_size // 2

        convs, norms = [], []
        for i in range(depth):
            cin = in_channels if i == 0 else out_channels
            convs.append(nn.Conv3d(cin, out_channels, kernel_size, padding=padding,
                                   padding_mode=padding_mode))
            norms.append(nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.negative_slope = negative_slope
        self.skip = (nn.Conv3d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(conv(h))
            # The last activation follows the residual add.
            if i < len(self.convs) - 1:
                h = F.leaky_relu(h, negative_slope=self.negative_slope)
        return F.leaky_relu(h + self.skip(x), negative_slope=self.negative_slope)


class StmEncoderV2(nn.Module):
    """Encode an STM z-stack ``(B, C, H, W)`` into ``(B, output_dim)``.

    A spatially strided 3D stem, residual 3D blocks over the z-slices, a 2D residual trunk and
    a pooled linear head.

    Parameters
    ----------
    in_channels : int
        Number of z-slices.
    stem_channels : int
        Width of the strided 3D stem.
    stem_stride : int
        Spatial (H, W) stride of the stem; 1 disables downsampling.
    conv3d_channels : tuple of int
        Widths of the residual 3D blocks.
    block_depth : int
        Convolutions per 3D block.
    z_pool : tuple of int or None
        Pooling factor along z for each 3D block; ``None`` repeats ``(2, 1, 2)``.
    filters : tuple of int
        Widths of the 2D trunk.
    output_dim : int
        Embedding size; must equal ``config.model.nf``.
    pool_size : int
        Spatial grid size before the head.
    """

    def __init__(self, in_channels=14, stem_channels=16, stem_stride=2,
                 conv3d_channels=(48, 96, 192), block_depth=2, z_pool=None,
                 filters=(128, 256), kernel_size=None, output_dim=256, n_groups=8,
                 pool_size=2, padding_mode='replicate', negative_slope=0.1):
        super().__init__()
        if kernel_size is None:
            kernel_size = (3,) * len(filters)
        if len(kernel_size) != len(filters):
            raise ValueError(f'filters and kernel_size must be the same length, '
                             f'got {len(filters)} and {len(kernel_size)}')
        if pool_size < 1:
            raise ValueError(f'pool_size must be >= 1, got {pool_size}')
        if in_channels < 1:
            raise ValueError(f'in_channels (= number of z-slices) must be >= 1, '
                             f'got {in_channels}')
        if stem_stride < 1:
            raise ValueError(f'stem_stride must be >= 1, got {stem_stride}')
        if not conv3d_channels:
            raise ValueError('conv3d_channels must not be empty')

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.negative_slope = negative_slope
        self.stem_stride = stem_stride

        # Strided 3D stem
        self.stem =nn.Conv3d(1, stem_channels, kernel_size=3, stride=(1, stem_stride, stem_stride),
                              padding=1, padding_mode=padding_mode)
        self.stem_norm = nn.GroupNorm(_groups_for(stem_channels, n_groups), stem_channels)

        # z pooling schedule
        n_stages = len(conv3d_channels)
        if z_pool is None:
            pattern = (2, 1, 2)
            z_pool = tuple(pattern[i % len(pattern)] for i in range(n_stages))
        if len(z_pool) != n_stages:
            raise ValueError(f'z_pool must have one entry per 3D stage, got {len(z_pool)} '
                             f'for {n_stages} stages')
        d = in_channels
        z_kernels = []
        for k in z_pool:
            k = k if (k > 1 and d >= 2 * k) else 1        # pool only if 2+ z-planes remain
            z_kernels.append(k)
            d //= k
        self.z_kernels = tuple(z_kernels)
        self.z_out = d

        widths_3d = [stem_channels] + list(conv3d_channels)
        self.blocks_3d = nn.ModuleList([
            ResBlock3d(widths_3d[i], widths_3d[i + 1], kernel_size=3, depth=block_depth,
                       n_groups=n_groups, padding_mode=padding_mode,
                       negative_slope=negative_slope)
            for i in range(n_stages)
        ])

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
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                # Kaiming gain for LeakyReLU in the 3D path, ReLU (a=0) in the 2D ResBlocks.
                a = self.negative_slope if isinstance(m, nn.Conv3d) else 0.0
                nn.init.kaiming_normal_(m.weight, a=a, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _check_scale(self, x):
        """Warn once if the input looks unnormalised (raw uint16 values)."""
        if bool(self._scale_checked):
            return
        self._scale_checked.fill_(True)
        m = float(x.detach().abs().max())
        if m > 50.0:
            import warnings
            warnings.warn(
                f'StmEncoderV2 got input with max |x| = {m:.1f}. Raw STM cubes are uint16 '
                f'(0..65535); NormalizeStmCube was probably not applied. Training runs but '
                f'converges poorly.', RuntimeWarning, stacklevel=2)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'StmEncoderV2 expects (B, C, H, W) with C = z-slices, got {tuple(x.shape)}. '
                f'The stack is stored (N, H, W, C) in the HDF5, so it must be permuted to '
                f'channels-first before this point.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'StmEncoderV2 built for in_channels={self.in_channels} z-slices but got '
                f'{x.size(1)}. Set config.model.stm_in_channels to match the corpus.')
        self._check_scale(x)

        h = x.unsqueeze(1)                                  # (B, 1, D, H, W)
        h = F.leaky_relu(self.stem_norm(self.stem(h)), negative_slope=self.negative_slope)

        for i, block in enumerate(self.blocks_3d):
            h = block(h)
            # H and W pool while >= 2; z follows the fixed schedule that flat_channels assumes.
            sk = 2 if min(h.shape[-2:]) >= 2 else 1
            zk = self.z_kernels[i]
            if sk > 1 or zk > 1:
                h = F.avg_pool3d(h, kernel_size=(zk, sk, sk), stride=(zk, sk, sk))

        b, c, d, hh, ww = h.shape
        h = h.reshape(b, c * d, hh, ww)

        h = self.proj(h)
        for block in self.blocks:
            h = block(h)
            if min(h.shape[-2:]) >= 2:
                h = self.pool(h)
        return self.out_norm(self.head(h))


# ---------------------------------------------------------------------------------------
# Command-line checks
# ---------------------------------------------------------------------------------------

def _mean_pairwise_cosine(E):
    """Mean off-diagonal cosine similarity of an ``(N, d)`` embedding matrix."""
    cos = torch.nn.functional.cosine_similarity(E[:, None, :], E[None, :, :], dim=-1)
    off = cos[~torch.eye(len(E), dtype=torch.bool, device=E.device)]
    return float(off.mean())


def _self_test():
    """Check shapes, gradients, 3D depth and the residual path on random input."""
    torch.manual_seed(0)
    cases = [(14, 128, 128), (10, 64, 64), (1, 128, 128), (14, 16, 16)]
    checks = 0
    for c, hh, ww in cases:
        enc = StmEncoderV2(in_channels=c, output_dim=256)
        x = torch.randn(2, c, hh, ww)
        y = enc(x)
        assert y.shape == (2, 256), f'{y.shape} != (2, 256)'
        y.sum().backward()
        missing = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f'parameters received no gradient: {missing[:4]}'
        assert torch.isfinite(y).all(), 'non-finite output'
        assert enc.z_out >= 1, 'z pooled to zero depth'
        n = sum(p.numel() for p in enc.parameters())
        print(f'  ok  C={c:3d} {hh:4d}x{ww:<4d} -> {tuple(y.shape)}  '
              f'z_kernels={enc.z_kernels} z_out={enc.z_out} '
              f'flat={enc.flat_channels:5d}  params={n/1e6:.2f}M')
        checks += 1

    # The default 3D stem must have at least six non-1x1 convolutions.
    enc = StmEncoderV2()
    n3d = sum(1 for m in enc.modules() if isinstance(m, nn.Conv3d) and m.kernel_size != (1, 1, 1))
    assert n3d >= 6, f'expected a deep 3D stem, got {n3d} 3D convolutions'
    print(f'  ok  3D stem is {n3d} convolutions deep (v1 has 3), widths '
          f'{tuple(b.convs[-1].out_channels for b in enc.blocks_3d)}')
    checks += 1

    # Identity skip when the width is unchanged, 1x1 conv otherwise.
    same = ResBlock3d(32, 32)
    diff = ResBlock3d(32, 64)
    assert isinstance(same.skip, nn.Identity), 'same-width block should skip through identity'
    assert isinstance(diff.skip, nn.Conv3d), 'width-changing block needs a 1x1 adapter'
    print('  ok  residual skip: identity when width is unchanged, 1x1 when it is not')
    checks += 1

    # With zeroed convolutions the block must pass the input through the skip.
    b = ResBlock3d(8, 8)
    with torch.no_grad():
        for conv in b.convs:
            conv.weight.zero_()
            conv.bias.zero_()
    x = torch.randn(2, 8, 4, 8, 8).abs() + 0.1        # positive, so LeakyReLU is identity
    assert torch.allclose(b(x), x, atol=1e-5), 'residual path does not carry the input'
    print('  ok  zeroing the convolutions leaves the skip intact (residual is real)')
    checks += 1

    # The unnormalised-input warning must fire exactly once.
    import warnings
    enc = StmEncoderV2(in_channels=14)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
    assert len(w) == 1, f'scale warning fired {len(w)} times, expected exactly 1'
    print('  ok  un-normalised input warns once, not every step')
    checks += 1

    # Distinct inputs must give distinct embeddings.
    torch.manual_seed(0)
    enc = StmEncoderV2(in_channels=14).eval()
    with torch.no_grad():
        E = enc(torch.randn(16, 14, 64, 64))
    cone = _mean_pairwise_cosine(E)
    assert cone < 0.9999, f'embeddings are identical at init: mean pairwise cosine {cone:.6f}'
    assert float(E.std(0).mean()) > 1e-3, 'embedding dimensions carry no across-sample variance'
    print(f'  ok  distinct inputs give distinct embeddings: mean pairwise cosine '
          f'{cone:+.4f} at init (v1 is 0.9951 here; the trained cone is 0.967)')
    checks += 1

    print(f'self-test: {checks}/{len(cases) + 5} passed')
    return checks == len(cases) + 5


def _compare(in_channels=14, hw=128):
    """Print output shape, 3D depth, width and parameter count of v1 and v2 side by side."""
    from .stm_encoder import StmEncoder

    # Same pool_size for both; the defaults differ.
    v1 = StmEncoder(in_channels=in_channels, output_dim=256, pool_size=2)
    v2 = StmEncoderV2(in_channels=in_channels, output_dim=256, pool_size=2)
    x = torch.randn(2, in_channels, hw, hw)

    rows = []
    for name, enc in (('v1 (ASDAFMNet-style)', v1), ('v2 (EDAFMNet-style)', v2)):
        with torch.no_grad():
            y = enc(x)
        n3d = sum(1 for m in enc.modules()
                  if isinstance(m, nn.Conv3d) and m.kernel_size != (1, 1, 1))
        widest = max((m.out_channels for m in enc.modules() if isinstance(m, nn.Conv3d)),
                     default=0)
        rows.append((name, sum(p.numel() for p in enc.parameters()), n3d, widest,
                     enc.z_kernels, enc.z_out, enc.flat_channels, tuple(y.shape)))

    print(f'{"":24s} {"params":>9s} {"3D convs":>9s} {"widest 3D":>10s} '
          f'{"z kernels":>12s} {"z out":>6s} {"flat":>6s}  out')
    for name, n, n3d, widest, zk, zo, flat, shape in rows:
        print(f'{name:24s} {n/1e6:8.2f}M {n3d:9d} {widest:10d} {str(zk):>12s} '
              f'{zo:6d} {flat:6d}  {shape}')
    ratio = rows[1][1] / rows[0][1]
    print(f'\nv2 is {ratio:.2f}x the parameters of v1, and {rows[1][2] // rows[0][2]}x '
          f'the 3D depth. The denoiser is 29.30M, so the encoder goes from '
          f'{rows[0][1] / 29.30e6 * 100:.0f}% to {rows[1][1] / 29.30e6 * 100:.0f}% of it.')
    return rows


def _bench(batch=128, in_channels=14, hw=128, steps=5):
    """Print parameters, forward+backward time and, on a GPU, peak memory for v1 and v2."""
    import time
    from .stm_encoder import StmEncoder

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {dev}   batch={batch}  cube={in_channels}x{hw}x{hw}')
    if dev == 'cpu':
        print('  (no GPU here -- peak memory is not measured; run this on a GPU node)')

    for name, cls in (('v1', StmEncoder), ('v2', StmEncoderV2)):
        enc = cls(in_channels=in_channels, output_dim=256).to(dev)
        x = torch.randn(batch, in_channels, hw, hw, device=dev)
        if dev == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            enc.zero_grad(set_to_none=True)
            enc(x).sum().backward()
        if dev == 'cuda':
            torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        n = sum(p.numel() for p in enc.parameters())
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else float('nan')
        print(f'  {name}: params {n/1e6:6.2f}M   fwd+bwd {dt*1000:8.1f} ms/step   '
              f'peak {peak:6.2f} GiB')
        del enc, x
        if dev == 'cuda':
            torch.cuda.empty_cache()


def _cone(h5_path, split=None, n=256, seed=0):
    """Print the mean pairwise cosine similarity of untrained v1 and v2 embeddings on real
    cubes."""
    import h5py
    import numpy as np
    from .stm_encoder import StmEncoder

    torch.manual_seed(seed)
    with h5py.File(h5_path, 'r') as f:
        split = split or list(f.keys())[0]
        g = f[split]
        X = g['X']
        n = min(n, X.shape[0])
        cubes = np.asarray(X[:n]).astype(np.float32)
    stack = torch.from_numpy(cubes).permute(0, 3, 1, 2).contiguous()
    in_channels = stack.shape[1]

    norm = NormalizeStmCube()
    normed = torch.stack([norm(c) for c in stack])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    normed = normed.to(dev)
    print(f'{h5_path}[{split}]  n={n}  C={in_channels}  device={dev}')

    for name, cls in (('v1 untrained', StmEncoder), ('v2 untrained', StmEncoderV2)):
        torch.manual_seed(seed)                       # same init draw for both
        enc = cls(in_channels=in_channels, output_dim=256).to(dev).eval()
        with torch.no_grad():
            E = enc(normed)
        print(f'  {name}: mean pairwise cosine {_mean_pairwise_cosine(E):+.6f}   '
              f'std/|act| {float(E.std(0).mean() / E.abs().mean()):.4f}')
    print('  reference (trained models, real cubes): '
          'v1 @75k 0.984, v1 @150k 0.967, TERS @28k 0.925')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--self-test', action='store_true', help='no data, no cluster, no GPU')
    ap.add_argument('--compare', action='store_true', help='v1 vs v2 shapes and parameters')
    ap.add_argument('--bench', action='store_true', help='params, peak memory, step time')
    ap.add_argument('--cone', metavar='H5', help='untrained cone on real cubes (v1 vs v2)')
    ap.add_argument('--split', default=None)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('-n', type=int, default=256)
    a = ap.parse_args()

    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if a.compare:
        _compare()
    if a.bench:
        _bench(batch=a.batch)
    if a.cone:
        _cone(a.cone, split=a.split, n=a.n)
    if not (a.self_test or a.compare or a.bench or a.cone):
        ap.print_help()
