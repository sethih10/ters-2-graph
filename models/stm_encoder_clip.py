"""3D convolutional STM encoder with contrastively pretrained weights.

Maps an STM z-stack ``(B, C, H, W)`` to ``(B, output_dim)``. Selected with
``config.model.stm_encoder = 'clip'``. Inputs must be normalised with ``NormalizeStmCube``.

Usage
-----
    python -m models.stm_encoder_clip --self-test
    python -m models.stm_encoder_clip --load CKPT
    python -m models.stm_encoder_clip --cone H5
"""

import torch
import torch.nn.functional as F
from torch import nn

from .ters_encoder import ResBlock, _groups_for
from .stm_encoder import NormalizeStmCube  # noqa: F401  (re-exported)

DEFAULT_CKPT = ('/scratch/project_2001912/sethih1/runs/clip-full-1264302/'
                'cube_encoder_best.pt')


class ResBlock3d(nn.Module):
    """3D residual block: conv-norm-ReLU, conv-norm, skip connection, ReLU.

    Parameter names are checkpoint keys; do not rename them.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_groups_for(out_channels, n_groups), out_channels)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Conv3d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class StmCubeEncoder(nn.Module):
    """Encode an STM z-stack ``(B, C, H, W)`` into ``(B, output_dim)``.

    A 3D convolutional stem over the z-slices, a 2D residual trunk and a pooled linear head.
    Submodule names and shapes match the pretrained checkpoint.

    Parameters
    ----------
    in_channels : int
        Number of z-slices.
    conv3d_channels : tuple of int
        Widths of the 3D stem stages.
    z_strides : tuple of int
        Pooling factor along z for each stem stage.
    filters : tuple of int
        Widths of the 2D trunk.
    output_dim : int
        Embedding size; must equal ``config.model.nf``.
    pool_size : int
        Spatial grid size before the head; the pretrained checkpoint uses 1.
    """

    def __init__(self, in_channels=14, conv3d_channels=(4, 8, 16), z_strides=(2, 2, 1),
                 filters=(64, 128, 256), kernel_size=3, output_dim=256, n_groups=8,
                 pool_size=1):
        super().__init__()
        if len(conv3d_channels) != len(z_strides):
            raise ValueError(f'conv3d_channels and z_strides must be the same length, got '
                             f'{len(conv3d_channels)} and {len(z_strides)}')
        if pool_size < 1:
            raise ValueError(f'pool_size must be >= 1, got {pool_size}')
        if in_channels < 1:
            raise ValueError(f'in_channels (= number of z-slices) must be >= 1, '
                             f'got {in_channels}')

        self.in_channels = in_channels
        self.output_dim = output_dim
        # Attribute name used by the pretraining code.
        self.latent_dim = output_dim
        self.pool_size = pool_size

        # 3D stem
        self.stem3d = nn.ModuleList()
        self.pool3d = nn.ModuleList()
        c_in = 1
        z = in_channels
        for c_out, zs in zip(conv3d_channels, z_strides):
            self.stem3d.append(ResBlock3d(c_in, c_out, kernel_size, n_groups=n_groups))
            # Do not pool z below depth 1.
            zs_eff = zs if z // zs >= 1 else 1
            self.pool3d.append(nn.MaxPool3d((zs_eff, 2, 2)))
            c_in = c_out
            z = max(1, z // zs_eff)
        self.z_out = z
        self.c3d_out = c_in
        self.flat_channels = self.c3d_out * self.z_out

        # Fold z into channels, then the 2D trunk.
        self.to_2d = nn.Conv2d(self.flat_channels, filters[0], kernel_size=1)
        widths = [filters[0]] + list(filters)
        self.blocks = nn.ModuleList([
            ResBlock(widths[i], widths[i + 1], kernel_size, n_groups=n_groups)
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
        """Initialise weights with the same scheme as pretraining."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
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
        m = float(x.detach().abs().max())
        if m > 50.0:
            import warnings
            warnings.warn(
                f'StmCubeEncoder got input with max |x| = {m:.1f}. Raw STM cubes are uint16 '
                f'(0..65535); NormalizeStmCube was probably not applied. The pretrained '
                f'weights expect its output.',
                RuntimeWarning, stacklevel=2)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f'StmCubeEncoder expects (B, C, H, W) with C = z-slices, got '
                f'{tuple(x.shape)}. The stack is stored (N, H, W, C) in the HDF5, so it must '
                f'be permuted to channels-first before this point.')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'StmCubeEncoder built for in_channels={self.in_channels} z-slices but got '
                f'{x.size(1)}. Set config.model.stm_in_channels to match the corpus '
                f"(len(f[split].attrs['z_slices'])).")
        self._check_scale(x)

        b = x.shape[0]
        x = x.unsqueeze(1)                                   # (B, 1, Z, H, W)
        for block, pool in zip(self.stem3d, self.pool3d):
            x = block(x)
            if x.shape[-1] >= 2 and x.shape[-2] >= 2:
                x = pool(x)
        x = x.reshape(b, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.to_2d(x)
        for block in self.blocks:
            x = block(x)
            if min(x.shape[-2:]) >= 2:
                x = self.pool(x)
        return self.out_norm(self.head(x))


# ---------------------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------------------

def read_ckpt_geometry(ckpt_path):
    """Infer the encoder configuration from a checkpoint's tensor shapes.

    Returns ``conv3d_channels``, ``filters``, ``output_dim``, ``pool_size``, ``flat_channels``,
    ``epoch`` and ``cube_encoder_class``. ``in_channels`` cannot be inferred from the weights.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)

    def need(key):
        if key not in sd:
            raise ValueError(
                f'{ckpt_path} has no tensor {key!r}, so it is not a StmCubeEncoder '
                f'checkpoint. Keys begin with: '
                f'{sorted({k.split(".")[0] for k in sd})[:8]}.')
        return sd[key]

    n3d = len({k.split('.')[1] for k in sd if k.startswith('stem3d.')})
    n2d = len({k.split('.')[1] for k in sd if k.startswith('blocks.')})
    conv3d_channels = tuple(int(need(f'stem3d.{i}.conv2.weight').shape[0]) for i in range(n3d))
    filters = tuple(int(need(f'blocks.{i}.conv2.weight').shape[0]) for i in range(n2d))

    head_w = need('head.2.weight')
    output_dim = int(head_w.shape[0])
    flat_head = int(head_w.shape[1])
    # head.2 is Linear(filters[-1] * pool_size**2, output_dim).
    if filters[-1] == 0 or flat_head % filters[-1]:
        raise ValueError(f'{ckpt_path}: head.2.weight is {tuple(head_w.shape)} but the last '
                         f'2D width is {filters[-1]}, which does not divide it.')
    pool_sq = flat_head // filters[-1]
    pool_size = int(round(pool_sq ** 0.5))
    if pool_size * pool_size != pool_sq:
        raise ValueError(f'{ckpt_path}: head implies pool_size**2 = {pool_sq}, which is not '
                         f'a perfect square. This is not a StmCubeEncoder head.')

    return {'conv3d_channels': conv3d_channels, 'filters': filters,
            'output_dim': output_dim, 'pool_size': pool_size,
            'flat_channels': int(need('to_2d.weight').shape[1]),
            'epoch': ckpt.get('epoch'), 'cube_encoder_class': ckpt.get('cube_encoder_class')}


def load_pretrained_stm_cube_encoder(encoder, ckpt_path, verbose=True):
    """Load a pretrained checkpoint into ``encoder``.

    Raises ``ValueError`` if the encoder class, shapes or keys do not match.
    """
    geom = read_ckpt_geometry(ckpt_path)
    cls = geom.get('cube_encoder_class')
    if cls not in (None, 'StmCubeEncoder'):
        raise ValueError(
            f'{ckpt_path} was written for {cls!r}, not StmCubeEncoder. Loading it here would '
            f'match on name and differ in meaning.')

    want, have = geom['pool_size'], encoder.pool_size
    if want != have:
        raise ValueError(
            f'{ckpt_path} was trained with pool_size={want} but this encoder was built with '
            f'pool_size={have}, so head.2.weight cannot be loaded. Set '
            f'config.model.stm_clip_pool_size={want}. (stm_pool_size is a separate setting '
            f'that applies to the v1 encoder.)')
    if geom['output_dim'] != encoder.output_dim:
        raise ValueError(
            f'{ckpt_path} has output_dim={geom["output_dim"]} but this encoder was built for '
            f'{encoder.output_dim}. output_dim must equal config.model.nf.')
    if geom['flat_channels'] != encoder.flat_channels:
        raise ValueError(
            f'{ckpt_path} has to_2d over {geom["flat_channels"]} channels but this encoder '
            f'reshapes z into {encoder.flat_channels} (= conv3d_channels[-1] * z_out, with '
            f'in_channels={encoder.in_channels}). Check config.model.stm_in_channels against '
            f"len(f[split].attrs['z_slices']).")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    missing, unexpected = encoder.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    if verbose:
        print(f'Loaded {len(sd)}/{len(sd)} tensors into StmCubeEncoder from {ckpt_path} '
              f'(epoch {geom["epoch"]}, pool_size={want}, '
              f'{sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M params)')
    return encoder


# ---------------------------------------------------------------------------------------
# Command-line checks
# ---------------------------------------------------------------------------------------

# Parameter names of each 3D stem block in the checkpoint.
_EXPECTED_3D_LEAVES = ('conv1.weight', 'conv1.bias', 'norm1.weight', 'norm1.bias',
                       'conv2.weight', 'conv2.bias', 'norm2.weight', 'norm2.bias',
                       'skip.weight', 'skip.bias')


def _self_test():
    """Check shapes, gradients, checkpoint keys and the loader on random input."""
    import tempfile
    import os
    import warnings
    from .stm_encoder_v2 import _mean_pairwise_cosine

    torch.manual_seed(0)
    cases = [(14, 128, 128), (10, 64, 64), (1, 128, 128), (14, 16, 16)]
    checks = 0
    for c, hh, ww in cases:
        enc = StmCubeEncoder(in_channels=c, output_dim=256)
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
              f'z_out={enc.z_out} flat={enc.flat_channels:5d}  params={n/1e6:.2f}M')
        checks += 1

    # state_dict keys and shapes must match the checkpoint.
    enc = StmCubeEncoder(in_channels=14, output_dim=256, pool_size=1)
    keys = set(enc.state_dict())
    assert len(keys) == 64, f'expected 64 tensors (the checkpoint has 64), got {len(keys)}'
    expected_3d = {f'stem3d.{i}.{leaf}' for i in range(3) for leaf in _EXPECTED_3D_LEAVES}
    assert expected_3d <= keys, (
        f'3D stem keys do not match the checkpoint. Missing: '
        f'{sorted(expected_3d - keys)[:6]}. ResBlock3d must keep the parameter names '
        f'conv1, norm1, conv2, norm2 and skip, which are the checkpoint keys.')
    for k in ('to_2d.weight', 'head.2.weight', 'out_norm.weight',
              'blocks.0.conv1.weight', 'blocks.2.skip.weight'):
        assert k in keys, f'missing checkpoint key {k!r}'
    assert '_scale_checked' not in keys, (
        'the scale-guard buffer leaked into state_dict; it must be persistent=False or a '
        'strict load of the released checkpoint breaks')
    assert tuple(enc.state_dict()['head.2.weight'].shape) == (256, 256), \
        'head shape at pool_size=1 must match the released (256, 256)'
    assert tuple(enc.state_dict()['to_2d.weight'].shape) == (64, 48, 1, 1), \
        'to_2d shape must match the released (64, 48, 1, 1)'
    print(f'  ok  state_dict is 64 tensors with the checkpoint\'s names '
          f'(head {tuple(enc.state_dict()["head.2.weight"].shape)}, '
          f'to_2d {tuple(enc.state_dict()["to_2d.weight"].shape)})')
    checks += 1

    # Round-trip the loader through a temporary file.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fake_cube_encoder.pt')
        torch.save({'state_dict': enc.state_dict(), 'epoch': 7,
                    'cube_encoder_class': 'StmCubeEncoder'}, p)

        geom = read_ckpt_geometry(p)
        assert geom['pool_size'] == 1, f'derived pool_size {geom["pool_size"]} != 1'
        assert geom['output_dim'] == 256, geom['output_dim']
        assert geom['conv3d_channels'] == (4, 8, 16), geom['conv3d_channels']
        assert geom['filters'] == (64, 128, 256), geom['filters']
        print(f'  ok  geometry recovered from weights alone: pool_size={geom["pool_size"]}, '
              f'conv3d={geom["conv3d_channels"]}, filters={geom["filters"]}')
        checks += 1

        fresh = StmCubeEncoder(in_channels=14, output_dim=256, pool_size=1)
        load_pretrained_stm_cube_encoder(fresh, p, verbose=False)
        same = all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(),
                                                     enc.state_dict().values()))
        assert same, 'round-tripped weights differ from what was saved'
        print('  ok  strict load round-trips every tensor')
        checks += 1

        # A pool_size mismatch must raise.
        wrong = StmCubeEncoder(in_channels=14, output_dim=256, pool_size=2)
        try:
            load_pretrained_stm_cube_encoder(wrong, p, verbose=False)
            raise AssertionError('pool_size=2 loaded a pool_size=1 checkpoint silently')
        except ValueError as exc:
            assert 'stm_clip_pool_size' in str(exc), \
                f'the error must name the config knob that fixes it, got: {exc}'
        print('  ok  a pool_size mismatch raises and names stm_clip_pool_size')
        checks += 1

        # A checkpoint for a different encoder class must be refused.
        torch.save({'state_dict': enc.state_dict(), 'cube_encoder_class': 'TersEncoder'},
                   p)
        try:
            load_pretrained_stm_cube_encoder(
                StmCubeEncoder(in_channels=14, output_dim=256, pool_size=1), p, verbose=False)
            raise AssertionError('a foreign cube_encoder_class was accepted')
        except ValueError as exc:
            assert 'TersEncoder' in str(exc), exc
        print('  ok  a checkpoint written for another class is refused by name')
        checks += 1

    # Identity skip when the width is unchanged, 1x1 conv otherwise.
    assert isinstance(ResBlock3d(32, 32).skip, nn.Identity), \
        'same-width block should skip through identity'
    assert isinstance(ResBlock3d(32, 64).skip, nn.Conv3d), \
        'width-changing block needs a 1x1 adapter'
    print('  ok  residual skip: identity when width is unchanged, 1x1 when it is not')
    checks += 1

    # The unnormalised-input warning must fire exactly once.
    enc = StmCubeEncoder(in_channels=14)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
    assert len(w) == 1, f'scale warning fired {len(w)} times, expected exactly 1'
    print('  ok  un-normalised input warns once, not every step')
    checks += 1

    # Distinct inputs must give distinct embeddings.
    torch.manual_seed(0)
    enc = StmCubeEncoder(in_channels=14).eval()
    with torch.no_grad():
        E = enc(torch.randn(16, 14, 64, 64))
    cone = _mean_pairwise_cosine(E)
    assert cone < 0.9999, f'embeddings are identical at init: mean pairwise cosine {cone:.6f}'
    assert float(E.std(0).mean()) > 1e-3, 'embedding dimensions carry no across-sample variance'
    print(f'  ok  distinct inputs give distinct embeddings: mean pairwise cosine '
          f'{cone:+.4f} at init (pretrained, on real cubes, is +0.168)')
    checks += 1

    total = len(cases) + 8
    print(f'self-test: {checks}/{total} passed')
    return checks == total


def _load(ckpt_path=DEFAULT_CKPT, in_channels=14):
    """Strict-load a checkpoint, print its geometry and run one forward pass on random input."""
    geom = read_ckpt_geometry(ckpt_path)
    print(f'{ckpt_path}\n  geometry from weights: {geom}')
    enc = StmCubeEncoder(in_channels=in_channels, output_dim=geom['output_dim'],
                         conv3d_channels=geom['conv3d_channels'], filters=geom['filters'],
                         pool_size=geom['pool_size'])
    load_pretrained_stm_cube_encoder(enc, ckpt_path)
    enc.eval()
    with torch.no_grad():
        y = enc(torch.randn(2, in_channels, 128, 128) * 0.5)
    print(f'  forward on random input -> {tuple(y.shape)}, finite={bool(torch.isfinite(y).all())}')
    return enc


def _cone(h5_path, ckpt_path=DEFAULT_CKPT, split=None, n=128, seed=0):
    """Print the mean pairwise cosine similarity of pretrained and randomly initialised
    embeddings on real cubes."""
    import h5py
    import numpy as np
    from .stm_encoder_v2 import _mean_pairwise_cosine

    torch.manual_seed(seed)
    with h5py.File(h5_path, 'r') as f:
        split = split or list(f.keys())[0]
        X = f[split]['X']
        n = min(n, X.shape[0])
        cubes = np.asarray(X[:n]).astype(np.float32)
    stack = torch.from_numpy(cubes).permute(0, 3, 1, 2).contiguous()
    in_channels = stack.shape[1]

    norm = NormalizeStmCube()
    normed = torch.stack([norm(c) for c in stack])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    normed = normed.to(dev)
    print(f'{h5_path}[{split}]  n={n}  C={in_channels}  device={dev}')

    geom = read_ckpt_geometry(ckpt_path)
    torch.manual_seed(seed)
    trained = StmCubeEncoder(in_channels=in_channels, output_dim=geom['output_dim'],
                             conv3d_channels=geom['conv3d_channels'],
                             filters=geom['filters'], pool_size=geom['pool_size'])
    load_pretrained_stm_cube_encoder(trained, ckpt_path, verbose=False)
    torch.manual_seed(seed)
    control = StmCubeEncoder(in_channels=in_channels, output_dim=geom['output_dim'],
                             conv3d_channels=geom['conv3d_channels'],
                             filters=geom['filters'], pool_size=geom['pool_size'])

    for name, enc in (('pretrained', trained), ('random init (control)', control)):
        enc = enc.to(dev).eval()
        with torch.no_grad():
            E = torch.cat([enc(normed[i:i + 16]) for i in range(0, len(normed), 16)])
        print(f'  {name:32s} mean pairwise cosine {_mean_pairwise_cosine(E):+.6f}   '
              f'std/|act| {float(E.std(0).mean() / E.abs().mean()):.4f}')
    print('  reference: v1 @150k 0.967, v1 @75k 0.984, TERS @28k 0.925. The 0.011 '
          'reported during contrastive pretraining is on the projection head g(h), which '
          'is not exported.')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--self-test', action='store_true', help='no data, no cluster, no GPU')
    ap.add_argument('--load', nargs='?', const=DEFAULT_CKPT, metavar='CKPT',
                    help='strict-load the real checkpoint and report its geometry')
    ap.add_argument('--cone', metavar='H5', help='pretrained vs random cone on real cubes')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--split', default=None)
    ap.add_argument('-n', type=int, default=128)
    a = ap.parse_args()

    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if a.load:
        _load(a.load)
    if a.cone:
        _cone(a.cone, ckpt_path=a.ckpt, split=a.split, n=a.n)
    if not (a.self_test or a.load or a.cone):
        ap.print_help()
