"""Conditioning encoder for STM z-stacks (``/scratch/project_2001912/stm-data``).

Drop-in sibling of :class:`~models.ters_encoder.TersEncoder`.  Same contract, so nothing in
the denoiser moves: ``(B, C, H, W) -> (B, output_dim)``, consumed by ``cond_lin`` at
models/dmt.py:348-349.

Why this is not just ``TersEncoder(in_channels=14)``
---------------------------------------------------
Mechanically it could be -- the tensor contract is identical.  The reason it is not is that
**the channel axis means something different in the two datasets.**

  * TERS: 100 *spectral* bins.  Unordered as far as the network is concerned; a 1x1 conv
    stem that freely mixes them is exactly right.
  * STM: 14 *constant-height slices* at z = 6..19 (``val.attrs['z_slices']``).  The axis is
    a physical distance, it is monotonically ordered, and the signal decays smoothly along
    it (measured per-slice means, 64 val cubes: 0.083 -> 0.019, a clean geometric decay).
    That decay envelope *is* the height information.  A 1x1 stem can represent it but has no
    inductive bias for it; a 3D convolution reads it directly.

This is the same argument the AFM structure-discovery literature settled on -- see the
provenance block -- and it is why this file exists rather than a config knob.

Provenance (ladder in docs/README.md; ledger in docs/02-reuse-map.md)
--------------------------------------------------------------------
  * ``ResBlock``, ``_groups_for``  -- **REUSE**, imported unmodified from
        models/ters_encoder.py:47 and :29.  The GroupNorm-instead-of-BatchNorm argument
        recorded there (DataParallel computes BatchNorm stats per replica) applies here
        verbatim, so the block is taken as-is rather than re-derived.
  * 3D stem (``Conv3d`` 1->4->8->16, k=3, ``AvgPool3d``, z-to-channels flatten)
                                   -- **ADAPT** of ``ASDAFMNet`` in SINGROUP/ml-spm
        (``mlspm/image/models.py``), the reference implementation for Alldritt et al.,
        "Automated structure discovery in atomic force microscopy", Sci. Adv. 6, eaay6913
        (2020).  That model reads a stack of constant-height AFM images with three 3x3x3
        ``Conv3d`` layers interleaved with ``AvgPool3d``, doubling channels while the
        pooling "collapses the data in the z direction from 3D to 2D", then flattens the
        surviving z into channels.  Kept: layer count, kernel size, the 4/8/16 widths, the
        replicate padding, LeakyReLU(0.1), and the (2,2,2),(2,2,2),(2,2,1) pool schedule.
        Dropped: everything after the bottleneck -- ASDAFMNet is an encoder/decoder that
        predicts an image, we need a vector.
  * pooling head, ``out_norm``, ``reset_parameters``
                                   -- **REUSE** of the interface TersEncoder exposes at
        models/ters_encoder.py:129-141, so swapping encoders stays a one-line change and
        ``cond_lin`` sees an identically-scaled input either way.
  * ``NormalizeStmCube``           -- **NEW**.  Defended below; the existing
        ``NormalizeVectorized`` is measurably wrong for this data.

.. warning::
   The ADAPT row was reconstructed from the published description of ``ASDAFMNet`` and the
   paper, not by importing ``mlspm`` (it is not a dependency here, and adding one would
   violate docs/06-code-practices.md 8).  Shapes and training behaviour are verified by
   ``--self-test``; layer-for-layer correspondence with ``mlspm`` is not.  Do not load an
   ``asdafm-light``/``asdafm-heavy`` checkpoint into this without diffing first.

Checked first and rejected:
  * ``TersEncoder`` -- see above; kept as the ablation baseline, not the default.
  * ``models/layers.py`` -- graph message passing only, no convolution of any rank.
  * ``SpecFormer`` -- 1-D token sequences; a 14x128x128 stack is not a spectrum.

Usage
-----
    python -m models.stm_encoder --self-test                       # no data needed
    python -m models.stm_encoder --inspect /scratch/project_2001912/stm-data/val.h5
    python -m models.stm_encoder --check   /scratch/project_2001912/stm-data/val.h5
"""

import torch
import torch.nn.functional as F
from torch import nn

# REUSE, not re-derivation: these are the TERS blocks, unmodified.  If the GroupNorm choice
# is ever revisited there, both encoders must move together -- which is the point of importing
# rather than copying.
from .ters_encoder import ResBlock, _groups_for


class NormalizeStmCube:
    """Per-cube normalisation for an STM z-stack.  NEW -- and deliberately *not*
    ``NormalizeVectorized`` (datasets/ters_dataset.py:31).

    **Why the TERS normaliser is wrong here.** ``NormalizeVectorized`` standardises each
    channel independently over ``(H, W)``.  On TERS that is right: spectral bins have
    unrelated units and dead channels to guard against.  On an STM z-stack it forces every
    slice to mean 0 and std 1 *separately*, which deletes the decay envelope across z --
    the one thing that distinguishes slice 6 from slice 19.  Measured on 64 val cubes:

        raw (x/65535)          per-slice mean, first -> last:  +0.083 -> +0.019
        per-slice standardised per-slice mean, first -> last:  -0.000 -> -0.000

    The envelope is not attenuated, it is *gone*.  Feeding that to a 3D conv stem whose
    entire purpose is to read along z would be self-defeating.

    **What this does instead**, in three steps:

    1. ``x / 65535`` undoes the uint16 packing.  The file stores ``uint16 per-sample
       max-scaled`` with ``recover = X.astype(float32) * scale[i]`` (root attrs).  The
       per-sample ``scale`` is a single constant per cube, so for a *conditioning* encoder
       -- which only ever sees one cube at a time and is followed by a LayerNorm -- it is an
       irrelevant global gain.  Dropping it keeps every cube on a common [0, 1] axis instead
       of spreading them over the ~1e-12..1e-11 range the raw scales span.  If absolute
       current ever matters downstream, that is what ``scale`` is for; it is not lost, just
       not used here.
    2. ``asinh(x / asinh_scale)`` tames the skew.  The distribution is extreme: median
       0.0032 but p99 0.527, i.e. the 99th percentile sits 5.2 sigma out, so a handful of
       bright pixels would dominate any linear normalisation.  ``asinh`` is linear for
       ``x << s`` and logarithmic for ``x >> s``, and unlike ``log`` it is defined at 0
       (7.4% of pixels are exactly 0).  The default ``s = 0.01`` puts the median at 0.32 s
       -- in the linear regime -- and p99 at 53 s, deep in the log regime, so the transform
       compresses precisely the tail and leaves the bulk alone.
    3. Standardise over **the whole cube at once**, not per slice.  This is the step that
       preserves the ordering: after it the per-slice means still run monotonically
       (+0.249 -> -0.229 on the same 64 cubes), so the envelope survives into the stem.

    Parameters
    ----------
    uint16_max : float
        Divisor for step 1.  Set to ``1.0`` if the cube was already converted to float.
    asinh_scale : float
        ``s`` in step 2.  ``None`` disables the compression (linear normalisation).
    eps : float
        Zero-std guard, same role as in ``NormalizeVectorized``.  A whole cube being
        constant is pathological rather than routine here -- unlike TERS, no dead slice was
        found (min per-sample-per-slice std over 64x14 slices was 0.029) -- but a corrupt
        record should produce zeros, not NaN.
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
    """STM z-stack -> a single conditioning vector.

    Input is the same channels-first cube the TERS path uses, ``(B, C, H, W)`` with
    ``C`` = number of z-slices, so the dataset/collate contract is unchanged.  Internally it
    is treated as a single-channel 3D volume ``(B, 1, D, H, W)`` with ``D = C``, which is
    what lets the stem convolve *along* z instead of mixing z as unordered features.

    Parameters
    ----------
    in_channels : int
        Number of z-slices.  14 for this corpus (``z_slices`` 6..19).  This is also ``D``,
        so it is needed at construction: the z-pooling schedule and hence the width handed
        to the 2D trunk depend on it.
    conv3d_channels : tuple of int
        Widths of the 3D stem.  ``(4, 8, 16)`` is the ASDAFMNet ladder.
    filters : tuple of int
        Widths of the 2D ResBlock ladder after the z-collapse.  Shorter than TersEncoder's
        ``(64, 128, 256, 512)`` on purpose: the stem has already downsampled 8x spatially
        (128 -> 16), so a fourth block would pool a 2x2 map.
    output_dim : int
        Must equal ``config.model.nf`` -- ``cond_lin`` is ``Linear(hidden_dim, time_dim)``
        (models/dmt.py:262).  256 for the current configs.
    pool_size : int
        Spatial grid kept by the head before flattening; ``1`` is a global average pool.
        Same knob and same rationale as ``TersEncoder.pool_size`` (docs/experimental.md E-1).
    """

    def __init__(self, in_channels=14, conv3d_channels=(4, 8, 16), filters=(64, 128, 256),
                 kernel_size=None, output_dim=256, n_groups=8, pool_size=1,
                 padding_mode='replicate', negative_slope=0.1):
        super().__init__()
        # Default tracks len(filters) instead of being a fixed 3-tuple, so overriding
        # `filters` alone (the knob anyone actually turns) cannot raise below.
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

        # --- NEW (stm): 3D stem, ADAPT of ASDAFMNet -----------------------------------
        # Conv3d over (D, H, W) with D = z.  padding=1 with 'replicate' keeps the slice
        # count intact through the convs -- zero-padding would invent two phantom slices
        # below and above the measured range, and the boundary slices are the informative
        # ones (slice 6 is nearest the molecule).
        widths_3d = [1] + list(conv3d_channels)
        self.conv3d = nn.ModuleList([
            nn.Conv3d(widths_3d[i], widths_3d[i + 1], kernel_size=3, padding=1,
                      padding_mode=padding_mode)
            for i in range(len(conv3d_channels))
        ])

        # The z-pool schedule is fixed at construction, not chosen per batch, because the
        # surviving depth sets the width of the 2D projection below.  It is derivable
        # because D == in_channels is enforced in forward().  Mirrors ASDAFMNet's
        # (2,2,2),(2,2,2),(2,2,1): pool z on every stage but the last, and never pool a
        # depth that would floor to 0.
        n_stages = len(conv3d_channels)
        d = in_channels
        z_kernels = []
        for i in range(n_stages):
            k = 2 if (i < n_stages - 1 and d >= 2) else 1
            z_kernels.append(k)
            d //= k
        self.z_kernels = tuple(z_kernels)
        self.z_out = d

        # Flatten the surviving z into channels (ASDAFMNet's `_flatten_z_to_channels`), then
        # a 1x1 conv to the trunk width -- the same role TersEncoder's spectral stem plays,
        # except the mixing here is over (z x feature) pairs rather than raw spectral bins.
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
        # Mirrors SpecFormer.out_norm (models/specformer.py:67) and TersEncoder.out_norm.
        # cond_lin was initialised for a LayerNorm-scaled input; handing it raw pooled
        # activations changes the scale of what gets added to time_emb at models/dmt.py:354.
        self.out_norm = nn.LayerNorm(output_dim)

        # One-shot input-scale check (docs/06-code-practices.md 5: make silent failures
        # loud).  Forgetting NormalizeStmCube leaves values in 0..65535, which does not
        # crash -- it just trains badly and slowly, which is the expensive failure.  A
        # buffer rather than a plain attribute so it survives DataParallel replication.
        self.register_buffer('_scale_checked', torch.zeros(1, dtype=torch.bool),
                             persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Matches the interface SpecFormer exposes at models/specformer.py:70-73."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # The 3D stem is LeakyReLU, not ReLU; using the ReLU gain here would
                # under-scale it by sqrt(2/(1+0.1^2)).
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
        if bool(self._scale_checked):
            return
        self._scale_checked.fill_(True)
        absmax = float(x.detach().abs().max())
        if absmax > 100.0:
            print(f'[StmEncoder] WARNING: input |max| = {absmax:.4g}. The raw HDF5 is '
                  f'uint16 (0..65535); this looks unnormalised. Wire '
                  f'models.stm_encoder.NormalizeStmCube into the dataset transform -- '
                  f'NOT datasets.ters_dataset.NormalizeVectorized, which standardises each '
                  f'z-slice separately and deletes the decay envelope this encoder reads.')

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

        # (B, C, H, W) -> (B, 1, D, H, W): one input feature map, z as a real spatial axis.
        h = x.unsqueeze(1)
        for i, conv in enumerate(self.conv3d):
            h = F.leaky_relu(conv(h), negative_slope=self.negative_slope)
            # H, W are pooled dynamically so any input resolution works (AdaptiveAvgPool2d
            # in the head absorbs whatever survives), but the z kernel is the fixed
            # schedule from __init__ -- self.flat_channels depends on it.
            sk = 2 if min(h.shape[-2:]) >= 2 else 1
            zk = self.z_kernels[i]
            if sk > 1 or zk > 1:
                h = F.avg_pool3d(h, kernel_size=(zk, sk, sk), stride=(zk, sk, sk))

        # z -> channels.  reshape, not mean: averaging over z here would undo the whole
        # reason for the 3D stem.
        b, c, d, hh, ww = h.shape
        h = h.reshape(b, c * d, hh, ww)

        h = self.proj(h)
        for block in self.blocks:
            h = block(h)
            # Same guard as TersEncoder: stop halving once there is nothing left to halve.
            if min(h.shape[-2:]) >= 2:
                h = self.pool(h)
        return self.out_norm(self.head(h))


# ---------------------------------------------------------------------------------------
# --- NEW (stm): --self-test / --inspect / --check, per docs/06-code-practices.md 3 ------
# ---------------------------------------------------------------------------------------

def _self_test():
    """Exercise the encoder with no HDF5 and no cluster.

    Cases are chosen to cover the failure modes, not to look thorough:
      14 slices @ 128x128  the real corpus geometry
      10 slices @  64x64   the ASD-STM slice count, odd resolution ladder
       1 slice  @ 128x128  degenerate z -- proves the z-pool schedule never floors to 0
      14 slices @  16x16   small input -- proves the spatial guard holds
    """
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

    # The normaliser is the part most likely to be got wrong by reuse, so test the property
    # that motivated it: the z envelope must survive.
    #
    # The cube plants ONE spatial pattern scaled by a decay envelope, rather than 14
    # independently drawn slices.  That is deliberate.  `asinh` is a per-pixel
    # nonlinearity, so a slice mean after the transform depends on the slice's whole
    # distribution, not just its mean -- give every slice its own random content and the
    # per-slice means come out non-monotone *before* normalisation ever runs (measured:
    # 2 inversions in 13 on a seed-0 cube), which would make this test assert something
    # neither this transform nor any other could deliver.  With a shared pattern, strict
    # ordering IS an invariant: a monotone per-pixel map applied to a uniformly scaled
    # slice must preserve the order of the means.  So an inversion here is a real bug.
    base = torch.rand(1, 32, 32) ** 3 * 65535.0
    envelope = torch.linspace(1.0, 0.22, 14).view(14, 1, 1)       # measured decay, 0.083->0.019
    norm = NormalizeStmCube()(base * envelope)
    means = norm.mean(dim=(1, 2))
    assert torch.all(means[:-1] > means[1:]), 'NormalizeStmCube destroyed the z envelope'
    assert abs(float(norm.mean())) < 1e-4 and abs(float(norm.std()) - 1) < 1e-2
    print(f'  ok  NormalizeStmCube keeps the envelope monotone: '
          f'{float(means[0]):+.3f} -> {float(means[-1]):+.3f}')
    ok += 1

    # The contrast that motivated the class: the TERS normaliser on the same cube.  Assert
    # it, do not just claim it in a docstring -- if datasets/ters_dataset.py ever changes
    # so that this no longer holds, the reason this file exists has changed with it.
    from datasets.ters_dataset import NormalizeVectorized
    ters_means = NormalizeVectorized()(base * envelope).mean(dim=(1, 2))
    assert float(ters_means.abs().max()) < 1e-4, (
        'NormalizeVectorized no longer flattens each slice -- re-check whether StmEncoder '
        'still needs its own normaliser')
    print(f'  ok  NormalizeVectorized (TERS) flattens the same envelope to '
          f'{float(ters_means[0]):+.3f} -> {float(ters_means[-1]):+.3f}  (why this class exists)')
    ok += 1

    # And the property that would silently break it: a constant cube must not yield NaN.
    flat = NormalizeStmCube()(torch.full((14, 8, 8), 1234.0))
    assert torch.isfinite(flat).all(), 'constant cube produced non-finite values'
    print('  ok  constant cube -> finite (zero-std guard)')
    ok += 1

    print(f'self-test: {ok}/{len(cases) + 3} passed')
    return ok == len(cases) + 3


def _inspect(h5_path):
    """Print the HDF5 layout and exit.  Run this before anything else touches a new file."""
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
    """Push real cubes through the real normaliser and the real encoder.

    This is the step that turns "the shapes look right" into "this consumes the corpus":
    it reads the z-slice count out of the file rather than trusting the default.
    """
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
