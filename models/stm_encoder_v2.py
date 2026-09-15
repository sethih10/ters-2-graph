"""Deeper 3D conditioning encoder for STM z-stacks.

Same contract as :class:`~models.stm_encoder.StmEncoder` -- ``(B, C, H, W) -> (B, output_dim)``,
consumed by ``cond_lin`` at models/dmt.py:348-349 -- so this is a drop-in swap and nothing in
the denoiser moves.  Selected with ``config.model.stm_encoder = 'v2'``.

Why a second encoder exists
---------------------------
The v1 encoder is not "missing" a 3D convolution; it has three (models/stm_encoder.py:200).
What it has is a *very small* one: widths 4 -> 8 -> 16, one convolution per stage, no
normalisation, no residual path, and a pooling schedule that collapses z from 14 to 3 in the
same three stages that also downsample space 8x.  1.49M parameters against a 29.30M denoiser.

The measurements that motivated this file (docs/09-research-summary.md 5b, jobs 1269011/1268972):

  * At 75k steps the conditioning was worth +0.98% of the loss; at 150k, +1.39%.  Against
    TERS's +10.5%.
  * Zeroing the cube costs +8.5% here and **+1784%** on TERS.
  * Embeddings sit in a cone of mean pairwise cosine **0.967**.
  * Zero exact molecule recoveries at either checkpoint.

The 75k reading was that the cone was a capacity ceiling.  Doubling the training steps
falsified that: the cone *widened* (0.984 -> 0.967) and every conditioning number improved
under nothing but more optimisation.  **The encoder was undertrained, not saturated** -- which
is the argument for spending parameters on it, and this file is that experiment.

Provenance (ladder in docs/README.md; ledger in docs/02-reuse-map.md)
--------------------------------------------------------------------
  * ``ResBlock``, ``_groups_for``  -- **REUSE**, imported unmodified from
        models/ters_encoder.py:47 and :29, for the 2D trunk after the z-collapse.
  * ``NormalizeStmCube``           -- **REUSE**, imported from models/stm_encoder.py:76.
        The input contract is unchanged, so the normaliser must not be duplicated.
  * ``ResBlock3d``                 -- **ADAPT** of ``Conv3dBlock`` in SINGROUP/ml-spm
        (``mlspm/modules.py``): ``depth`` convolutions of kernel 3 per block, activation
        after each, and a residual ``h(x) + x`` whose skip is adapted by a 1x1 convolution
        when the width changes.  Kept: depth-2 default, k=3, replicate padding,
        LeakyReLU(0.1), the residual and its 1x1 adapter.
        **Deviation: GroupNorm is added, where ml-spm has no normalisation at all.**  Two
        reasons, both already settled in this repo: ``nn.DataParallel`` (models/utils.py:27)
        computes BatchNorm statistics per replica, which is why models/ters_encoder.py:47
        chose GroupNorm; and a 6-convolution 3D stack is deep enough that an unnormalised
        LeakyReLU ladder is a training risk the v1 3-layer stem did not carry.
  * stage widths ``(48, 96, 192)`` and the z-pool schedule ``(2, 1, 2)``
                                   -- **ADAPT** of ``AttentionUNet`` / ``EDAFMNet``, the
        *larger* sibling of ASDAFMNet in the same ml-spm codebase, which reads an AFM
        z-stack with three ``Conv3dBlock`` stages of exactly those widths and pools z on the
        first and last stage only.  This is the reference implementation's own answer to
        "what does a serious 3D stack encoder look like", and it is 12x wider than the
        ASDAFMNet ladder v1 took.
  * spatially-strided stem        -- **NEW**.  Not in either reference, and required here:
        see the memory note below.

.. warning::
   **The STM literature does not endorse 3D convolution over the z-stack.**  ASD-STM
   (Kurki & Oinonen et al., *ACS Nano* **18**, 11130 (2024); arXiv:2312.08854), the
   structure-discovery model for *this* modality, feeds a **single randomly chosen slice**
   from its 10-image stack to a 2D attention U-Net -- no 3D convolution anywhere.  The 3D
   stack encoders cited above are all from the *AFM* side.  That is a live argument for
   hypothesis (a) in docs/09 5b (14 z-slices may carry genuinely less joint information than
   100 TERS spectral channels), and this encoder does not refute it -- it tests it.  If a
   12x wider, 2x deeper, normalised, residual 3D stem does not move the cone, (a) is the
   answer and the next move is the measurement, not another architecture.

.. warning::
   The ADAPT rows were reconstructed from the published sources, not by importing ``mlspm``
   (not a dependency here; adding one would violate docs/06-code-practices.md 8).  Shapes,
   gradients and scaling are verified by ``--self-test``; layer-for-layer correspondence is
   not.  Do not load ml-spm checkpoints into this without diffing first.

Memory, which is a design constraint and not an afterthought
------------------------------------------------------------
The blocks are wide and the cube is 14x128x128, so the naive ordering -- widen first, pool
later -- puts a 48-channel activation at full 128x128 resolution.  That is 2.75M floats per
sample, 1.4 GB per stored tensor at ``batch_size=128``, several times over for backward.
The measured v1 training peak is 23.4 GiB of 95 (slurm/train_stm_gpumedium.sbatch).  So the
stem strides spatially **before** the wide blocks, and the widths only grow as the spatial
resolution falls.  Cost is measured by ``--bench``, not asserted here.

Usage
-----
    python -m models.stm_encoder_v2 --self-test              # no data, no cluster, no GPU
    python -m models.stm_encoder_v2 --bench                  # params + peak memory + step time
    python -m models.stm_encoder_v2 --compare                # v1 vs v2, side by side
"""

import torch
import torch.nn.functional as F
from torch import nn

# REUSE, not re-derivation.  If the GroupNorm choice is revisited in ters_encoder, every
# encoder must move together -- which is the point of importing rather than copying.
from .ters_encoder import ResBlock, _groups_for
from .stm_encoder import NormalizeStmCube  # noqa: F401  (re-exported; see module docstring)


class ResBlock3d(nn.Module):
    """A residual 3D convolution block.  ADAPT of ml-spm ``Conv3dBlock`` (see the module
    provenance block), plus GroupNorm.

    ``depth`` convolutions of kernel ``kernel_size``; the first maps ``in_channels ->
    out_channels`` and the rest keep the width.  The skip is an identity when the width is
    unchanged and a 1x1 convolution when it is not, so a same-width block is a true
    shortcut -- the same rule models/ters_encoder.py:70 follows in 2D.

    ``padding_mode='replicate'`` matters on the z axis specifically: zero padding would
    invent phantom slices below z=6 and above z=19, and the boundary slices are the
    informative ones (z=6 is nearest the molecule).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, depth=2, n_groups=8,
                 padding_mode='replicate', negative_slope=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError(f'depth must be >= 1, got {depth}')
        padding = kernel_size // 2                      # 'same': the ladder pools, the convs do not

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
            # Last activation is applied AFTER the residual add, so the skip is not squashed
            # before it is used -- same ordering as ResBlock (models/ters_encoder.py:76).
            if i < len(self.convs) - 1:
                h = F.leaky_relu(h, negative_slope=self.negative_slope)
        return F.leaky_relu(h + self.skip(x), negative_slope=self.negative_slope)


class StmEncoderV2(nn.Module):
    """STM z-stack -> one conditioning vector, via a deep residual 3D stem.

    Input is the same channels-first cube v1 takes, ``(B, C, H, W)`` with ``C`` = z-slices,
    treated internally as a single-channel volume ``(B, 1, D, H, W)`` with ``D = C``.

    Parameters
    ----------
    in_channels : int
        Number of z-slices; 14 for this corpus.  Also ``D``, so the z-pool schedule and the
        width handed to the 2D trunk are fixed at construction from it.
    stem_channels : int
        Width of the strided stem that runs before the wide blocks.  See the memory note in
        the module docstring -- this is the knob that keeps the encoder trainable at
        ``batch_size=128``.
    stem_stride : int
        Spatial stride of that stem.  2 halves H and W once, at width ``stem_channels``,
        which is where it is cheap.  1 disables it (and roughly quadruples block-0 memory).
    conv3d_channels : tuple of int
        Widths of the residual 3D ladder.  ``(48, 96, 192)`` is the EDAFMNet ladder.
    block_depth : int
        Convolutions per 3D block.  2 is the ml-spm default, giving 6 3D convolutions here
        against v1's 3.
    z_pool : tuple of int or None
        Per-stage z pooling.  ``None`` uses EDAFMNet's ``(2, 1, 2)`` pattern, extended or
        truncated to ``len(conv3d_channels)``.  A stage is silently demoted to 1 when the
        surviving depth would floor to 0, so small-``D`` corpora still build.
    filters : tuple of int
        Widths of the 2D ResBlock ladder after the z-collapse.
    output_dim : int
        Must equal ``config.model.nf`` -- ``cond_lin`` is ``Linear(hidden_dim, time_dim)``.
    pool_size : int
        Spatial grid kept by the head before flattening.  Same knob as v1.
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

        # --- NEW: strided stem, before the widths grow.  Stride is spatial only; z is never
        # strided here, because the whole point of the file is to keep z alive longer.
        self.stem = nn.Conv3d(1, stem_channels, kernel_size=3, stride=(1, stem_stride, stem_stride),
                              padding=1, padding_mode=padding_mode)
        self.stem_norm = nn.GroupNorm(_groups_for(stem_channels, n_groups), stem_channels)

        # --- z-pool schedule, fixed at construction because self.flat_channels depends on it.
        n_stages = len(conv3d_channels)
        if z_pool is None:
            pattern = (2, 1, 2)                            # EDAFMNet
            z_pool = tuple(pattern[i % len(pattern)] for i in range(n_stages))
        if len(z_pool) != n_stages:
            raise ValueError(f'z_pool must have one entry per 3D stage, got {len(z_pool)} '
                             f'for {n_stages} stages')
        d = in_channels
        z_kernels = []
        for k in z_pool:
            k = k if (k > 1 and d >= 2 * k) else 1        # never floor the depth to 0
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

        # z -> channels, then 1x1 to the 2D trunk width.  reshape, not mean: averaging over z
        # would undo the entire reason for a 3D stem.
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
        # cond_lin was initialised for a LayerNorm-scaled input; see models/stm_encoder.py:243.
        self.out_norm = nn.LayerNorm(output_dim)

        # Same one-shot loud-failure guard as v1 (docs/06-code-practices.md 5): forgetting
        # NormalizeStmCube leaves values in 0..65535, which trains badly rather than crashing.
        self.register_buffer('_scale_checked', torch.zeros(1, dtype=torch.bool),
                             persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                # The 3D path is LeakyReLU and the 2D ResBlock path is ReLU.  a=0 is the ReLU
                # gain; using it on the LeakyReLU convs under-scales them, which is the bug
                # models/stm_encoder.py:256 calls out.
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
        """Warn once if the cube looks un-normalised.  Copied in behaviour from
        models/stm_encoder.py:272 so both encoders fail the same loud way."""
        if bool(self._scale_checked):
            return
        self._scale_checked.fill_(True)
        m = float(x.detach().abs().max())
        if m > 50.0:
            import warnings
            warnings.warn(
                f'StmEncoderV2 got input with max |x| = {m:.1f}. Raw STM cubes are uint16 '
                f'(0..65535); NormalizeStmCube was probably not applied. Training will be '
                f'slow and bad rather than crashing.', RuntimeWarning, stacklevel=2)

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
            # H, W pool dynamically so any input resolution works (AdaptiveAvgPool2d in the
            # head absorbs the remainder); the z kernel is the fixed schedule from __init__,
            # because self.flat_channels depends on it.
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
# --- NEW: --self-test / --bench / --compare / --cone, per docs/06-code-practices.md 3 ---
# ---------------------------------------------------------------------------------------

def _mean_pairwise_cosine(E):
    """Mean off-diagonal cosine similarity of a (N, d) embedding matrix.

    This is the statistic docs/09-research-summary.md 5b tracks: 0.984 at 75k, 0.967 at
    150k, against TERS's 0.925.  Same definition as tools/diagnose_conditioning.py:101 so
    the numbers are directly comparable -- do not "improve" it independently here.
    """
    cos = torch.nn.functional.cosine_similarity(E[:, None, :], E[None, :, :], dim=-1)
    off = cos[~torch.eye(len(E), dtype=torch.bool, device=E.device)]
    return float(off.mean())


def _self_test():
    """Exercise the encoder with no HDF5, no cluster and no GPU.

    Cases mirror models/stm_encoder.py:327 exactly, so a failure that appears here and not
    there is about this architecture and not about the corpus geometry:
      14 slices @ 128x128  the real corpus geometry
      10 slices @  64x64   the ASD-STM slice count, odd resolution ladder
       1 slice  @ 128x128  degenerate z -- proves the z-pool schedule never floors to 0
      14 slices @  16x16   small input -- proves the spatial guard holds
    """
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

    # Depth is the point of the file, so assert it rather than trusting the docstring.
    enc = StmEncoderV2()
    n3d = sum(1 for m in enc.modules() if isinstance(m, nn.Conv3d) and m.kernel_size != (1, 1, 1))
    assert n3d >= 6, f'expected a deep 3D stem, got {n3d} 3D convolutions'
    print(f'  ok  3D stem is {n3d} convolutions deep (v1 has 3), widths '
          f'{tuple(b.convs[-1].out_channels for b in enc.blocks_3d)}')
    checks += 1

    # A same-width 3D block must be a true identity shortcut, not a 1x1 projection.
    same = ResBlock3d(32, 32)
    diff = ResBlock3d(32, 64)
    assert isinstance(same.skip, nn.Identity), 'same-width block should skip through identity'
    assert isinstance(diff.skip, nn.Conv3d), 'width-changing block needs a 1x1 adapter'
    print('  ok  residual skip: identity when width is unchanged, 1x1 when it is not')
    checks += 1

    # Residual path must actually carry signal: zero every conv weight and the block should
    # reduce to (a LeakyReLU of) its skip, not to zero.  This is what catches a block wired
    # up as plain sequential convolutions by accident.
    b = ResBlock3d(8, 8)
    with torch.no_grad():
        for conv in b.convs:
            conv.weight.zero_()
            conv.bias.zero_()
    x = torch.randn(2, 8, 4, 8, 8).abs() + 0.1        # positive, so LeakyReLU is identity
    assert torch.allclose(b(x), x, atol=1e-5), 'residual path does not carry the input'
    print('  ok  zeroing the convolutions leaves the skip intact (residual is real)')
    checks += 1

    # The un-normalised-input guard must fire exactly once, like v1's.
    import warnings
    enc = StmEncoderV2(in_channels=14)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
    assert len(w) == 1, f'scale warning fired {len(w)} times, expected exactly 1'
    print('  ok  un-normalised input warns once, not every step')
    checks += 1

    # Distinct inputs must produce DISTINCT embeddings -- the weakest possible version of
    # the property the whole project turns on.
    #
    # Note what this does NOT assert.  A randomly initialised encoder of this family sits at
    # mean pairwise cosine ~0.995 on random input -- measured, v1 0.9951 and v2 0.9957 under
    # the same seed -- so a "cone must be wide at init" assertion would fail on v1 too and
    # would be testing a property neither architecture has.  The trained cone (0.984 at 75k,
    # 0.967 at 150k) is *wider* than init, i.e. training pulls these apart rather than
    # collapsing them.  The real measurement is --cone, on real cubes.  Here we only demand
    # that the embeddings are not literally the same vector.
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
    """v1 vs v2, side by side: shapes, depth, width, parameters."""
    from .stm_encoder import StmEncoder

    # pool_size=2 for BOTH. The two classes have different defaults (v1 1, v2 2) and the head
    # is Linear(filters[-1] * pool_size^2, output_dim), so leaving the defaults alone compares
    # v1 with a 4x smaller head and overstates the gap. 2 is what configs/diffspectra_stm.py
    # actually sets for v1, so this is the like-for-like count.
    v1 = StmEncoder(in_channels=in_channels, output_dim=256, pool_size=2)
    v2 = StmEncoderV2(in_channels=in_channels, output_dim=256, pool_size=2)
    x = torch.randn(2, in_channels, hw, hw)

    rows = []
    for name, enc in (('v1 (ASDAFMNet ladder)', v1), ('v2 (EDAFMNet ladder)', v2)):
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
    """Measure what the module docstring refuses to assert: parameters, peak memory and
    forward+backward time, at the real training batch size.

    docs/06-code-practices.md 4: a cost written down without a measurement is a guess, and
    this project has already been wrong by 7.7x that way (docs/05-progress-log.md,
    2026-09-08).  Run it on a GPU; on CPU it reports time only.
    """
    import time
    from .stm_encoder import StmEncoder

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {dev}   batch={batch}  cube={in_channels}x{hw}x{hw}')
    if dev == 'cpu':
        print('  (no GPU here -- peak memory is not measured; run this on gputest)')

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
    """The untrained-encoder cone, on REAL cubes -- the measurement docs/09 5b calls the
    cheapest unrun experiment in the project.

    The question it settles: STM embeddings sit at mean pairwise cosine 0.967 after 150k
    steps.  Is that narrow cone something training *caused* (collapse under a loss with no
    term holding embeddings apart -- hypothesis (c)), or is it where a randomly initialised
    encoder already sits on this data, i.e. a property of the measurement (hypothesis (a))?

    A random-init encoder near 0.97 supports (a).  One far below it supports (c), and makes
    an auxiliary loss the next thing to try rather than a bigger encoder.
    """
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
    print('  reference (tools/diagnose_conditioning.py, real runs): '
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
