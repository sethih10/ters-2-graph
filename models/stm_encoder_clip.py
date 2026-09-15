"""Contrastively **pretrained** 3D conditioning encoder for STM z-stacks.

Same contract as :class:`~models.stm_encoder.StmEncoder` and
:class:`~models.stm_encoder_v2.StmEncoderV2` -- ``(B, C, H, W) -> (B, output_dim)``, consumed
by ``cond_lin`` at models/dmt.py:329 -- so this is a drop-in swap and nothing in the denoiser
moves.  Selected with ``config.model.stm_encoder = 'clip'``.

Why a third encoder exists
--------------------------
v1 and v2 are both trained **from scratch by the diffusion gradient alone**, and
docs/08-faq.md Q5 names the consequence: *"Every term in the objective supervises the
molecule.  Zero terms supervise the cube representation."*  The measured result
(docs/09-research-summary.md 5b, jobs 1269011/1268972) is a conditioning embedding sitting in
a cone of mean pairwise cosine **0.967**, a shuffled-vs-matched loss effect of **+1.39%**
against TERS's +10.5%, and **zero exact molecule recoveries in 400 guesses**.

docs/09 5b ranks the fix third in its own next-steps list: *"Pre-train TersEncoder before the
diffusion run -- the stage upstream has and we do not."*  That stage now exists, for this
modality, in the sibling `ters-clip` folder: InfoNCE between the cube embedding and a
TorchMD-NET ET embedding of the same molecule.  **This file is the encoder that stage trained,
so its weights can be loaded.**  Measured here, on 128 held-out cubes of ``test.h5`` pushed
through this repo's own :class:`~models.stm_encoder.NormalizeStmCube`:

    raw cubes                                  +0.518
    this encoder, pretrained weights           +0.168
    this encoder, random init                  +0.987
    v1, random init                            +0.956
    v1, trained 150k steps (docs/09 5b)        +0.967

Provenance (ladder in docs/README.md; ledger in docs/02-reuse-map.md 26)
-----------------------------------------------------------------------
  * ``StmCubeEncoder``  -- **ADAPT** (verbatim port) of
        /users/harshise/ters-clip/models/stm_cube_encoder.py:45-144.
        Kept **exactly**: every submodule name and every tensor shape, because those names
        *are* the checkpoint -- see the warning below.
        **Deviations, all additive and none of them touching a parameter:**
          1. ``latent_dim`` is accepted as ``output_dim`` (the name the other two STM
             encoders use, because it must equal ``config.model.nf``).  ``self.latent_dim``
             is still set, so cross-repo code reading it keeps working.
          2. ``_check_scale`` added -- v1 (models/stm_encoder.py:272) and v2
             (models/stm_encoder_v2.py:277) both have one, the ters-clip original does not.
             Registered ``persistent=False`` so it stays out of ``state_dict`` and a strict
             load is still 64/64.
          3. ``--self-test`` / ``--load`` / ``--cone`` entry points, per
             docs/06-code-practices.md 3.
          4. The z-pool kernel is clamped to the surviving depth.  **This is a bug fix in
             the original, found by the self-test below**: ters-clip guards only the two
             spatial axes before pooling, so a stack shallower than ``z_strides[i]``
             (``in_channels=1``, say) pools z to size 0 and raises, while ``__init__`` has
             already computed ``flat_channels`` as though it survived.  The clamp is a no-op
             at every depth the original handled -- 14 -> 7 -> 3 -> 3 at the corpus geometry,
             ``flat_channels`` 48 either way -- so **the released weights are unaffected**.
             v1 (models/stm_encoder.py:213) and v2 (models/stm_encoder_v2.py:216) both
             already guard z this way.
  * ``ResBlock3d``      -- **ADAPT** (verbatim port) of
        /users/harshise/ters-clip/models/hsi_conv3d.py:58-83.
        **Why this is NOT a REUSE of models/stm_encoder_v2.py:96**, which is the first thing
        anyone will ask.  That block is *numerically identical* to this one when built as
        ``ResBlock3d(c_in, c_out, depth=2, padding_mode='zeros', negative_slope=0.0)`` --
        verified by copying weights across and diffing the output (max abs diff 0.0).  But it
        stores them under ``convs.0.* / norms.0.*`` where this checkpoint has
        ``conv1.* / norm1.* / conv2.* / norm2.*``, so reusing it would put a **key-remapping
        layer between a checkpoint and the class that consumes it**.  That is precisely the
        silent-failure surface this file exists to remove: the remap is 4 entries today and
        is one rename in v2 away from loading 3 tensors of 64 while printing success.  The
        parameter names are part of the artefact.  15 lines of duplication is the cheaper
        risk, and this comment is the price of it.
  * ``ResBlock``, ``_groups_for``
                        -- **REUSE**, imported unmodified from models/ters_encoder.py:49
        and :36.  Safe *and* key-compatible: ters-clip's models/blocks.py:46 is a declared
        verbatim copy of that class, and ``diff`` of the two executable bodies is clean (only
        a comment differs), so the checkpoint's ``blocks.N.{conv1,norm1,conv2,norm2,skip}``
        land on the imported class unchanged.  Importing rather than copying keeps the
        property models/stm_encoder.py:73 and models/stm_encoder_v2.py:92 both rely on -- if
        the GroupNorm choice is ever revisited, every encoder moves together.
  * ``NormalizeStmCube`` -- **REUSE**, imported from models/stm_encoder.py:76 and already
        wired into the dataset at datasets/stm_dataset.py:40.  ters-clip's copy
        (data/stm_pairs.py:59) is a byte-identical body with identical defaults -- verified
        by ``diff``, zero hunks -- so the encoder sees the same input distribution it was
        trained on.  **This is the reason the port works at all**, and it is not luck:
        ters-clip reused the normaliser *from here*.

.. warning::
   **``pool_size`` must be 1 for the shipped checkpoint, and the checkpoint cannot tell you
   that.**  ``head.2.weight`` is ``Linear(filters[-1] * pool_size**2, output_dim)``; the
   released tensor is ``(256, 256)`` with ``filters[-1] = 256``, so ``pool_size = 1``.  The
   checkpoint's own ``args`` dict records **no** ``cube_pool_size``, ``stem_channels`` or
   ``filters`` -- it predates those flags in ters-clip's train_clip.py -- so a loader that
   trusts ``ckpt['args']`` raises ``KeyError``.  ``pool_size`` is derivable from the weights
   and nowhere else; :func:`load_pretrained_stm_cube_encoder` derives and asserts it.
   Note this **differs from configs/diffspectra_stm.py's ``stm_pool_size = 2``**, which is
   v1's measured setting (docs/experimental.md E-1) and is deliberately not shared here.

.. warning::
   **The InfoNCE target was O(3)-invariant, so this encoder was trained to discard pose.**
   ters-clip aligned the cube embedding to a TorchMD-NET ET structure embedding, which is
   invariant to rotation and reflection (measured there: cosine 1.000000 under both), and the
   corpus ships ~4.21 images of each molecule *in different orientations* mapping to
   byte-identical targets.  Meanwhile datasets/build_dataset.py:155 sets
   ``aug_rotation = False`` for DMT, so the denoiser trains on file-frame coordinates and
   pose is a real part of its target.  A frozen encoder here therefore supplies identity and
   **withholds orientation**.  Two readings, and they point opposite ways: it may cost
   denoising loss, or the current +1.39% shuffled-matched effect may be substantially pose
   rather than identity, in which case trading it is the entire point.  Unmeasured either
   way.  The generation metrics (top-10 exact, Tanimoto, MCES) are graph-level and
   pose-blind, so the loss is the place this would show up.

.. warning::
   **The checkpoint has known defects.**  10.6% of its training batches were discarded as
   non-finite under fp16 AMP; it memorises hard (train top-1 0.993 against val 0.276, with
   validation positive-pair similarity peaking at epoch 3-8 and declining to epoch 50); it is
   a single seed.  Epoch 21 was selected on size-controlled retrieval over chance, not on val
   loss.  Its exported vector probes at **2.58x** chance against a random-init control at
   1.01x -- real signal, and *not* molecule identification.  Do not expect exact recoveries
   from this; expect the shuffled-matched effect to move off +1.39%, and measure that first.

Usage
-----
    python -m models.stm_encoder_clip --self-test     # no data, no cluster, no GPU
    python -m models.stm_encoder_clip --load CKPT     # strict-load the real checkpoint
    python -m models.stm_encoder_clip --cone H5       # pretrained vs random, on real cubes
"""

import torch
import torch.nn.functional as F
from torch import nn

# REUSE, not re-derivation, and key-compatible with the checkpoint -- see the provenance
# block.  If the GroupNorm choice is revisited in ters_encoder, every encoder moves together.
from .ters_encoder import ResBlock, _groups_for
from .stm_encoder import NormalizeStmCube  # noqa: F401  (re-exported; see module docstring)

# The published artefact this file exists to load.  docs/12-results.md in ters-clip calls it
# "THE MODEL"; job 1264302, epoch 21 of 50, 1.31M parameters.
DEFAULT_CKPT = ('/scratch/project_2001912/sethih1/runs/clip-full-1264302/'
                'cube_encoder_best.pt')


class ResBlock3d(nn.Module):
    """A residual 3D convolution block.  ADAPT (verbatim port) of ters-clip's
    models/hsi_conv3d.py:58 -- which is itself that repo's 2D ``ResBlock`` with a third
    convolution axis and nothing else changed.

    Post-activation order (conv-norm-act, conv-norm, add, act), GroupNorm rather than
    BatchNorm for the reason models/ters_encoder.py:52 gives, identity shortcut except where
    the width changes.

    **Do not "clean this up" into models/stm_encoder_v2.py:96.**  The two are numerically
    identical; they do not share parameter names, and the names are what the checkpoint is
    keyed on.  The module docstring has the full argument.
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
    """``(B, C, H, W) -> (B, output_dim)``, with 3D convolutions over the height axis.

    ADAPT (verbatim port) of ters-clip's models/stm_cube_encoder.py:45.  Submodule names and
    tensor shapes are load-bearing: they are the keys of the released checkpoint.

    Parameters
    ----------
    in_channels : int
        Number of z-slices.  14 for this corpus.  Unlike ``TersEncoder`` this is the DEPTH of
        a single-channel volume, not a channel count -- which is the whole point.
    conv3d_channels : tuple of int
        Widths of the 3D stem.  The checkpoint is ``(4, 8, 16)``.
    z_strides : tuple of int
        Per-stage pooling factor along z.  Spatial pooling is 2 at every stage.  The default
        pools z twice and then leaves it, so 14 -> 7 -> 3 -> 3: the axis is reduced but never
        crushed to 1, which is the failure mode ters-ae measured at -20.17 dB.
    filters : tuple of int
        Widths of the 2D ladder after the z-collapse.
    output_dim : int
        Must equal ``config.model.nf`` -- ``cond_lin`` is ``Linear(hidden_dim, time_dim)``
        (models/dmt.py:329).  256 for the current configs, and 256 in the checkpoint.
    pool_size : int
        Spatial grid kept by the head before flattening; 1 is a global average pool.
        **1 for the released checkpoint** -- see the module warning.
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
        # Kept so cross-repo code (and ters-clip's own ProjectionHead contract) still reads.
        self.latent_dim = output_dim
        self.pool_size = pool_size

        # --- 3D stem: convolves ALONG z ---
        self.stem3d = nn.ModuleList()
        self.pool3d = nn.ModuleList()
        c_in = 1
        z = in_channels
        for c_out, zs in zip(conv3d_channels, z_strides):
            self.stem3d.append(ResBlock3d(c_in, c_out, kernel_size, n_groups=n_groups))
            # --- DEVIATION 4 (see the module provenance block): clamp the z kernel to the
            # depth that is actually there.  The ters-clip original stores MaxPool3d((zs,2,2))
            # unconditionally while computing z as max(1, z // zs), so for a stack shallower
            # than zs the two disagree: __init__ believes the axis survives and forward pools
            # it to size 0 and raises.  Clamping here is a no-op for every depth the pool
            # could already handle -- 14 -> 7 -> 3 -> 3 at the corpus geometry, so
            # flat_channels stays 48 and the released weights are unaffected -- and it makes
            # the degenerate case build instead of crash, which is what v1
            # (models/stm_encoder.py:213) and v2 (models/stm_encoder_v2.py:216) both do.
            zs_eff = zs if z // zs >= 1 else 1
            self.pool3d.append(nn.MaxPool3d((zs_eff, 2, 2)))
            c_in = c_out
            z = max(1, z // zs_eff)
        self.z_out = z
        self.c3d_out = c_in
        self.flat_channels = self.c3d_out * self.z_out

        # --- collapse z into channels, then the 2D ladder ---
        # Reshape rather than mean-pool: a mean over z would throw away the envelope the 3D
        # stem just spent three stages reading.  The 1x1 conv that follows is then free to
        # weight the surviving slices, which is a different operation from averaging them.
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
        # ters-clip kept this because its loss is cosine; it matters here for a different
        # reason -- cond_lin was initialised for a LayerNorm-scaled input (models/dmt.py:329,
        # and the same note at models/stm_encoder.py:243).  Either way it stays.
        self.out_norm = nn.LayerNorm(output_dim)

        # --- ADDED, not in the ters-clip original.  persistent=False keeps it OUT of
        # state_dict, so a strict load of the released checkpoint is still 64/64.
        self.register_buffer('_scale_checked', torch.zeros(1, dtype=torch.bool),
                             persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Verbatim from the ters-clip original, so a from-scratch control arm of this class
        starts from the same distribution the pretrained arm was trained from."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _check_scale(self, x):
        """--- ADDED: warn once if the cube looks un-normalised.  Same behaviour as
        models/stm_encoder.py:272 and models/stm_encoder_v2.py:277, so all three STM encoders
        fail the same loud way (docs/06-code-practices.md 5).

        It matters more here than for the other two: this encoder's weights were fitted to
        NormalizeStmCube's output specifically, so a scale mismatch does not merely train
        badly, it silently discards the pretraining.
        """
        if bool(self._scale_checked):
            return
        self._scale_checked.fill_(True)
        m = float(x.detach().abs().max())
        if m > 50.0:
            import warnings
            warnings.warn(
                f'StmCubeEncoder got input with max |x| = {m:.1f}. Raw STM cubes are uint16 '
                f'(0..65535); NormalizeStmCube was probably not applied. These weights were '
                f'trained on its output, so the pretraining is being thrown away.',
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
# --- NEW (stm-clip): the checkpoint loader ---------------------------------------------
# ---------------------------------------------------------------------------------------
# Deliberately NOT a reuse of models/dmt.py:365 load_pretrained_ters_encoder.  That loader
# raises only when *zero* keys match, and this checkpoint is the case that breaks it: its 2D
# half is already key- and shape-identical to StmEncoder v1, so pointing it at v1 under the
# shipped config loads 31 of 40 tensors -- a random 3D stem and a random head -- and prints
# "Loaded 31/40 tensors" without raising.  ters-clip's models/structure_encoder.py:87 names
# that exact behaviour as the bug it was written against:
#
#     "A partial load of a pretrained encoder is not a warning, it is a different experiment."
#
# So this one is all-or-nothing.  Recorded as a refusal in docs/07-change-manifest.md 5.

def read_ckpt_geometry(ckpt_path):
    """Recover the encoder shape from the **weights**, because the metadata does not have it.

    ters-clip's train_clip.py saves ``vars(args)`` alongside the state dict, but the released
    checkpoint predates its ``--cube-pool-size`` / ``--stem-channels`` / ``--filters`` flags,
    so those keys are simply absent and ``ckpt['args']['cube_pool_size']`` is a ``KeyError``.
    Every number below is therefore derived from a tensor shape.

    Returns a dict of exactly the keyword arguments :class:`StmCubeEncoder` needs, except
    ``in_channels`` -- the stem only records ``c3d_out * z_out``, so the slice count is not
    recoverable and stays the corpus's business (``config.model.stm_in_channels``).
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
    # head is Linear(filters[-1] * pool_size**2, output_dim); solve for pool_size and demand
    # that it comes out an exact integer rather than rounding one into place.
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
    """Load a ters-clip ``cube_encoder_*.pt`` into ``encoder``, all or nothing.

    Raises -- never warns -- on a missing key, an unexpected key, or a shape mismatch, and
    the message names the config knob that fixes the common case (``pool_size``).  A run that
    prints "Loaded" must have loaded *everything*, because the alternative is an arm that
    reports pretrained conditioning while carrying a random 3D stem.
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
            f'config.model.stm_clip_pool_size={want}. (Note this is NOT stm_pool_size, which '
            f'is v1\'s knob and is 2.)')
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
    # strict=True does the missing/unexpected check; do it explicitly first so the message
    # says which side is wrong instead of printing two lists of 64 names.
    missing, unexpected = encoder.load_state_dict(sd, strict=True)
    assert not missing and not unexpected          # strict=True raises before reaching here
    if verbose:
        print(f'Loaded {len(sd)}/{len(sd)} tensors into StmCubeEncoder from {ckpt_path} '
              f'(epoch {geom["epoch"]}, pool_size={want}, '
              f'{sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M params)')
    return encoder


# ---------------------------------------------------------------------------------------
# --- NEW (stm-clip): --self-test / --load / --cone, per docs/06-code-practices.md 3 ----
# ---------------------------------------------------------------------------------------

# The 64 state_dict keys of the released checkpoint, by shape of name rather than by listing
# them.  This is the regression guard on the one decision this file turns on: if anyone
# "simplifies" ResBlock3d into models/stm_encoder_v2.py's ModuleList form, these stop
# matching and the self-test fails HERE rather than a loader silently matching 3 of 64.
_EXPECTED_3D_LEAVES = ('conv1.weight', 'conv1.bias', 'norm1.weight', 'norm1.bias',
                       'conv2.weight', 'conv2.bias', 'norm2.weight', 'norm2.bias',
                       'skip.weight', 'skip.bias')


def _self_test():
    """Exercise the encoder with no HDF5, no cluster, no GPU and no checkpoint.

    Cases mirror models/stm_encoder.py:327 and models/stm_encoder_v2.py:353 exactly, so a
    failure that appears here and not there is about this architecture, not the corpus:
      14 slices @ 128x128  the real corpus geometry
      10 slices @  64x64   the ASD-STM slice count, odd resolution ladder
       1 slice  @ 128x128  degenerate z -- proves the z-pool schedule never floors to 0
      14 slices @  16x16   small input -- proves the spatial guard holds
    """
    import tempfile
    import os
    import warnings
    from .stm_encoder_v2 import _mean_pairwise_cosine      # REUSE: one definition of the cone

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

    # --- THE key test: the state_dict must have the names the checkpoint is keyed on. ---
    enc = StmCubeEncoder(in_channels=14, output_dim=256, pool_size=1)
    keys = set(enc.state_dict())
    assert len(keys) == 64, f'expected 64 tensors (the checkpoint has 64), got {len(keys)}'
    expected_3d = {f'stem3d.{i}.{leaf}' for i in range(3) for leaf in _EXPECTED_3D_LEAVES}
    assert expected_3d <= keys, (
        f'3D stem keys do not match the checkpoint. Missing: '
        f'{sorted(expected_3d - keys)[:6]}. If ResBlock3d was replaced with '
        f"models/stm_encoder_v2.py's ModuleList block, THIS is the breakage -- see the "
        f'provenance block in this module.')
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

    # --- the loader, round-tripped through a temp file: no cluster, no real checkpoint. ---
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'fake_cube_encoder.pt')
        torch.save({'state_dict': enc.state_dict(), 'epoch': 7,
                    'cube_encoder_class': 'StmCubeEncoder'}, p)

        geom = read_ckpt_geometry(p)
        assert geom['pool_size'] == 1, f'derived pool_size {geom["pool_size"]} != 1'
        assert geom['output_dim'] == 256, geom['output_dim']
        assert geom['conv3d_channels'] == (4, 8, 16), geom['conv3d_channels']
        assert geom['filters'] == (64, 128, 256), geom['filters']
        print(f'  ok  geometry recovered from WEIGHTS alone: pool_size={geom["pool_size"]}, '
              f'conv3d={geom["conv3d_channels"]}, filters={geom["filters"]}')
        checks += 1

        fresh = StmCubeEncoder(in_channels=14, output_dim=256, pool_size=1)
        load_pretrained_stm_cube_encoder(fresh, p, verbose=False)
        same = all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(),
                                                     enc.state_dict().values()))
        assert same, 'round-tripped weights differ from what was saved'
        print('  ok  strict load round-trips every tensor')
        checks += 1

        # The shipped config's pool_size=2 must RAISE, not partially load.  This is the
        # failure mode docs/07-change-manifest.md 5 records as a deliberate refusal.
        wrong = StmCubeEncoder(in_channels=14, output_dim=256, pool_size=2)
        try:
            load_pretrained_stm_cube_encoder(wrong, p, verbose=False)
            raise AssertionError('pool_size=2 loaded a pool_size=1 checkpoint silently')
        except ValueError as exc:
            assert 'stm_clip_pool_size' in str(exc), \
                f'the error must name the config knob that fixes it, got: {exc}'
        print('  ok  a pool_size mismatch raises and names stm_clip_pool_size')
        checks += 1

        # And the other partial-load route: a checkpoint for a different class.
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

    # A same-width 3D block must be a true identity shortcut, not a 1x1 projection.
    assert isinstance(ResBlock3d(32, 32).skip, nn.Identity), \
        'same-width block should skip through identity'
    assert isinstance(ResBlock3d(32, 64).skip, nn.Conv3d), \
        'width-changing block needs a 1x1 adapter'
    print('  ok  residual skip: identity when width is unchanged, 1x1 when it is not')
    checks += 1

    # The un-normalised-input guard must fire exactly once.  It matters more here than for
    # v1/v2: these weights were fitted to NormalizeStmCube's output specifically.
    enc = StmCubeEncoder(in_channels=14)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
        enc(torch.randn(1, 14, 32, 32) * 20000.0)
    assert len(w) == 1, f'scale warning fired {len(w)} times, expected exactly 1'
    print('  ok  un-normalised input warns once, not every step')
    checks += 1

    # Distinct inputs must give distinct embeddings.  Same caveat as v2's self-test: a
    # randomly initialised encoder of this family sits at ~0.99, so "the cone must be wide"
    # is NOT assertable at init -- it is a property of the trained weights and belongs to
    # --cone on real cubes.  Measured there: +0.168 pretrained against +0.987 random init.
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
    """Strict-load the real checkpoint and report what came back.  Needs the file; this is
    the check the sibling repo does not have -- ters-clip/docs/10-open-problems.md P9:
    *"there is no export test for the thing actually being trained."*
    """
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
    """The measurement this whole file is for: pretrained vs random-init cone on REAL cubes,
    through this repo's own NormalizeStmCube.

    Reference points, all mean pairwise cosine: v1 trained 150k steps sits at 0.967
    (docs/09-research-summary.md 5b); ters-clip reports 0.011 but that is on ``g(h)``, the
    projection head, which its train_clip.py:477 DISCARDS at export -- so it is not the
    number this path gets.  What ``cond_lin`` consumes is ``h``.
    """
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

    for name, enc in (('pretrained (job 1264302 ep21)', trained), ('random init (control)', control)):
        enc = enc.to(dev).eval()
        with torch.no_grad():
            E = torch.cat([enc(normed[i:i + 16]) for i in range(0, len(normed), 16)])
        print(f'  {name:32s} mean pairwise cosine {_mean_pairwise_cosine(E):+.6f}   '
              f'std/|act| {float(E.std(0).mean() / E.abs().mean()):.4f}')
    print('  reference: v1 @150k 0.967, v1 @75k 0.984, TERS @28k 0.925 (docs/09 5b). '
          "ters-clip's 0.011 is on g(h), which is discarded at export.")


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
