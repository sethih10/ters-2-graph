"""STM pipeline checks.

    python -m tests.test_stm

Mirrors tests/test_phases.py: not pytest (not installed in the container, D-2), same
``check()`` reporting, same two-tier structure -- a synthetic tier that runs anywhere, then a
real-data tier that runs against /scratch and **skips cleanly** when it is not mounted.

The synthetic corpus here plants the pathologies that matter for *this* modality, which are
not the TERS ones:

  [PLANTED] an atom at the origin        -- the unpadding trap. ``xyz`` is zero-padded to 54
                                            rows, and a coordinate-based padding test would
                                            silently delete real atoms.
  [PLANTED] a monotone z-decay envelope  -- the property NormalizeVectorized destroys and
                                            NormalizeStmCube must preserve.
  [PLANTED] Cl / Br / S / F              -- elements outside the TERS H/C/N/O map.
  [PLANTED] a constant cube              -- the zero-std guard.
"""

import os
import os.path as osp
import shutil
import tempfile

import numpy as np
import torch

FAILED = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not cond:
        FAILED.append(name)


# --- synthetic corpus -------------------------------------------------------------------

N_Z = 14
HW = 32                       # smaller than the real 128 so the suite stays fast
PAD = 54

# (name, Z, positions) -- chosen to cover the elements the TERS map does not have.
MOLECULES = [
    ('methane',       [6, 1, 1, 1, 1],
     [(0, 0, 0), (1.09, 0, 0), (-.36, 1.03, 0), (-.36, -.51, .89), (-.36, -.51, -.89)]),
    ('chloromethane', [6, 17, 1, 1, 1],
     [(0, 0, 0), (1.78, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]),
    ('bromomethane',  [6, 35, 1, 1, 1],
     [(0, 0, 0), (1.93, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]),
    ('fluoromethane', [6, 9, 1, 1, 1],
     [(0, 0, 0), (1.38, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]),
    ('methanethiol',  [6, 16, 1, 1, 1, 1],
     [(0, 0, 0), (1.82, 0, 0), (-.38, 1.03, 0), (-.38, -.51, .89), (-.38, -.51, -.89),
      (2.16, 1.23, 0)]),
    ('ethanol',       [6, 6, 8, 1, 1, 1, 1, 1, 1],
     [(0, 0, 0), (1.52, 0, 0), (2.05, 1.25, 0), (-.38, -.52, .89), (-.38, -.52, -.89),
      (-.38, 1.03, 0), (1.90, -.53, .89), (1.90, -.53, -.89), (2.99, 1.20, 0)]),
    # [PLANTED] the only molecule here with non-zero formal charges: RDKit assigns N+1 / O-1,
    # so fc = [0, 1, 0, -1, 0, 0, 0]. Without it the "fc holds formal charges" check below is
    # vacuous -- every fc would be 0 and an implementation that wrote zeros, or wrote the
    # fractional partial charge from xyz column 4, would both pass. Geometry is an
    # MMFF-optimised embedding of 'C[N+](=O)[O-]' (randomSeed=0xC0FFEE), the same molecule
    # preprocess_ters.self_test uses for the same purpose.
    ('nitromethane',  [6, 7, 8, 8, 1, 1, 1],
     [(-.653, -.016, .005), (.833, .022, -.014), (1.42, -1.05, -.208), (1.37, 1.118, .185),
      (-.958, -.21, 1.036), (-1.027, .95, -.342), (-.986, -.814, -.661)]),
]

NITRO = 6                     # index of nitromethane in MOLECULES

CONST_CUBE = 1                # index of the molecule whose cube is planted constant


def build_synthetic(h5_dir, n_repeat=3):
    """Write {train,val,test}.h5 in the real STM layout, with pathologies planted."""
    import h5py
    import os

    os.makedirs(h5_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    # The real envelope: per-slice means decay 0.083 -> 0.019 over 14 slices.
    envelope = np.linspace(1.0, 0.22, N_Z).astype(np.float32)

    for split in ('train', 'val', 'test'):
        mols = MOLECULES * n_repeat
        n = len(mols)
        X = np.zeros((n, HW, HW, N_Z), dtype=np.uint16)
        xyz = np.zeros((n, PAD, 5), dtype=np.float32)
        scale = np.full((n,), 1.5e-11, dtype=np.float32)
        cid = np.array([f'{1000 + i}'.encode() for i in range(n)], dtype='S16')

        for i, (name, z, pos) in enumerate(mols):
            # cube: one spatial pattern scaled by the envelope, cubed to match the real skew
            base = (rng.random((HW, HW)).astype(np.float32) ** 3)
            cube = base[:, :, None] * envelope[None, None, :]
            if i % len(MOLECULES) == CONST_CUBE:
                cube = np.full_like(cube, 0.5)          # [PLANTED] constant cube
            X[i] = np.clip(cube * 65535.0, 0, 65535).astype(np.uint16)

            # xyz: zero-padded to 54. Molecule 0's first atom sits AT THE ORIGIN.
            xyz[i, :len(z), :3] = np.asarray(pos, dtype=np.float32)
            xyz[i, :len(z), 3] = np.asarray(z, dtype=np.float32)
            xyz[i, :len(z), 4] = rng.uniform(-0.3, 0.2, len(z))    # partial charges

        with h5py.File(osp.join(h5_dir, f'{split}.h5'), 'w') as f:
            g = f.create_group(split)
            g.create_dataset('X', data=X)
            g.create_dataset('xyz', data=xyz)
            g.create_dataset('scale', data=scale)
            g.create_dataset('cid', data=cid)
            g.attrs['z_slices'] = np.arange(6, 6 + N_Z)
            g.attrs['encoding'] = 'uint16 per-sample max-scaled'
    return len(MOLECULES) * n_repeat


# --- tiers ------------------------------------------------------------------------------

def tier_encoder():
    print('\n=== Encoder (models/stm_encoder.py) ===')
    from models.stm_encoder import StmEncoder, NormalizeStmCube, _self_test
    check('encoder self-test (shapes, gradients, normaliser)', _self_test())

    enc = StmEncoder(in_channels=N_Z, output_dim=256)
    x = torch.randn(2, N_Z, 128, 128)
    y = enc(x)
    check('contract (B,C,H,W) -> (B,256)', y.shape == (2, 256), str(tuple(y.shape)))

    # [PLANTED] the contrast that justifies a separate normaliser
    from datasets.ters_dataset import NormalizeVectorized
    base = torch.rand(1, 16, 16) ** 3 * 65535.0
    envelope = torch.linspace(1.0, 0.22, N_Z).view(N_Z, 1, 1)
    cube = base * envelope
    stm_means = NormalizeStmCube()(cube).mean(dim=(1, 2))
    ters_means = NormalizeVectorized()(cube).mean(dim=(1, 2))
    check('[PLANTED] NormalizeStmCube keeps the z envelope',
          bool(torch.all(stm_means[:-1] > stm_means[1:])),
          f'{float(stm_means[0]):+.3f} -> {float(stm_means[-1]):+.3f}')
    check('[PLANTED] NormalizeVectorized would destroy it (why the class exists)',
          float(ters_means.abs().max()) < 1e-4,
          f'{float(ters_means[0]):+.3f} -> {float(ters_means[-1]):+.3f}')

    # --- ADDED (stm-clip): the pretrained encoder (models/stm_encoder_clip.py).
    from models.stm_encoder_clip import (StmCubeEncoder, read_ckpt_geometry,
                                         _self_test as clip_self_test)
    check('clip encoder self-test (shapes, keys, strict loader)', clip_self_test())

    clip = StmCubeEncoder(in_channels=N_Z, output_dim=256, pool_size=1)
    yc = clip(x)
    check('clip: contract (B,C,H,W) -> (B,256)', yc.shape == (2, 256), str(tuple(yc.shape)))

    # [PLANTED] The state_dict names ARE the checkpoint. models/stm_encoder_v2.py's
    # ResBlock3d is numerically identical to this file's but stores convs.N/norms.N, so
    # "simplifying" the port to reuse it would silently stop the released weights loading.
    # This is the assertion that catches that, with no checkpoint and no data.
    keys = set(clip.state_dict())
    check('[PLANTED] clip state_dict has the checkpoint\'s 64 keys, not v2\'s names',
          len(keys) == 64 and 'stem3d.0.conv1.weight' in keys
          and 'stem3d.0.convs.0.weight' not in keys,
          f'{len(keys)} keys')
    check('[PLANTED] the scale-guard buffer stays out of state_dict',
          '_scale_checked' not in keys)
    check('[PLANTED] clip head at pool_size=1 matches the released (256, 256)',
          tuple(clip.state_dict()['head.2.weight'].shape) == (256, 256),
          str(tuple(clip.state_dict()['head.2.weight'].shape)))

    # The real artefact, when it is mounted. Skipped rather than failed off-cluster, because
    # tier_encoder must stay runnable with no data and no scratch (docs/06 3).
    from models.stm_encoder_clip import DEFAULT_CKPT
    if os.path.exists(DEFAULT_CKPT):
        geom = read_ckpt_geometry(DEFAULT_CKPT)
        check('released checkpoint geometry is the shape we build',
              geom['pool_size'] == 1 and geom['output_dim'] == 256
              and geom['conv3d_channels'] == (4, 8, 16)
              and geom['filters'] == (64, 128, 256), str(geom))
        n_before = float(clip.state_dict()['to_2d.weight'].abs().sum())
        from models.stm_encoder_clip import load_pretrained_stm_cube_encoder
        load_pretrained_stm_cube_encoder(clip, DEFAULT_CKPT, verbose=False)
        check('released checkpoint strict-loads all 64 tensors',
              float(clip.state_dict()['to_2d.weight'].abs().sum()) != n_before)
    else:
        print(f'  SKIP  released checkpoint not mounted ({DEFAULT_CKPT})')


def tier_preprocess(tmp, h5_dir, n_expected):
    print('\n=== Preprocess (datasets/preprocess_stm.py) ===')
    from datasets.preprocess_stm import process_split, self_test, STM_TYPES

    check('preprocess self-test (unpadding, 10-element map)', self_test())
    check('STM_TYPES covers the measured element set', len(STM_TYPES) == 10,
          str(sorted(STM_TYPES)))

    graph_dir = osp.join(tmp, 'graphs')
    for split in ('train', 'val', 'test'):
        s = process_split(osp.join(h5_dir, f'{split}.h5'),
                          osp.join(graph_dir, f'graphs_{split}.pt'))
        check(f'{split}: all molecules survived', s['n_kept'] == s['n_total'],
              f'{s["n_kept"]}/{s["n_total"]}')

    recs = torch.load(osp.join(graph_dir, 'graphs_train.pt'), weights_only=False)
    check('sidecar length matches the corpus', len(recs) == n_expected,
          f'{len(recs)} vs {n_expected}')
    # [PLANTED] methane's first atom is at the origin -- it must not have been unpadded away
    check('[PLANTED] atom at the origin survived unpadding',
          int(recs[0]['num_atom']) == 5, f"{int(recs[0]['num_atom'])} atoms")
    check('h5_index is a real back-pointer',
          [int(r['h5_index']) for r in recs] == list(range(len(recs))))
    check('cid recorded per molecule', all(r.get('cid') for r in recs))
    # fc must be integer FORMAL charges, never the fractional partial charge in xyz col 4.
    # Checked on the planted nitromethane, where the right answer is known and non-zero.
    fcs = torch.cat([r['fc'].flatten() for r in recs]).float()
    check('fc values are integers, not fractional partial charges',
          bool(torch.all(fcs == fcs.round())), f'max |fc| = {float(fcs.abs().max()):.1f}')
    nitro_fc = recs[NITRO]['fc'].tolist()
    check('[PLANTED] nitromethane keeps its N+1 / O-1 formal charges',
          nitro_fc == [0, 1, 0, -1, 0, 0, 0], str(nitro_fc))
    # and the partial charges really are in the file, so the test above is not passing by
    # virtue of the column being absent
    import h5py
    with h5py.File(osp.join(h5_dir, 'train.h5'), 'r') as f:
        partial = f['train']['xyz'][NITRO][:7, 4]
    check('[PLANTED] xyz col 4 does hold fractional partial charges (and is ignored)',
          bool(np.any(np.abs(partial - np.round(partial)) > 1e-3)),
          f'{np.round(partial, 3).tolist()}')
    return graph_dir


def tier_dataset(h5_dir, graph_dir):
    print('\n=== Dataset + collate ===')
    from datasets.build_dataset import CollateStm, EdgeComSpectraTransform
    from datasets.datasets_config import get_dataset_info
    from datasets.stm_dataset import StmGraphDataset
    from torch.utils.data import DataLoader

    info = get_dataset_info('stm_10elem')
    check('stm_10elem has 10 elements', len(info['atom_encoder']) == 10,
          str(list(info['atom_encoder'])))
    check('atom_encoder and atom_decoder agree',
          list(info['atom_encoder']) == info['atom_decoder'])

    transform = EdgeComSpectraTransform(info['atom_encoder'].values(),
                                        include_aromatic=False, use_normalize=True)
    ds = StmGraphDataset(osp.join(h5_dir, 'train.h5'),
                         osp.join(graph_dir, 'graphs_train.pt'),
                         split='train', transform=transform)
    check('cube key resolved through the split group', ds.channels_key == 'train/X',
          ds.channels_key)
    check('in_channels read from the file', ds.in_channels == N_Z, str(ds.in_channels))
    check('z_slices cross-check ran', ds.z_slices is not None and len(ds.z_slices) == N_Z)

    sample = ds[0]
    check('cube arrives as (C,H,W) under .stm',
          hasattr(sample, 'stm') and tuple(sample.stm.shape) == (N_Z, HW, HW),
          str(tuple(sample.stm.shape)))
    check('cube is NOT stored under .ters', not hasattr(sample, 'ters'))
    check('normalised cube is finite', bool(torch.isfinite(sample.stm).all()))

    # [PLANTED] constant cube -> zero-std guard, must be finite not NaN
    const = ds[CONST_CUBE]
    check('[PLANTED] constant cube normalises finite (zero-std guard)',
          bool(torch.isfinite(const.stm).all()) and float(const.stm.abs().max()) < 1e-3,
          f'max |x| = {float(const.stm.abs().max()):.2e}')

    batch = CollateStm()([ds[i] for i in range(4)])
    check('CollateStm stacks the cube into context',
          tuple(batch['context'].shape) == (4, N_Z, HW, HW),
          str(tuple(batch['context'].shape)))

    # fork-safety: an h5py handle must never be inherited across fork()
    dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=CollateStm(),
                    prefetch_factor=1)
    b = next(iter(dl))
    check('DataLoader workers read the HDF5 (fork-safety)',
          bool(torch.isfinite(b['context']).all()))
    ds.close()


def tier_config(h5_dir, graph_dir):
    print('\n=== Config -> dataloader -> model ===')
    from configs.diffspectra_stm import get_config
    from datasets.build_dataset import get_dataloader, get_dataset
    from models.dmt import DMT

    c = get_config()
    c.data.root = h5_dir
    c.data.graph_root = graph_dir
    c.training.batch_size = c.training.eval_batch_size = 4
    c.data.num_workers = 0

    check('config.model.nf == encoder output_dim slot', c.model.nf == 256)
    check('atom_types matches the info dict', c.data.atom_types == 10)
    check('begin_ckpt agrees with n_iters/snapshot_freq',
          c.eval.begin_ckpt == c.training.n_iters // c.training.snapshot_freq,
          f'{c.eval.begin_ckpt} vs {c.training.n_iters // c.training.snapshot_freq}')

    train, _, val, test, _ = get_dataset(c)
    check('get_dataset dispatches on data.name == STM',
          type(train).__name__ == 'StmGraphDataset')
    loader, _, _ = get_dataloader(train, val, test, c)
    batch = next(iter(loader))
    check('config -> dataloader -> batch',
          batch['context'].shape[1] == c.model.stm_in_channels)

    model = DMT(c)
    check('config builds DMT with an StmEncoder matching the data',
          type(model.cond_encoder).__name__ == 'StmEncoder'
          and model.cond_encoder.in_channels == train.in_channels)
    with torch.no_grad():
        v = model.cond_lin(model.cond_encoder(batch['context']))
    check('real batch -> finite conditioning vector',
          bool(torch.isfinite(v).all()) and v.shape[1] == model.cond_lin.out_features)

    # --- v2 encoder (models/stm_encoder_v2.py) through the SAME config and the same batch.
    # The point is the wiring, not the architecture: config.model.stm_encoder='v2' must pick
    # a different class, and that class must consume the real batch and feed cond_lin, whose
    # shape contract is what everything downstream depends on.
    c2 = get_config()
    c2.data.root, c2.data.graph_root = h5_dir, graph_dir
    c2.training.batch_size = c2.training.eval_batch_size = 4
    c2.data.num_workers = 0
    c2.model.stm_encoder = 'v2'
    m2 = DMT(c2)
    check('stm_encoder="v2" builds StmEncoderV2',
          type(m2.cond_encoder).__name__ == 'StmEncoderV2')
    check('v2 is deeper in 3D than v1',
          sum(1 for mod in m2.cond_encoder.modules()
              if isinstance(mod, torch.nn.Conv3d) and mod.kernel_size != (1, 1, 1))
          > sum(1 for mod in model.cond_encoder.modules()
                if isinstance(mod, torch.nn.Conv3d) and mod.kernel_size != (1, 1, 1)))
    with torch.no_grad():
        v2 = m2.cond_lin(m2.cond_encoder(batch['context']))
    check('v2: real batch -> finite conditioning vector of the same shape',
          bool(torch.isfinite(v2).all()) and v2.shape == v.shape)

    # v2 must NOT inherit v1's widths. The config sets stm_conv3d_channels=[4,8,16] for v1,
    # so a shared knob name would build a "deep" encoder 12x too narrow and still call it v2
    # -- the exact bug this namespacing exists to prevent.
    check('v2 ignores v1 width knobs (stm_v2_* namespace)',
          m2.cond_encoder.blocks_3d[-1].convs[-1].out_channels
          == c2.model.stm_v2_conv3d_channels[-1],
          f'{m2.cond_encoder.blocks_3d[-1].convs[-1].out_channels} vs '
          f'{c2.model.stm_v2_conv3d_channels[-1]}')

    # --- ADDED (stm-clip): the pretrained arm through the SAME config and batch.
    c4 = get_config()
    c4.data.root, c4.data.graph_root = h5_dir, graph_dir
    c4.training.batch_size = c4.training.eval_batch_size = 4
    c4.data.num_workers = 0
    c4.model.stm_encoder = 'clip'
    m4 = DMT(c4)
    check('stm_encoder="clip" builds StmCubeEncoder',
          type(m4.cond_encoder).__name__ == 'StmCubeEncoder')
    with torch.no_grad():
        v4 = m4.cond_lin(m4.cond_encoder(batch['context']))
    check('clip: real batch -> finite conditioning vector of the same shape',
          bool(torch.isfinite(v4).all()) and v4.shape == v.shape)

    # clip must read stm_clip_*, never stm_* or stm_v2_*. pool_size is the sharp case: the
    # config sets stm_pool_size=2 for v1 and stm_clip_pool_size=1 for this arm, and a shared
    # name would build a head the released weights cannot load into.
    check('clip ignores v1/v2 knobs (stm_clip_* namespace)',
          m4.cond_encoder.pool_size == c4.model.stm_clip_pool_size
          and m4.cond_encoder.pool_size != c4.model.stm_pool_size,
          f'clip pool_size={m4.cond_encoder.pool_size}, '
          f'stm_pool_size={c4.model.stm_pool_size}')
    check('clip 2D widths come from stm_clip_filters',
          m4.cond_encoder.blocks[-1].conv2.out_channels == c4.model.stm_clip_filters[-1])

    # [PLANTED] A pretrained path on a NON-clip encoder must RAISE, not partially load.
    # This checkpoint's 2D half is key- and shape-identical to v1's, so the repo's existing
    # load_pretrained_ters_encoder would match 31 of v1's 40 tensors -- random 3D stem,
    # random head -- and print "Loaded 31/40 tensors" without raising.
    c5 = get_config(); c5.data.root, c5.data.graph_root = h5_dir, graph_dir
    c5.model.stm_encoder = 'v1'
    c5.model.pretrained_stm_encoder_path = '/nonexistent/cube_encoder_best.pt'
    try:
        DMT(c5)
        check('[PLANTED] pretrained path on a non-clip encoder raises', False,
              'it built something instead')
    except ValueError as exc:
        check('[PLANTED] pretrained path on a non-clip encoder raises',
              'clip' in str(exc), str(exc)[:70])

    # An unknown version must raise rather than silently falling back to v1.
    # NOTE: the sentinel is deliberately NOT 'v3' any more -- 'clip' took the third slot and
    # a future 'v3' would disarm this check by becoming a real value. 'v9' is unclaimed.
    c3 = get_config(); c3.data.root, c3.data.graph_root = h5_dir, graph_dir
    c3.model.stm_encoder = 'v9'
    try:
        DMT(c3)
        check('an unknown stm_encoder value raises', False, 'it built something instead')
    except ValueError as exc:
        check('an unknown stm_encoder value raises',
              'v1' in str(exc) and 'v2' in str(exc) and 'clip' in str(exc),
              str(exc)[:70])

    # The TERS path must still build from the same code.
    from configs.diffspectra_ters import get_config as ters_config
    tc = ters_config()
    tmodel = DMT(tc)
    check('TERS branch still builds a TersEncoder (no regression)',
          type(tmodel.cond_encoder).__name__ == 'TersEncoder')


REAL_DIR = '/scratch/project_2001912/stm-data'
REAL_GRAPHS = '/scratch/project_2001912/stm-graphs'


def tier_real():
    """Checks against the real corpus; skipped cleanly when it is not mounted."""
    import os

    print('\n=== Real data (/scratch STM corpus) ===')
    real_h5 = osp.join(REAL_DIR, 'val.h5')
    if not osp.exists(real_h5):
        print(f'  SKIP  {real_h5} not mounted')
        return

    import h5py
    from datasets.preprocess_stm import census
    from models.stm_encoder import NormalizeStmCube, StmEncoder

    with h5py.File(real_h5, 'r') as f:
        g = f['val']
        check('cube is (N, 128, 128, 14)', g['X'].shape[1:] == (128, 128, 14),
              str(g['X'].shape))
        check('z_slices attr matches the channel count',
              len(g.attrs['z_slices']) == g['X'].shape[3],
              str(list(g.attrs['z_slices'])))
        check('xyz is zero-padded to 54 rows', g['xyz'].shape[1:] == (54, 5),
              str(g['xyz'].shape))
        cubes = np.asarray(g['X'][:8]).astype(np.float32)

    rep = census(real_h5, limit=2000)
    check('no element outside STM_TYPES', not rep['unknown_elements'],
          str(rep['unknown_elements']) or 'none')
    check('element census is the expected 10', len(rep['elements']) == 10,
          str(list(rep['elements'])))

    stack = torch.from_numpy(cubes).permute(0, 3, 1, 2).contiguous()
    normed = torch.stack([NormalizeStmCube()(c) for c in stack])
    per_slice = normed.mean(dim=(0, 2, 3))
    check('real cubes: z envelope survives normalisation',
          bool(torch.all(per_slice[:-1] > per_slice[1:])),
          f'{float(per_slice[0]):+.3f} -> {float(per_slice[-1]):+.3f}')

    enc = StmEncoder(in_channels=14, output_dim=256).eval()
    with torch.no_grad():
        y = enc(normed)
    check('real cubes -> finite 256-d vector',
          y.shape == (8, 256) and bool(torch.isfinite(y).all()), str(tuple(y.shape)))

    if not osp.exists(osp.join(REAL_GRAPHS, 'graphs_val.pt')):
        print(f'  SKIP  {REAL_GRAPHS}/graphs_val.pt not built yet '
              f'(run: python -m datasets.preprocess_stm --data-dir {REAL_DIR} '
              f'--out-dir {REAL_GRAPHS} --stats)')
        return

    from datasets.stm_dataset import StmGraphDataset
    ds = StmGraphDataset(osp.join(REAL_DIR, 'val.h5'),
                         osp.join(REAL_GRAPHS, 'graphs_val.pt'), split='val')
    check('real sidecar joins the real cubes', len(ds) > 0, f'{len(ds)} molecules')
    s = ds[0]
    check('real sample carries a (14,128,128) stack',
          tuple(s.stm.shape) == (14, 128, 128), str(tuple(s.stm.shape)))
    check('real sample graph is non-empty',
          int(s.num_atom) > 0 and s.edge_index.numel() > 0,
          f'{int(s.num_atom)} atoms')
    ds.close()


def tier_metrics():
    """--- ADDED (eval): the evaluation path, which runs through tools/ and not sampling.py.

    Synthetic only -- no checkpoint, no /scratch. What it guards is the pair of
    ``evaluation/stability.py`` edits that made the metrics reachable for a cube corpus at
    all, plus the ``mol_process`` switch in ``tools/sample_cond.py`` that feeds them.

    The two planted pathologies here are both REGRESSIONS THAT PRINTED A NUMBER rather than
    raising, which is the failure mode this project keeps hitting:

      [PLANTED] a 3D metric that used to raise    -- 'Fail to get dataset bond info.' (D-8)
      [PLANTED] a 2D stability that used to be 0  -- an early return, not a score (D-11)
    """
    print('\n=== Metrics (evaluation/stability.py, tools/) ===')
    from datasets.datasets_config import get_dataset_info
    from evaluation import get_edm_metric, get_2D_edm_metric
    from evaluation.stability import check_stability, check_2D_stability
    from sampling import mol_process

    info = get_dataset_info('stm_10elem')

    # CH3Cl and CH3Br: five atoms, every valence satisfied, and both contain an element the
    # TERS H/C/N/O map does not have. stm_10elem indices are H0 C1 N2 O3 F4 Si5 P6 S7 Cl8 Br9.
    B, N, A = 2, 5, 10
    one_hot = torch.zeros(B, N, A)
    pos = torch.zeros(B, N, 3)
    fc = torch.zeros(B, N, 1)
    edges = torch.zeros(B, N, N, dtype=torch.long)
    geom = [(0, 0, 0), (1.78, 0, 0), (-.37, 1.03, 0), (-.37, -.51, .89), (-.37, -.51, -.89)]
    for i, types in enumerate([[1, 8, 0, 0, 0], [1, 9, 0, 0, 0]]):
        for j, (t, p) in enumerate(zip(types, geom)):
            one_hot[i, j, t] = 1.0
            pos[i, j] = torch.tensor(p)
        for j in range(1, N):                      # single bonds from the carbon
            edges[i, 0, j] = edges[i, j, 0] = 1

    # tools/sample_cond.py must use mol_process, NOT mol_process_2D: the 2D variant sets
    # pos=None (sampling.py:48) and the whole 3D suite is then impossible to run.
    processed = mol_process(one_hot, pos, fc, [N, N], edges)
    check('mol_process keeps positions (sample_cond feeds the 3D metric)',
          len(processed[0]) == 4 and processed[0][0] is not None
          and tuple(processed[0][0].shape) == (N, 3),
          f'{len(processed[0])} fields, pos {tuple(processed[0][0].shape)}')

    # [PLANTED] D-8: 'STM' used to fall past the QM9/Geom branches into a bare raise.
    try:
        stab_3d, rdkit_3d, _ = get_edm_metric(info)(processed)
        ok_3d = True
    except ValueError as exc:
        stab_3d, rdkit_3d, ok_3d = {}, {}, False
        check('[PLANTED] 3D metric runs for name="STM"', False, str(exc))
    if ok_3d:
        check('[PLANTED] 3D metric runs for name="STM" (was: Fail to get dataset bond info.)',
              stab_3d['mol_stable'] == 1.0,
              f"mol_stable={stab_3d['mol_stable']:.3f}")

    # [PLANTED] D-11: 'STM' hit `return 0, 0` before the kekulize-and-count block, so every
    # log printed mol_stable 0.0 beside a real Validity. Two fully-valent molecules must now
    # score 1.0, and the atom count must be the real one rather than a placeholder.
    stab_2d, rdkit_2d, mols_2d = get_2D_edm_metric(info)(processed)
    check('[PLANTED] 2D stability is measured, not an early return (D-11)',
          stab_2d['mol_stable'] == 1.0 and stab_2d['atom_stable'] == 1.0,
          f"mol_stable={stab_2d['mol_stable']:.3f} atom_stable={stab_2d['atom_stable']:.3f}")
    check('2D metric still reports validity for the same mols',
          rdkit_2d['Validity'] == 1.0, f"Validity={rdkit_2d['Validity']:.3f}")
    check('positions reach the rdkit mols (conformer attached)',
          all(m.GetNumConformers() == 1 for m in mols_2d))

    # Both edits are additive. QM9 must be untouched, and an unrecognised corpus must still
    # fail loudly rather than silently scoring against the wrong bond table.
    qm9 = get_dataset_info('qm9_with_h')
    oh = torch.zeros(1, 5, 5); oh[0, 0, 1] = 1.0
    mp = torch.tensor([[[0., 0, 0], [1.09, 0, 0], [-.36, 1.03, 0],
                        [-.36, -.51, .89], [-.36, -.51, -.89]]])
    me = torch.zeros(1, 5, 5, dtype=torch.long)
    for j in range(1, 5):
        oh[0, j, 0] = 1.0
        me[0, 0, j] = me[0, j, 0] = 1
    methane = mol_process(oh, mp, torch.zeros(1, 5, 1), [5], me)
    check('QM9 3D branch unchanged (methane still stable)',
          get_edm_metric(qm9)(methane)[0]['mol_stable'] == 1.0)
    check('QM9 2D branch unchanged (methane still stable)',
          get_2D_edm_metric(qm9)(methane)[0]['mol_stable'] == 1.0)

    p = processed[0]
    unknown = {'name': 'Nonesuch', 'atom_decoder': info['atom_decoder']}
    try:
        check_stability(p[0], p[1], unknown)
        check('unknown corpus still raises in 3D', False, 'it returned instead')
    except ValueError:
        check('unknown corpus still raises in 3D', True)
    check('unknown corpus still early-returns in 2D (not a wrong number)',
          check_2D_stability(p[0], p[1], p[3], p[2],
                             {**unknown, 'atom_fc_num': info['atom_fc_num']})[:2] == (0, 0))

    # --- ADDED (scale): the parallel stability path must agree with the serial one EXACTLY.
    # A parallel metric that disagrees with its serial version is worse than no metric, and the
    # recombination is not trivial: the factory returns FRACTIONS, so each is multiplied back by
    # its own denominator (mol_stable by molecules, atom_stable by atoms), and Unique/Novelty are
    # set operations over the whole cell, so eval_rdmol has to run once in the parent rather than
    # per chunk. Both halves are checked here.
    from tools.eval_cond import stability_metrics
    # A DELIBERATELY MIXED cell. With every molecule stable the fractions are all 1.0 and the
    # recombination arithmetic is untested -- 1.0 recombines correctly under any wrong formula.
    # So plant an unstable one: a carbon with a single bond, which no valence table allows. The
    # molecule counts and atom counts then differ between chunks (5 atoms vs 2), which is exactly
    # what a per-chunk average of averages would get wrong.
    bad_oh = torch.zeros(1, N, A); bad_pos = torch.zeros(1, N, 3)
    bad_fc = torch.zeros(1, N, 1); bad_ed = torch.zeros(1, N, N, dtype=torch.long)
    bad_oh[0, 0, 1] = 1.0                        # C
    bad_oh[0, 1, 0] = 1.0                        # H  -- CH, one bond short of anything valid
    bad_pos[0, 1] = torch.tensor([1.09, 0.0, 0.0])
    bad_ed[0, 0, 1] = bad_ed[0, 1, 0] = 1
    unstable = mol_process(bad_oh, bad_pos, bad_fc, [2], bad_ed)
    mixed = (processed + unstable) * 27          # 81 molecules, 2/3 stable, ragged chunk sizes
    big = mixed
    for kind in ('3d', '2d'):
        ser_stab, ser_rdkit, _ = stability_metrics(big, info, kind, None, workers=1)
        par_stab, par_rdkit, par_mols = stability_metrics(big, info, kind, None, workers=4)
        same = (all(abs(ser_stab[k] - par_stab[k]) < 1e-12 for k in ser_stab)
                and all(abs(ser_rdkit[k] - par_rdkit[k]) < 1e-12 for k in ser_rdkit))
        # A mixed cell must not score 1.0; if it does, the planted molecule is not unstable
        # and the comparison above proves nothing.
        check(f'parallel {kind} stability == serial (mixed cell)',
              same and len(par_mols) == len(big) and ser_stab['mol_stable'] < 1.0,
              f"ser {ser_stab['mol_stable']:.6f}/{ser_stab['atom_stable']:.6f} "
              f"par {par_stab['mol_stable']:.6f}/{par_stab['atom_stable']:.6f}")

    # --- ADDED (scale): DataParallel batch padding arithmetic.
    # DMT.forward reshapes a FLATTENED edge_mask by the xh batch size (models/dmt.py:436), so an
    # uneven DataParallel split makes the two describe different molecules. sample_cond pads each
    # chunk up to a multiple of the device count and drops the extras; if the drop were ever off,
    # every later target's candidates would shift by one, silently.
    for n_real, n_gpu, want in [(100, 4, 0), (101, 4, 3), (5, 4, 3), (23, 1, 0), (8, 4, 0)]:
        pad = (-n_real) % n_gpu if n_gpu > 1 else 0
        check(f'pad {n_real} molecules across {n_gpu} GPU(s) -> +{want}',
              pad == want and (n_real + pad) % n_gpu == 0, f'got +{pad}')

    # --- ADDED (scale): the portable export. It runs at the very END of a run that may have
    # taken hours, which is the worst place to discover a typo -- so it is a module-level
    # function and is exercised here on a hand-built cell instead.
    import tempfile as _tf
    from tools.sample_cond import write_portable
    _, _, mols_for_sdf = get_2D_edm_metric(info)(processed)
    fake_results = [{'mol': mi, 'ds_index': 1000 + mi, 'truth_smiles': 'C',
                     'samples': [{'sample': 0, 'smiles': 'C', 'valid': True,
                                  'tanimoto': 0.5, 'exact': False}]}
                    for mi in range(len(processed))]
    with _tf.TemporaryDirectory() as td:
        n_sdf, n_rows = write_portable(td, 'test', '', fake_results, mols_for_sdf, 1)
        check('write_portable emits one csv row per sample', n_rows == len(fake_results))
        # sanitize=False: an invalid generated molecule is writeable but not strictly
        # re-readable, and dropping it would be worse than keeping it.
        from rdkit import Chem as _C
        recs = [m for m in _C.SDMolSupplier(osp.join(td, 'generated_test.sdf'),
                                            removeHs=False, sanitize=False)]
        sdf_keys = [(m.GetProp('ds_index'), m.GetProp('sample')) for m in recs if m is not None]
        csv_keys = [(r['ds_index'], r['sample'])
                    for r in __import__('csv').DictReader(
                        open(osp.join(td, 'generated_test.csv')))]
        check('SDF and CSV are written in the same order',
              len(recs) == n_sdf and sdf_keys == csv_keys[:len(sdf_keys)],
              f'{n_sdf} sdf records, {n_rows} csv rows')

    # The cube attribute probe both tools rely on.
    from tools.sample_cond import cube_of, MODALITY_DEFAULTS
    check('--modality knows both corpora', sorted(MODALITY_DEFAULTS) == ['stm', 'ters'])
    # A real PyG Data with a graph but no cube -- what a sidecar-only dataset actually serves
    # when load_cubes=False. The message must name what it looked for AND what was there.
    from torch_geometric.data import Data
    for label, sample in [('graph without a cube', Data(pos=torch.zeros(3, 3))),
                          # ...and an object with no PyG API at all, because cube_of's own
                          # `data.keys` diagnostic used to raise over the top of its message.
                          ('object with no keys', torch.nn.Module())]:
        try:
            cube_of(sample)
            check(f'cube_of raises on a {label}', False, 'it returned instead')
        except AttributeError as exc:
            check(f'cube_of raises on a {label}',
                  'ters' in str(exc) and 'stm' in str(exc), str(exc)[:70])


def tier_resume(tmp):
    """The training resume path: save_checkpoint -> restore_checkpoint round-trip.

    Guards a bug that reached a real run. utils.restore_checkpoint is the ONLY loader in the
    repo that omitted weights_only=False, and torch 2.6 flipped that default to True. Our
    checkpoints carry numpy scalars in the optimizer/EMA state, which the restricted unpickler
    refuses -- so the first genuine resume would have died on `Weights only load failed`.
    Nothing caught it because both prior runs started from step 0, where restore_checkpoint
    returns before it ever loads anything.
    """
    print('\n=== Resume (checkpoint round-trip) ===')
    from utils import restore_checkpoint, save_checkpoint

    # save_checkpoint does not create its parent; run_lib.py:213 does that for the real run.
    ckpt_dir = osp.join(tmp, 'checkpoints-meta', 'checkpoint.pth')
    os.makedirs(osp.dirname(ckpt_dir), exist_ok=True)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # A numpy scalar is what the restricted unpickler trips on, and it is not contrived: the
    # real checkpoint's pickle carries `numpy._core.multiarray scalar`.
    class _Ema:
        def __init__(self):
            self._s = {'decay': np.float32(0.999), 'num_updates': np.int64(7)}

        def state_dict(self):
            return self._s

        def load_state_dict(self, s):
            self._s = s

    ema = _Ema()
    # A real backward, because Adam only materialises its moment buffers once there are
    # gradients -- and empty moments are exactly what a broken resume would silently produce.
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    model.weight.data.fill_(0.25)
    save_checkpoint(ckpt_dir, dict(optimizer=opt, model=model, ema=ema, step=75001))
    check('save_checkpoint wrote a file', osp.exists(ckpt_dir))

    fresh = torch.nn.Linear(4, 4)
    fresh.weight.data.fill_(0.0)
    state = dict(optimizer=torch.optim.Adam(fresh.parameters(), lr=1e-3),
                 model=fresh, ema=_Ema(), step=0)
    try:
        state = restore_checkpoint(ckpt_dir, state, 'cpu')
        loaded = True
        detail = ''
    except Exception as exc:
        loaded = False
        detail = f'{type(exc).__name__}: {str(exc)[:80]}'
    check('restore_checkpoint loads a checkpoint holding numpy scalars', loaded, detail)
    if not loaded:
        return

    check('step is restored, so training does not silently start over',
          state['step'] == 75001, str(state['step']))
    check('model weights are restored', bool(torch.allclose(fresh.weight.data,
                                                            torch.full((4, 4), 0.25))))
    check('optimizer state is restored, not reinitialised',
          len(state['optimizer'].state_dict()['state']) > 0)
    check('ema state is restored', state['ema'].state_dict()['num_updates'] == 7)

    # A missing file must NOT raise -- run_lib relies on that to start a fresh run.
    absent = dict(optimizer=opt, model=model, ema=ema, step=0)
    same = restore_checkpoint(osp.join(tmp, 'nope', 'checkpoint.pth'), absent, 'cpu')
    check('a missing checkpoint returns the state untouched', same['step'] == 0)


def main():
    tmp = tempfile.mkdtemp(prefix='stm-tests-')
    try:
        h5_dir = osp.join(tmp, 'h5')
        n = build_synthetic(h5_dir)
        tier_encoder()
        graph_dir = tier_preprocess(tmp, h5_dir, n)
        tier_dataset(h5_dir, graph_dir)
        tier_config(h5_dir, graph_dir)
        tier_metrics()
        tier_resume(tmp)
        tier_real()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f'{len(FAILED)} FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
