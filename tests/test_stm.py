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


def main():
    tmp = tempfile.mkdtemp(prefix='stm-tests-')
    try:
        h5_dir = osp.join(tmp, 'h5')
        n = build_synthetic(h5_dir)
        tier_encoder()
        graph_dir = tier_preprocess(tmp, h5_dir, n)
        tier_dataset(h5_dir, graph_dir)
        tier_config(h5_dir, graph_dir)
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
