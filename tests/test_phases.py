"""Phase 1-4 checks that run without the cluster.

    python -m tests.test_phases

Builds a synthetic corpus (tests/make_synthetic.py), then asserts the properties that would
otherwise only fail on the GPU cluster hours into a run. The final block additionally runs
against the real planar TERS file when it is mounted, and skips cleanly when it is not. Planted-pathology tests are marked
[PLANTED]; those are the ones that catch silent corruption rather than crashes.

Not pytest -- pytest is not installed in the container (docs/05-progress-log.md D-2).
"""

import os.path as osp
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn

FAILED = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not cond:
        FAILED.append(name)


def phase1(tmp, h5_dir):
    from datasets.preprocess_ters import process_split, self_test
    print('\n=== Phase 1: geometry -> graph ===')
    # self_test() returns the list of FAILURES, so it is falsy on success -- matching the CLI,
    # which does `raise SystemExit(1 if self_test() else 0)`.
    failures = self_test()
    check('self-test (6 molecules, exact SMILES recovery)', not failures, str(failures or ''))
    graph_dir = osp.join(tmp, 'graphs')
    for split in ('train', 'val', 'test'):
        s = process_split(osp.join(h5_dir, f'{split}.h5'),
                          osp.join(graph_dir, f'graphs_{split}.pt'))
        check(f'{split}: all molecules survived', s['n_kept'] == s['n_total'],
              f'{s["n_kept"]}/{s["n_total"]}')
    return graph_dir


def phase2(h5_dir, graph_dir):
    from datasets.build_dataset import CollateTers, EdgeComSpectraTransform
    from datasets.datasets_config import get_dataset_info
    from datasets.ters_dataset import TersGraphDataset
    from tests.make_synthetic import CONST_CHANNEL, DEAD_CHANNEL
    from torch.utils.data import DataLoader

    print('\n=== Phase 2: dataset + collate ===')
    info = get_dataset_info('qm9_second_half')
    tf = EdgeComSpectraTransform(info['atom_encoder'].values(), include_aromatic=False,
                                 use_normalize=True)
    ds = TersGraphDataset(osp.join(h5_dir, 'val.h5'), osp.join(graph_dir, 'graphs_val.pt'),
                          channels_key='channels_100', transform=tf)
    s = ds[0]
    check('cube is (C, H, W) after permute', tuple(s.ters.shape) == (100, 8, 8))
    check('in_channels exposed for the encoder', ds.in_channels == 100)
    check('num_nodes inferred from pos', s.num_nodes == len(s.pos))

    ch = s.ters.reshape(100, -1)
    live = [c for c in range(100) if c not in (DEAD_CHANNEL, CONST_CHANNEL)]
    check('live channels standardised', bool((ch[live].std(1) - 1).abs().max() < 1e-3))
    check('[PLANTED] dead channel -> 0, not NaN',
          bool(torch.isfinite(ch[DEAD_CHANNEL]).all() and ch[DEAD_CHANNEL].abs().max() < 1e-3))
    check('[PLANTED] constant channel -> 0, not NaN',
          bool(torch.isfinite(ch[CONST_CHANNEL]).all() and ch[CONST_CHANNEL].abs().max() < 1e-3))

    batch = CollateTers()([ds[i] for i in range(4)])
    B, N = batch['atom_one_hot'].shape[:2]
    nm, em = batch['atom_mask'], batch['edge_mask']
    check('context stacked to (B, C, H, W)', tuple(batch['context'].shape) == (4, 100, 8, 8))
    check('edge_mask flattened to (B*N*N, 1)', tuple(em.shape) == (B * N * N, 1))
    expect = nm.unsqueeze(1) * nm.unsqueeze(2) * (~torch.eye(N, dtype=bool)).unsqueeze(0)
    check('inherited mask convention reproduced exactly',
          torch.equal(em.reshape(B, N, N), expect))
    check('padding beyond num_atom is zero',
          all(bool((batch['positions'][i, int(nm[i].sum()):] == 0).all()) for i in range(B)))

    # Fork-safety: read in the parent FIRST, which is the case the lazy open exists to survive.
    _ = ds[1]
    dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=CollateTers(), prefetch_factor=1)
    seen = sum(b['atom_mask'].shape[0] for b in dl)
    check('DataLoader(num_workers=2) after a parent-side read', seen == len(ds), f'{seen} samples')


def phase3():
    from models.ters_encoder import TersEncoder
    print('\n=== Phase 3: conditioning encoder ===')
    enc = TersEncoder(in_channels=100, output_dim=256)

    enc.eval()
    with torch.no_grad():
        shapes_ok = all(enc(torch.randn(2, 100, h, w)).shape == (2, 256)
                        for h, w in [(256, 256), (16, 16), (8, 8), (1, 1)])
    check('(B,100,H,W) -> (B,256) for cubes 256x256 down to 1x1', shapes_ok)

    bn = [n for n, m in enc.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    check('no BatchNorm (DataParallel computes BN stats per replica)', not bn, str(bn))

    # The property BatchNorm would destroy: a sample's output must not depend on its batch.
    enc.train()
    x = torch.randn(4, 100, 16, 16)
    together = enc(x)
    alone = torch.cat([enc(x[i:i + 1]) for i in range(4)])
    check('batch-independent in train mode', bool((together - alone).abs().max() < 1e-5),
          f'max delta {(together - alone).abs().max():.1e}')
    contaminated = x.clone()
    contaminated[1:] = torch.randn(3, 100, 16, 16) * 100
    check('sample 0 unmoved when the rest of the batch is scaled 100x',
          bool((enc(contaminated)[0] - together[0]).abs().max() == 0))

    # NB: .sum() would report every parameter dead -- LayerNorm's normalised components sum to
    # exactly 0, so d(sum)/d(input) == 0. Project onto noise instead.
    enc.zero_grad()
    y = enc(torch.randn(2, 100, 16, 16))
    (y * torch.randn_like(y)).sum().backward()
    dead = [n for n, p in enc.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    check('every parameter receives gradient', not dead, f'{len(dead)} dead')

    check('reset_parameters/out_norm interface matches SpecFormer',
          hasattr(enc, 'reset_parameters') and isinstance(enc.out_norm, nn.LayerNorm))

    for bad, why in [(torch.randn(2, 100, 16), 'rank'), (torch.randn(2, 400, 16, 16), 'channels')]:
        try:
            enc(bad)
            check(f'wrong {why} raises', False)
        except ValueError:
            check(f'wrong {why} raises ValueError', True)


def phase4(h5_dir, graph_dir):
    from configs.base_ters import get_config as base_cfg
    from configs.diffspectra_ters import get_config as main_cfg
    from datasets.build_dataset import CollateTers, get_dataloader, get_dataset

    print('\n=== Phase 4: configuration ===')
    c, b = main_cfg(), base_cfg()
    check('both configs import on a CPU-only node (guarded // num_gpus)',
          c.training.num_gpus >= 1)
    check('warmup < n_iters, so the LR reaches target', c.optim.warmup < c.training.n_iters,
          f'{c.optim.warmup} < {c.training.n_iters}')
    check('ters_in_channels agrees with ters_key',
          c.model.ters_in_channels == int(c.data.ters_key.split('_')[1]))
    check('begin_ckpt == n_iters / snapshot_freq',
          c.training.n_iters // c.training.snapshot_freq == c.eval.begin_ckpt)
    check("save_mols is the lowercase string 'false'", c.eval.save_mols == 'false')
    check('position normalize_factor > 1',
          float(c.model.normalize_factors.split(',')[0]) > 1)
    check('SpecFormer-only keys removed',
          not any(k in c.model for k in ('patch_len', 'stride', 'pretrained_specformer_path')))
    check('base and main agree on ters_key / include_aromatic / atom_types',
          (b.data.ters_key, b.data.include_aromatic, b.data.atom_types)
          == (c.data.ters_key, c.data.include_aromatic, c.data.atom_types))

    # Point both at the synthetic corpus and drive the real entry points.
    for cfg in (c, b):
        cfg.data.root, cfg.data.graph_root = h5_dir, graph_dir
        cfg.data.info_name = 'qm9_second_half'      # 'ters_hcno' arrives in Phase 5

    _, train_c, val_c, test_c, _ = get_dataset(c)
    _, train_b, _, _, _ = get_dataset(b)
    check('main config loads cubes', hasattr(train_c[0], 'ters'))
    check('base config skips cubes (load_cubes=False)', not hasattr(train_b[0], 'ters'))
    check('base config still serves rdmol for the MOSES sets', hasattr(train_b[0], 'rdmol'))

    try:
        CollateTers()([train_b[0], train_b[1]])
        check('graph-only data is refused by the training collate', False)
    except AttributeError:
        check('graph-only data is refused by the training collate', True)

    loader, _, _ = get_dataloader(train_c, val_c, test_c, c)
    batch = next(iter(loader))
    check('config -> dataloader -> batch', batch['context'].shape[1] == c.model.ters_in_channels)

    from models.dmt import DMT
    model = DMT(c)
    check('config builds DMT with a TersEncoder matching the data',
          type(model.cond_encoder).__name__ == 'TersEncoder'
          and model.cond_encoder.in_channels == train_c.in_channels)
    with torch.no_grad():
        v = model.cond_lin(model.cond_encoder(batch['context']))
    check('real batch -> finite conditioning vector',
          bool(torch.isfinite(v).all()) and v.shape[1] == model.cond_lin.out_features)


REAL_H5 = '/scratch/project_2001912/planar_hdf5_0.5/test.h5'


def phase_real(tmp):
    """Checks that need the real planar TERS file; skipped when it is not mounted.

    These are the ones that caught what the synthetic corpus could not: the HDF5 stores the
    per-molecule geometry as *groups* keyed '0'..'N-1', not as dense arrays.
    """
    import h5py

    print('\n=== Real data (/scratch planar file) ===')
    if not osp.exists(REAL_H5):
        print(f'  SKIP -- {REAL_H5} not mounted')
        return

    from datasets.preprocess_ters import iter_molecules, process_split
    from datasets.ters_dataset import TersGraphDataset

    with h5py.File(REAL_H5, 'r') as f:
        # The bug this guards: h5py returns group keys lexicographically, so
        # enumerate(keys) pairs graph i with the wrong cube for every i >= 2.
        keys = list(f['atomic_numbers'].keys())
        check('h5py group keys really are lexicographic (the trap)',
              keys[:4] == ['0', '1', '10', '100'], f'{keys[:4]}')
        mols = list(iter_molecules(f, 'atomic_numbers', 'atom_positions'))
        check('iter_molecules reads the group layout', len(mols) == len(f['channels_100']),
              f'{len(mols)} molecules')
        check('iter_molecules yields dense cube order',
              [i for i, _, _ in mols[:4]] == [0, 1, 2, 3])
        # index i must be group key str(i), not keys[i]
        z0 = np.asarray(f['atomic_numbers'][str(2)]).ravel()
        check('molecule 2 is group key "2", not the 2nd lexicographic key',
              np.array_equal(mols[2][1], z0[z0 > 0]))

        zs = [m[1] for m in mols]
        check('only H/C/N/O present -> data.atom_types = 4',
              sorted({int(v) for z in zs for v in z}) == [1, 6, 7, 8])
        check('largest molecule fits data.max_node',
              max(len(z) for z in zs) == 36, f'max={max(len(z) for z in zs)}')

    out = osp.join(tmp, 'graphs_real.pt')
    summary = process_split(REAL_H5, out, want_stats=True)
    check('every real molecule survives Phase 1',
          summary['n_kept'] == summary['n_total'], f'{summary["n_kept"]}/{summary["n_total"]}')
    check('no covalent fallback needed on real geometry', summary['n_fallback'] == 0)

    geo = summary['geometry']
    pn = geo['suggested_pos_norm']
    cut = geo['suggested_spatial_cut_off']
    check('suggested pos_norm reflects the ~18A field of view', pn >= 5, f'{pn}')
    # The regression that matters: spatial_cut_off is squared *normalised* distance.
    check('spatial_cut_off suggestion divides by pos_norm',
          cut < 0.5, f'{cut} (a raw-angstrom answer would be ~2.5)')
    check('suggested cutoff keeps every real bond',
          np.sqrt(cut) * pn >= 1.6, f'{np.sqrt(cut) * pn:.2f} A vs longest bond 1.601 A')

    ds = TersGraphDataset(h5_path=REAL_H5, graphs_path=out, channels_key='channels_100')
    check('real dataset assembles', len(ds) == summary['n_kept'] and ds.in_channels == 100)
    d = ds[0]
    check('real cube arrives as (C,H,W)', tuple(d.ters.shape) == (100, 256, 256))
    check('normalised real cube is finite', bool(torch.isfinite(d.ters).all()))
    dead = (d.ters.reshape(100, -1).abs().max(1).values == 0)
    check('real cubes DO contain dead channels (zero-guard is load-bearing)',
          bool(dead.any()), f'{int(dead.sum())}/100 dead')
    check('dead channels normalise to exactly 0, not NaN',
          bool((d.ters[dead] == 0).all()))

    # Alignment: the cube at h5_index must belong to the molecule in the sidecar.
    with h5py.File(REAL_H5, 'r') as f:
        FR = f['frequencies']
        hits = 0
        for k in range(24):
            s = ds[k]
            i = int(s.h5_index)
            live = (s.ters.reshape(100, -1).abs().max(1).values > 0).numpy()
            modes = np.zeros(100, bool)
            for v in np.asarray(FR[str(i)]).ravel():
                b = int(v / 4000 * 100)
                if 0 <= b < 100:
                    modes[b] = True
            hits += int(modes.any() and live[modes].mean() > 0.9)
        check('cube at h5_index carries that molecule\'s vibrational modes',
              hits >= 22, f'{hits}/24')


def main():
    tmp = tempfile.mkdtemp(prefix='ters-tests-')
    try:
        from tests.make_synthetic import build
        h5_dir = osp.join(tmp, 'h5')
        build(h5_dir)
        graph_dir = phase1(tmp, h5_dir)
        phase2(h5_dir, graph_dir)
        phase3()
        phase4(h5_dir, graph_dir)
        phase_real(tmp)
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
