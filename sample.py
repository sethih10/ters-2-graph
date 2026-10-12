"""Generate molecules conditioned on TERS cubes or STM z-stacks.

Each sample takes its atom count from the target molecule. Writes `mols_*.pkl` for
target-similarity scoring, `processed_*.pkl` for distributional scoring and, unless
`--no-geometry` is set, `geometry_*.pkl` for the geometry figures.

Usage
-----
    python sample.py --modality stm --ckpt CKPT --split test
"""
import argparse, csv, json, os, pickle, random, time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

from datasets import get_dataset
from models.ema import ExponentialMovingAverage
from models import create_model
from utils import get_data_inverse_scaler, get_self_cond_fn
from diffusion import NoiseScheduleVP
from evaluation import get_2D_edm_metric
from sampling import AncestralSampler, mol_process, sample_batch
import sys

from absl import flags
from ml_collections import config_flags


# Per-modality defaults: (config, aux config).
MODALITY_DEFAULTS = {
    'ters': ('configs/diffspectra_ters.py', 'configs/base_ters.py'),
    'stm':  ('configs/diffspectra_stm.py', 'configs/base_stm.py'),
}



def morgan(mol):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def canon(mol):
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def cube_of(data):
    """Return the conditioning cube of a TERS (``.ters``) or STM (``.stm``) sample."""
    for attr in ('ters', 'stm'):
        if hasattr(data, attr):
            return getattr(data, attr)
    try:
        present = sorted(str(k) for k in data.keys)
    except (AttributeError, TypeError):
        present = sorted(k for k in vars(data) if not k.startswith('_')) or '<unknown>'
    raise AttributeError(
        f'sample: no conditioning cube on this sample (looked for .ters and .stm). '
        f'Attributes present: {present}')


def write_portable(out_dir, split, tag, results, rd_mols, n_samples):
    """Write generated_<split>.sdf + .csv. Returns (n_sdf_records, n_csv_rows)."""
    sdf_path = os.path.join(out_dir, f'generated_{split}{tag}.sdf')
    csv_path = os.path.join(out_dir, f'generated_{split}{tag}.csv')
    n_sdf = n_rows = 0
    writer = Chem.SDWriter(sdf_path)
    try:
        with open(csv_path, 'w', newline='') as fcsv:
            wcsv = csv.writer(fcsv)
            wcsv.writerow(['ds_index', 'sample', 'smiles', 'tanimoto', 'valid', 'exact',
                           'truth_smiles'])
            for e in results:
                for g in e['samples']:
                    wcsv.writerow([e['ds_index'], g['sample'], g['smiles'],
                                   f"{g['tanimoto']:.4f}", int(g['valid']), int(g['exact']),
                                   e['truth_smiles']])
                    n_rows += 1
                    m = rd_mols[e['mol'] * n_samples + g['sample']]
                    if m is None:
                        continue
                    try:
                        m = Chem.Mol(m)          # copy: do not mutate rd_mols
                        m.SetProp('_Name', f"{split}_{e['ds_index']}_s{g['sample']}")
                        m.SetProp('ds_index', str(e['ds_index']))
                        m.SetProp('sample', str(g['sample']))
                        m.SetProp('tanimoto', f"{g['tanimoto']:.4f}")
                        m.SetProp('truth_smiles', e['truth_smiles'] or '')
                        writer.write(m)
                        n_sdf += 1
                    except Exception:
                        continue
    finally:
        writer.close()
    print(f'wrote {sdf_path} ({n_sdf} records) and {csv_path} ({n_rows} rows)')
    return n_sdf, n_rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--modality', default='ters', choices=sorted(MODALITY_DEFAULTS),
                    help='selects default --config/--config_original_qm9 only (default: ters)')
    ap.add_argument('--config', help='Python config file; override fields with --config.FIELD=VALUE', default=None)
    ap.add_argument('--config_original_qm9', '--config-qm9', dest='config_qm9', default=None)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--n-mols', type=int, default=5)
    ap.add_argument('--n-samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shuffle-cubes', action='store_true',
                    help='null control: pair each target with another molecule\'s cube (a '
                         'derangement, so no target keeps its own). Target, atom count and '
                         'scoring are unchanged, so only the cube differs. If matched and '
                         'shuffled runs score the same, generation does not use the cube and '
                         'the Tanimoto reflects the prior. Outputs get a _shuf suffix.')
    ap.add_argument('--batch-mols', type=int, default=0,
                    help='targets per generation batch (0 = all targets in one batch). '
                         'Set this for more than about 20 targets: at 10 samples each, 500 '
                         'targets is 5000 molecules in one forward pass, which runs out of GPU '
                         'memory. Chunks split between targets, never within a target.')
    ap.add_argument('--n-shards', type=int, default=1,
                    help='split the target list across this many independent processes, one '
                         'per GPU (sampling with DataParallel measured 4.3x slower). '
                         'Each shard needs its own --out and its own '
                         'CUDA_VISIBLE_DEVICES; merge with merge_shards.py before scoring.')
    ap.add_argument('--shard', type=int, default=0, help='which shard, in [0, --n-shards)')
    ap.add_argument('--save-sdf', action='store_true',
                    help='also write generated_<split>.sdf (3D coordinates, one record per '
                         'sample) and a generated_<split>.csv of ds_index/sample/smiles/'
                         'tanimoto. mols_*.pkl is Python-only; this is the portable form.')
    ap.add_argument('--no-geometry', action='store_true',
                    help='skip the geometry_*.pkl payload (only needed for the 2D/3D figures; '
                         'it holds a full cube per target, so it dominates disk on big runs)')
    ap.add_argument('--out', default=None)
    args, config_overrides = ap.parse_known_args()

    _cfg, _cfg_q = MODALITY_DEFAULTS[args.modality]
    args.config = args.config or _cfg
    args.config_qm9 = args.config_qm9 or _cfg_q
    # Default: <workdir>/generated, next to <workdir>/checkpoints/<ckpt>.
    args.out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))), 'generated')

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    config_argv = [sys.argv[0], f'--config={args.config}',
                   f'--config_original_qm9={args.config_qm9}', *config_overrides]
    config_values = flags.FlagValues()
    config_flags.DEFINE_config_file('config', flag_values=config_values,
                                    sys_argv=config_argv)
    config_flags.DEFINE_config_file('config_original_qm9', flag_values=config_values,
                                    sys_argv=config_argv)
    try:
        config_values(config_argv)
    except flags.Error as exc:
        ap.error(str(exc))
    config = config_values.config; config.device = torch.device('cuda')
    cfg_q = config_values.config_original_qm9
    os.makedirs(args.out, exist_ok=True)
    tag = '_shuf' if args.shuffle_cubes else ''

    _, _, val_ds, test_ds, dataset_info = get_dataset(config)
    get_dataset(cfg_q)
    ds = test_ds if args.split == 'test' else val_ds
    print(f'{args.split} split: {len(ds)} molecules')

    ck = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    # Infer the pool size from the checkpoint; head.2 is Linear(filters[-1] * pool_size**2, nf).
    _pfx = 'stm' if config.data.name == 'STM' else 'ters'
    _defaults = {'stm': (64, 128, 256), 'ters': (64, 128, 256, 512)}[_pfx]
    _hw = next((v for k, v in ck['model'].items() if k.endswith('cond_encoder.head.2.weight')), None)
    if _hw is not None:
        _filters = tuple(getattr(config.model, f'{_pfx}_filters', _defaults))
        _grid = _hw.shape[1] // _filters[-1]
        _inferred = int(round(_grid ** 0.5))
        if _inferred * _inferred * _filters[-1] != _hw.shape[1]:
            raise ValueError(f'cannot infer {_pfx}_pool_size: head.2.weight has in_features '
                             f'{_hw.shape[1]}, not a square multiple of {_filters[-1]}')
        if _inferred != getattr(config.model, f'{_pfx}_pool_size', 1):
            print(f'{_pfx}_pool_size: config says '
                  f'{getattr(config.model, f"{_pfx}_pool_size", 1)}, '
                  f'checkpoint says {_inferred} -- using the checkpoint')
        setattr(config.model, f'{_pfx}_pool_size', _inferred)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    model.load_state_dict(ck['model'], strict=True)
    ema.load_state_dict(ck['ema']); ema.copy_to(model.parameters()); model.eval()
    print(f"checkpoint step {int(ck['step'])} (EMA weights)")

    noise_scheduler = NoiseScheduleVP(config.sde.schedule,
                                      continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)
    inverse_scaler = get_data_inverse_scaler(config)
    time_steps = torch.linspace(noise_scheduler.T, 1e-3, config.sampling.steps, device=config.device)
    sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data,
                               config.pred_edge, config.model.self_cond, get_self_cond_fn(config))

    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:args.n_mols]
    # Shards are strided slices of one seeded permutation; all must share --seed and --n-mols.
    if args.n_shards > 1:
        if not 0 <= args.shard < args.n_shards:
            raise SystemExit(f'--shard must be in [0, {args.n_shards}), got {args.shard}')
        idx = idx[args.shard::args.n_shards]
        print(f'shard {args.shard}/{args.n_shards}: {len(idx)} of {args.n_mols} targets')
        args.n_mols = len(idx)          # every count below is per-shard from here on
    # Null control: derange the cubes within this shard; targets and atom counts are unchanged.
    cube_src = list(idx)
    if args.shuffle_cubes:
        rng = random.Random(args.seed + 1)
        for _ in range(1000):
            rng.shuffle(cube_src)
            if all(int(a) != int(b) for a, b in zip(cube_src, idx)):
                break
        else:
            raise RuntimeError('could not derange the cube pairing')
        print(f'shuffled cubes (null control): cube order {[int(x) for x in cube_src]} '
              f'vs target order {[int(x) for x in idx]}')

    cubes, n_nodes, truth = [], [], []
    for k, i in enumerate(idx):
        mol = ds[int(i)]
        cube_mol = ds[int(cube_src[k])] if args.shuffle_cubes else mol
        # The TERS sidecar stores num_atom as a plain int; QM9S stores a tensor.
        na = mol.num_atom
        n_at = int(na.item()) if torch.is_tensor(na) else int(na)
        for _ in range(args.n_samples):            # same cube, different initial noise
            cubes.append(cube_of(cube_mol))
            n_nodes.append(n_at)
        truth.append(mol.rdmol)
    truth_graphs = [] if args.no_geometry else [ds[int(i)] for i in idx]
    B = len(n_nodes)
    chunk_mols = args.batch_mols or args.n_mols
    n_batches = -(-args.n_mols // chunk_mols)
    print(f'generating {B} molecules = {args.n_mols} spectra x {args.n_samples} samples, '
          f'{config.sampling.steps} ancestral steps, '
          f'{n_batches} batch(es) of <= {chunk_mols * args.n_samples}')

    # DataParallel needs the batch to divide by the GPU count; chunks are padded, then trimmed.
    n_gpu = torch.cuda.device_count() if config.device.type == 'cuda' else 1
    if n_gpu > 1:
        print(f'{n_gpu} GPUs visible -- DataParallel; chunks padded to a multiple of {n_gpu} '
              f'so every replica gets an equal split')

    processed, pos_list, atom_list, edge_list = [], [], [], []
    t_gen = time.time()
    for bi, c0 in enumerate(range(0, args.n_mols, chunk_mols), 1):
        c1 = min(c0 + chunk_mols, args.n_mols)
        sl = slice(c0 * args.n_samples, c1 * args.n_samples)
        cn = n_nodes[sl]
        chunk_cubes = cubes[sl]
        n_real = len(cn)
        # Padded entries are dropped after sampling ([:n_real]).
        pad = (-n_real) % n_gpu
        if pad:
            cn = cn + [cn[-1]] * pad
            chunk_cubes = chunk_cubes + [chunk_cubes[-1]] * pad
        cb = len(cn); max_n = max(cn)
        assert cb % n_gpu == 0, (
            f'batch of {cb} does not divide across {n_gpu} GPUs. DMT.forward reshapes a '
            f'flattened edge_mask by the xh batch size, so an uneven DataParallel split makes '
            f'the two disagree. The padding above should prevent this.')
        # (B, C, H, W), stacked as in CollateTers.
        context = torch.stack(chunk_cubes, dim=0).to(config.device)
        pos, one_hot, fc, edge_types = sample_batch(model, sampler, context, cn, config,
                                                    inverse_scaler)
        # Drop the padding so the flat index stays mi * n_samples + si.
        processed += mol_process(one_hot, pos, fc, cn, edge_types)[:n_real]
        for j, n in enumerate(cn[:n_real]):
            pos_list.append(pos[j, :n].cpu().numpy())
            atom_list.append(one_hot[j, :n].argmax(-1).cpu().numpy())
            edge_list.append(edge_types[j, :n, :n].cpu().numpy())
        if n_batches > 1:
            el = time.time() - t_gen
            mem = (torch.cuda.max_memory_allocated() / 1024 ** 3
                   if config.device.type == 'cuda' else 0.0)
            print(f'  batch {bi}/{n_batches}: targets {c0}-{c1 - 1} ({n_real} molecules'
                  f'{f", +{pad} pad" if pad else ""}, max_n {max_n})  '
                  f'{el:.0f}s elapsed  eta {el / bi * (n_batches - bi):.0f}s  '
                  f'{len(processed) / el:.1f} mol/s  peak {mem:.1f} GiB', flush=True)
        del context, pos, one_hot, edge_types
        torch.cuda.empty_cache()
    # Metric-2D only: Metric-3D is not calibrated for STM (see eval_cond).
    stab, rdkit_dict, rd_mols = get_2D_edm_metric(dataset_info)(processed)
    print(f'\nstability: {stab}')
    print(f'rdkit    : {rdkit_dict}')

    with open(os.path.join(args.out, f'processed_{args.split}{tag}.pkl'), 'wb') as f:
        pickle.dump({'processed': processed,
                     'n_mols': args.n_mols, 'n_samples': args.n_samples,
                     'split': args.split, 'shuffled_cubes': bool(args.shuffle_cubes),
                     'modality': args.modality, 'config': args.config,
                     'config_qm9': args.config_qm9,
                     'info_name': config.data.info_name,
                     'ckpt': args.ckpt, 'step': int(ck['step'])}, f)

    results = []
    for mi in range(args.n_mols):
        gt = truth[mi]
        gt_smiles, gt_fp = canon(gt), morgan(gt)
        gen = []
        for si in range(args.n_samples):
            m = rd_mols[mi * args.n_samples + si]
            smi, tan, valid = None, 0.0, False
            if m is not None:
                try:
                    Chem.SanitizeMol(m); valid = True
                except Exception:
                    valid = False
                smi = canon(m); fp = morgan(m)
                if fp is not None and gt_fp is not None:
                    tan = float(DataStructs.TanimotoSimilarity(gt_fp, fp))
            gen.append({'sample': si, 'smiles': smi, 'valid': valid,
                        'tanimoto': tan, 'exact': bool(smi is not None and smi == gt_smiles)})
        best = max(g['tanimoto'] for g in gen)
        results.append({'mol': mi, 'ds_index': int(idx[mi]), 'n_atoms': n_nodes[mi * args.n_samples],
                        'truth_smiles': gt_smiles, 'best_tanimoto': best,
                        'mean_tanimoto': float(np.mean([g['tanimoto'] for g in gen])),
                        'n_valid': sum(g['valid'] for g in gen),
                        'n_exact': sum(g['exact'] for g in gen), 'samples': gen})

    results.sort(key=lambda r: -r['best_tanimoto'])
    print('\n' + '=' * 88)
    print(f'{args.split.upper()} SET -- {args.n_mols} spectra x {args.n_samples} samples, ranked best to worst')
    print('=' * 88)
    print(f"{'rank':>5}{'idx':>7}{'atoms':>7}{'valid':>7}{'exact':>7}{'best Tan':>10}{'mean Tan':>10}  truth SMILES")
    for r, e in enumerate(results):
        print(f"{r:>5}{e['ds_index']:>7}{e['n_atoms']:>7}{e['n_valid']:>4}/{args.n_samples}"
              f"{e['n_exact']:>4}/{args.n_samples}{e['best_tanimoto']:>10.3f}{e['mean_tanimoto']:>10.3f}"
              f"  {e['truth_smiles']}")
    print('=' * 88)
    for e in results:
        print(f"\n  spectrum idx {e['ds_index']}  truth: {e['truth_smiles']}")
        for g in e['samples']:
            flag = 'EXACT' if g['exact'] else ('ok' if g['valid'] else 'INVALID')
            print(f"    sample {g['sample']}  Tan {g['tanimoto']:.3f}  [{flag:>7}]  {g['smiles']}")

    # Geometry payload for the figures; positions are in angstroms.
    def dense_edges(g, n):
        e = torch.zeros(n, n, dtype=torch.long)
        ei, et = g.edge_index, g.edge_type
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a < n and b < n:
                e[a, b] = e[b, a] = int(et[k])
        return e

    if args.no_geometry:
        print('skipping geometry payload (--no-geometry)')
    else:
        geo = {'n_mols': args.n_mols, 'n_samples': args.n_samples,
               'order': [e['ds_index'] for e in results], 'mol_order': [e['mol'] for e in results],
               'truth': [], 'gen': [], 'cube_sum': [], 'cube_max': []}
        for mi in range(args.n_mols):
            g = truth_graphs[mi]
            n = n_nodes[mi * args.n_samples]
            geo['truth'].append({'pos': g.pos[:n].cpu().numpy(),
                                 'atom_type': g.atom_type[:n].cpu().numpy(),
                                 'edges': dense_edges(g, n).numpy(),
                                 'smiles': results[[r['mol'] for r in results].index(mi)]['truth_smiles']})
            c = cube_of(g).cpu().numpy()                   # (C, H, W) normalised cube
            geo['n_channels'] = int(c.shape[0])
            geo['cube_sum'].append(c.sum(axis=0))
            geo['cube_max'].append(c.max(axis=0))
            for si in range(args.n_samples):
                b = mi * args.n_samples + si
                geo['gen'].append({'mol': mi, 'sample': si,
                                   'pos': pos_list[b],
                                   'atom_type': atom_list[b],
                                   'edges': edge_list[b]})
        with open(os.path.join(args.out, f'geometry_{args.split}{tag}.pkl'), 'wb') as f:
            pickle.dump(geo, f)
        print(f'wrote geometry_{args.split}.pkl  '
              f'(cube {geo["cube_sum"][0].shape}, {len(geo["gen"])} generated geometries)')

    out = {'split': args.split, 'shuffled_cubes': bool(args.shuffle_cubes), 'ckpt': args.ckpt, 'step': int(ck['step']),
           'sampling_steps': config.sampling.steps,
           'stability': stab, 'rdkit': rdkit_dict, 'results': results}
    with open(os.path.join(args.out, f'samples_{args.split}{tag}.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(args.out, f'mols_{args.split}{tag}.pkl'), 'wb') as f:
        pickle.dump({'truth': truth, 'generated': rd_mols,
                     'n_mols': args.n_mols, 'n_samples': args.n_samples,
                     'order': [e['ds_index'] for e in results], 'results': results}, f)
    print(f"\nwrote {args.out}/samples_{args.split}.json and mols_{args.split}.pkl")

    if args.save_sdf:
        write_portable(args.out, args.split, tag, results, rd_mols, args.n_samples)


if __name__ == '__main__':
    main()
