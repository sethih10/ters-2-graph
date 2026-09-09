"""Conditional generation from TERS cubes -- the branch sampling.py does not have (D-8).

STANDALONE. This does not modify sampling.py. It reuses that file's AncestralSampler,
post_process and mol_process unchanged; the only thing sampling.py lacks is the code that
picks `context`, because every branch there is keyed on config.data.spectra_version, which
a TERS config does not define. That selection is the ~15 lines below marked NEW.

For each held-out molecule we take ITS cube as the condition and its true atom count, then
draw n_samples independent reverse-diffusion trajectories from different initial noise.
A perfect model would return the same molecule every time.

Run on a GPU node:
    python -m tools.sample_ters --ckpt <path> --split test --n-mols 5 --n-samples 5
"""
import argparse, importlib.util, json, os, pickle, random

# --- ADDED (stm): output paths resolve to /scratch, never inside the repo ($HOME is a
# 15 GB quota and one run is 3.7 GB). Both import forms, because these are run BOTH as
# `python3 tools/x.py` (sibling import) and as `python3 -m tools.x` (package import).
try:
    from paths import generated
except ImportError:
    from tools.paths import generated
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

from datasets import get_dataset
from models.ema import ExponentialMovingAverage
from models import create_model
from models.utils import (sample_combined_position_feature_noise,
                          sample_symmetric_edge_feature_noise, assert_mean_zero_with_mask)
from losses import get_self_cond_fn
from utils import get_data_inverse_scaler
from diffusion import NoiseScheduleVP
from evaluation import get_2D_edm_metric
from sampling import AncestralSampler, post_process, mol_process_2D


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.get_config()


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


# --- ADDED (stm): read the cube from whichever attribute the modality uses -------------
# TERS stores it as `.ters`, STM as `.stm` -- the `cube_attr` class constant on the dataset
# (datasets/ters_dataset.py:66). Probed on the object rather than passed in, because this
# script's only other modality dependency is the config path, and threading a second flag
# through would be one more thing to get wrong at the call site.
def cube_of(data):
    for attr in ('ters', 'stm'):
        if hasattr(data, attr):
            return getattr(data, attr)
    raise AttributeError(
        f'sample: no conditioning cube on this sample (looked for .ters and .stm). '
        f'Attributes present: {sorted(k for k in data.keys)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/diffspectra_ters.py')
    ap.add_argument('--config-qm9', default='configs/base_ters.py')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--n-mols', type=int, default=5)
    ap.add_argument('--n-samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shuffle-cubes', action='store_true',
                    help='NULL CONTROL. Pair each target with ANOTHER molecule\'s cube (a '
                         'derangement, so no target keeps its own). Target, atom count and '
                         'scoring are unchanged, so the cube is the only thing that differs. '
                         'If matched and shuffled score the same, generation is not using the '
                         'cube -- the Tanimoto is just the prior. Outputs get a _shuf suffix.')
    ap.add_argument('--batch-mols', type=int, default=0,
                    help='TARGETS per generation batch (0 = all in one, the old behaviour). '
                         'Set this to evaluate more than ~20 targets: the protocol wants 10 '
                         'samples each, so 500 targets is 5000 molecules in one forward pass '
                         'and will OOM. Chunks split on targets, never within a target.')
    ap.add_argument('--no-geometry', action='store_true',
                    help='skip the geometry_*.pkl payload (only needed for the 2D/3D figures; '
                         'it holds a full cube per target, so it dominates disk on big runs)')
    ap.add_argument('--out', default=generated('ters-runs/ters_20260828_925597'))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    config = load_config(args.config); config.device = torch.device('cuda')
    cfg_q = load_config(args.config_qm9)
    os.makedirs(args.out, exist_ok=True)
    tag = '_shuf' if args.shuffle_cubes else ''

    _, _, val_ds, test_ds, dataset_info = get_dataset(config)
    get_dataset(cfg_q)
    ds = test_ds if args.split == 'test' else val_ds
    print(f'{args.split} split: {len(ds)} molecules')

    # Read the encoder's pooling grid OFF THE CHECKPOINT rather than trusting the config.
    # config.model.ters_pool_size is an experiment knob (docs/experimental.md E-1), so the
    # default moves; a checkpoint trained under the old default then fails strict load with a
    # bare shape mismatch on cond_encoder.head.2.weight. The checkpoint is the authority on
    # its own architecture -- infer, announce, and let every arm be sampled with one command.
    ck = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    # --- ADDED (stm): same inference, but against whichever encoder this config builds ----
    # StmEncoder's head has the identical (AdaptiveAvgPool2d, Flatten, Linear) shape, so the
    # `head.2.weight` probe works unchanged -- only the knob and filter-width names differ.
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

    # ---- NEW: pick held-out molecules and use THEIR cubes as the condition ----------------
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:args.n_mols]
    # NULL CONTROL: derange the cube->molecule pairing. Everything else (target molecule,
    # atom count, ranking, scoring) is untouched, so any drop in Tanimoto is attributable to
    # the cube alone. Without this baseline a Tanimoto is uninterpretable: the model is handed
    # the true atom count and the chemical space is narrow, so the prior alone scores > 0.
    cube_src = list(idx)
    if args.shuffle_cubes:
        rng = random.Random(args.seed + 1)
        for _ in range(1000):
            rng.shuffle(cube_src)
            if all(int(a) != int(b) for a, b in zip(cube_src, idx)):
                break
        else:
            raise RuntimeError('could not derange the cube pairing')
        print(f'SHUFFLED CUBES (null control): cube order {[int(x) for x in cube_src]} '
              f'vs target order {[int(x) for x in idx]}')

    cubes, n_nodes, truth = [], [], []
    for k, i in enumerate(idx):
        mol = ds[int(i)]
        cube_mol = ds[int(cube_src[k])] if args.shuffle_cubes else mol
        # The TERS sidecar stores num_atom as a plain int; QM9S stores a tensor. Take both.
        na = mol.num_atom
        n_at = int(na.item()) if torch.is_tensor(na) else int(na)
        for _ in range(args.n_samples):            # same cube, different initial noise
            cubes.append(cube_of(cube_mol))
            n_nodes.append(n_at)
        truth.append(mol.rdmol)
    # --- ADDED (stm): only the geometry payload needs these, and each carries a full
    # cube (917 KB). At 23123 targets that is ~21 GB of host RAM held for nothing.
    truth_graphs = [] if args.no_geometry else [ds[int(i)] for i in idx]
    B = len(n_nodes)
    # --- ADDED (stm): generate in chunks of TARGETS ---------------------------------------
    # This was a single batch of n_mols*n_samples, which silently capped the whole evaluation
    # at ~20 targets -- 0.09% of the 23123-molecule test split. The paper protocol wants 10
    # samples per target, so 500 targets is 5000 molecules in one forward pass and OOMs.
    # Chunking splits on TARGETS and never within a target, so all 10 samples of a molecule
    # stay in one batch and the per-target aggregation downstream is untouched.
    # batch_mols=0 keeps the original single-batch behaviour exactly.
    chunk_mols = args.batch_mols or args.n_mols
    n_batches = -(-args.n_mols // chunk_mols)
    print(f'generating {B} molecules = {args.n_mols} spectra x {args.n_samples} samples, '
          f'{config.sampling.steps} ancestral steps, '
          f'{n_batches} batch(es) of <= {chunk_mols * args.n_samples}')
    # --------------------------------------------------------------------------------------

    node_nf = config.data.atom_types + int(config.model.include_fc_charge)
    processed, pos_list, atom_list, edge_list = [], [], [], []
    for c0 in range(0, args.n_mols, chunk_mols):
        c1 = min(c0 + chunk_mols, args.n_mols)
        sl = slice(c0 * args.n_samples, c1 * args.n_samples)
        cn = n_nodes[sl]
        cb = len(cn); max_n = max(cn)
        context = torch.stack(cubes[sl], dim=0).to(config.device)   # build_dataset.py:570

        node_mask = torch.zeros(cb, max_n)
        for i in range(cb):
            node_mask[i, :cn[i]] = 1
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        edge_mask *= ~torch.eye(max_n, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask.view(cb * max_n * max_n, 1).to(config.device)
        node_mask = node_mask.unsqueeze(2).to(config.device)

        with torch.no_grad():
            z = sample_combined_position_feature_noise(cb, max_n, node_nf, node_mask)
            assert_mean_zero_with_mask(z[:, :, :3], node_mask)
            edge_z = sample_symmetric_edge_feature_noise(cb, max_n, config.model.edge_ch,
                                                         edge_mask)
            x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
            pos, one_hot, fc, edge_types = post_process(x_node, config.data.atom_types,
                                                        config.model.include_fc_charge, node_mask,
                                                        inverse_scaler, x_edge, edge_mask,
                                                        config.data.compress_edge)
        # mol_process_2D trims each molecule to its own n_nodes (sampling.py:41-47), so
        # chunks with different max_n concatenate with no padding reconciliation.
        processed += mol_process_2D(one_hot, fc, cn, edge_types)
        for j, n in enumerate(cn):
            pos_list.append(pos[j, :n].cpu().numpy())
            atom_list.append(one_hot[j, :n].argmax(-1).cpu().numpy())
            edge_list.append(edge_types[j, :n, :n].cpu().numpy())
        if n_batches > 1:
            print(f'  batch {c0 // chunk_mols + 1}/{n_batches}: targets {c0}-{c1 - 1} '
                  f'({cb} molecules)', flush=True)
        del context, node_mask, edge_mask, z, edge_z, x_node, x_edge, pos, one_hot, edge_types
        torch.cuda.empty_cache()
    # The 3D metric (get_edm_metric -> check_stability) CANNOT be used here: it infers bonds
    # from interatomic distances via per-dataset tables selected by name at
    # evaluation/stability.py:50-54, and raises 'Fail to get dataset bond info.' for anything
    # that is not QM9 or Geom. ters_hcno has no such table -- a real Phase 5 gap.
    # The 2D path is the right one regardless: this model has pred_edge=True, so bond orders
    # are PREDICTED by the network, and check_2D_stability reads them directly instead of
    # re-deriving them from geometry. It needs only atom_decoder and atom_fc_num, both of
    # which ters_hcno defines.
    stab, rdkit_dict, rd_mols = get_2D_edm_metric(dataset_info)(processed)
    print(f'\nstability: {stab}')
    print(f'rdkit    : {rdkit_dict}')

    # ---- score every sample against the molecule its spectrum came from -------------------
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

    # ---- geometry: the model's ACTUAL predicted coordinates, for plotting -----------------
    # post_process already ran pos through inverse_scaler, so these are angstroms, not the
    # pos/7 the network works in (config.model.normalize_factors).
    def dense_edges(g, n):
        e = torch.zeros(n, n, dtype=torch.long)
        ei, et = g.edge_index, g.edge_type
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a < n and b < n:
                e[a, b] = e[b, a] = int(et[k])
        return e

    # --- ADDED (stm): the geometry payload is optional ------------------------------------
    # It holds a full (H, W) cube projection per target, so it is the largest thing the run
    # writes (~65 MB at 500 targets) -- yet it is consumed only by the 2D/3D figures, which
    # are made from a handful of molecules. Skip it on large scoring runs.
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
            # --- ADDED (stm): record the depth. cube_sum collapses it, so without this the
            # plotter cannot label the panel and previously hardcoded '100 ch' -- right for
            # TERS, wrong for a 14-slice STM stack.
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


if __name__ == '__main__':
    main()
