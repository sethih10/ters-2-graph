"""Conditional generation from an image-cube condition -- TERS cubes or STM z-stacks.

Was ``tools/sample_ters.py``; renamed because it has handled both modalities since the STM
retrofit and the old name was actively misleading (``metric_evaluation/run_metrics.py``
drives it with STM defaults). ``tools/sample_ters.py`` remains as a deprecation shim.

STANDALONE. This does not modify sampling.py. It reuses that file's AncestralSampler,
post_process and mol_process unchanged; the only thing sampling.py lacks is the code that
picks `context`, because every branch there is keyed on config.data.spectra_version, which
a cube config does not define (defect D-8, still open -- see docs/05-progress-log.md).
That selection is the ~15 lines below marked NEW.

For each held-out molecule we take ITS cube as the condition and its true atom count, then
draw n_samples independent reverse-diffusion trajectories from different initial noise.
A perfect model would return the same molecule every time.

Outputs, per run, under --out:
    samples_<split>[_shuf].json     ranked per-target summary (human-readable)
    mols_<split>[_shuf].pkl         truth + generated RDKit mols  -> tools/paper_metrics.py
    processed_<split>[_shuf].pkl    (pos, atom_type, edge_type, fc) tuples
                                    -> tools/eval_cond.py  (distributional metrics)
    geometry_<split>[_shuf].pkl     coordinates + cube projections (unless --no-geometry)

Run on a GPU node:
    python -m tools.sample_cond --modality stm --ckpt <path> \\
        --split test --n-mols 5 --n-samples 5
"""
import argparse, csv, importlib.util, json, os, pickle, random, time

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
from sampling import AncestralSampler, post_process, mol_process


# --- ADDED (eval): the two config paths and the output dir are the ONLY things that differ
# between modalities in this file -- everything else branches on config.data.name at runtime.
# Kept as a table rather than forked into a second script for that reason.
MODALITY_DEFAULTS = {
    'ters': ('configs/diffspectra_ters.py', 'configs/base_ters.py',
             'ters-runs/ters_20260828_925597'),
    'stm':  ('configs/diffspectra_stm.py', 'configs/base_stm.py',
             'stm-runs/samples'),
}


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
    # --- ADDED (eval): the `data.keys` probe is itself guarded. It is a PyG Data API, and on
    # anything else it raises its own AttributeError -- which REPLACES this message with a
    # useless one, at exactly the moment the message is the only diagnostic available.
    try:
        present = sorted(str(k) for k in data.keys)
    except (AttributeError, TypeError):
        present = sorted(k for k in vars(data) if not k.startswith('_')) or '<unknown>'
    raise AttributeError(
        f'sample: no conditioning cube on this sample (looked for .ters and .stm). '
        f'Attributes present: {present}')


# --- ADDED (scale): portable export --------------------------------------------------------
# mols_*.pkl is a Python pickle of RDKit Mol objects: fine for paper_metrics.py, useless to
# anything else and unreadable without this exact RDKit. An SDF is the interchange format, and
# it carries real 3D coordinates -- `processed` holds them because this script uses mol_process
# rather than mol_process_2D. Module-level rather than inline in main() so tests/test_stm.py can
# drive it from an existing pickle: it runs at the very END of a run that may have taken hours,
# which is the worst possible place to discover a typo.
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
            # Iterate `results` rather than raw index order: it is sorted by best_tanimoto, and
            # carrying ds_index explicitly is what keeps the SDF and the CSV in the same order.
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
                        m = Chem.Mol(m)          # copy: never mutate what the pickle holds
                        m.SetProp('_Name', f"{split}_{e['ds_index']}_s{g['sample']}")
                        m.SetProp('ds_index', str(e['ds_index']))
                        m.SetProp('sample', str(g['sample']))
                        m.SetProp('tanimoto', f"{g['tanimoto']:.4f}")
                        m.SetProp('truth_smiles', e['truth_smiles'] or '')
                        writer.write(m)
                        n_sdf += 1
                    except Exception:
                        # A molecule RDKit will not serialise must not kill the export of a run
                        # that took hours; it is already in the CSV and in mols_*.pkl.
                        continue
    finally:
        writer.close()
    print(f'wrote {sdf_path} ({n_sdf} records) and {csv_path} ({n_rows} rows)')
    return n_sdf, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    # --- ADDED (eval): --modality only picks defaults. An explicit --config / --config-qm9 /
    # --out still wins, so every existing invocation (including run_metrics.py, which always
    # passes all three) behaves exactly as before.
    ap.add_argument('--modality', default='ters', choices=sorted(MODALITY_DEFAULTS),
                    help='selects default --config/--config-qm9/--out only (default: ters, '
                         'preserving the behaviour of the old tools/sample_ters.py)')
    ap.add_argument('--config', default=None)
    ap.add_argument('--config-qm9', default=None)
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
    ap.add_argument('--n-shards', type=int, default=1,
                    help='split the target list across this many independent processes. THIS '
                         'is how to use several GPUs -- DataParallel is 4.3x slower here '
                         '(gputest job 1229202). Each shard needs its own --out and its own '
                         'CUDA_VISIBLE_DEVICES; merge with tools/merge_shards.py before scoring.')
    ap.add_argument('--shard', type=int, default=0, help='which shard, in [0, --n-shards)')
    ap.add_argument('--save-sdf', action='store_true',
                    help='also write generated_<split>.sdf (3D coordinates, one record per '
                         'sample) and a generated_<split>.csv of ds_index/sample/smiles/'
                         'tanimoto. mols_*.pkl is Python-only; this is the portable form.')
    ap.add_argument('--no-geometry', action='store_true',
                    help='skip the geometry_*.pkl payload (only needed for the 2D/3D figures; '
                         'it holds a full cube per target, so it dominates disk on big runs)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    _cfg, _cfg_q, _out = MODALITY_DEFAULTS[args.modality]
    args.config = args.config or _cfg
    args.config_qm9 = args.config_qm9 or _cfg_q
    args.out = args.out or generated(_out)

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
    # --- ADDED (scale): shard the TARGET list across independent single-GPU processes -------
    # This, not DataParallel, is how to use more than one GPU here. MEASURED (gputest job
    # 1229202): DataParallel on 4 GPUs is 4.3x SLOWER than 1 GPU (0.9 vs 3.9 mol/s), because it
    # re-replicates a 587 MB model to every device on every forward call and sampling makes 250
    # sequential forward calls per chunk. Sharding replicates nothing -- each process owns one
    # GPU and a disjoint slice of targets.
    #
    # The slice is taken AFTER the permutation, and the permutation is seeded, so every shard
    # sees the same ordering and the union over shards is exactly the unsharded run's target
    # set -- no target is dropped or duplicated. Strided (`[shard::n_shards]`) rather than
    # contiguous so the shards stay balanced in molecule size: targets are in random order, but
    # a contiguous block could still draw an unlucky run of 50-atom molecules and finish last.
    #
    # Each shard writes its own --out directory. Merge with tools/merge_shards.py before
    # scoring; nothing downstream understands a partial cell.
    if args.n_shards > 1:
        if not 0 <= args.shard < args.n_shards:
            raise SystemExit(f'--shard must be in [0, {args.n_shards}), got {args.shard}')
        idx = idx[args.shard::args.n_shards]
        print(f'SHARD {args.shard}/{args.n_shards}: {len(idx)} of {args.n_mols} targets')
        args.n_mols = len(idx)          # every count below is per-shard from here on
    # NULL CONTROL: derange the cube->molecule pairing. Everything else (target molecule,
    # atom count, ranking, scoring) is untouched, so any drop in Tanimoto is attributable to
    # the cube alone. Without this baseline a Tanimoto is uninterpretable: the model is handed
    # the true atom count and the chemical space is narrow, so the prior alone scores > 0.
    # NOTE with --n-shards: the derangement below runs WITHIN a shard, so a null-control run
    # pairs each target with another target from its own shard rather than from the whole
    # split. Still a valid null (no target keeps its own cube), just drawn from a smaller pool.
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
    # --- ADDED (scale): DataParallel batch alignment ---------------------------------------
    # models/utils.py:27 wraps EVERY model in torch.nn.DataParallel, so this script already
    # uses all visible GPUs with no sharding code. The catch is that the batch cannot be split
    # arbitrarily: DMT.forward takes `edge_mask` PRE-FLATTENED to [B*N*N, 1] and immediately
    # does `adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)` (models/dmt.py:436) with `bs`
    # taken from xh.shape. DataParallel scatters xh and edge_mask along dim 0 INDEPENDENTLY,
    # so they describe the same molecules only when the batch divides evenly across devices.
    # B=101 on 4 GPUs with N=10: xh -> 26/26/26/23 while edge_mask -> 2525 rows each = 25.25
    # molecules, and the reshape then wants 2600 elements and has 2525.
    #
    # Not avoidable by picking a tidy --batch-mols: the final chunk is min(c0+chunk, n_mols),
    # and 23123 = 61 x 379, so no divisor makes the tail divisible by 4 at n_samples=5.
    # So pad each chunk up to a multiple of the device count and drop the extras after
    # post_process. Samples are independent, so a padded entry costs at most (n_gpu - 1)
    # wasted molecules per chunk and changes nothing about the real ones.
    n_gpu = torch.cuda.device_count() if config.device.type == 'cuda' else 1
    if n_gpu > 1:
        print(f'{n_gpu} GPUs visible -- DataParallel; chunks padded to a multiple of {n_gpu} '
              f'(see models/dmt.py:436)')

    processed, pos_list, atom_list, edge_list = [], [], [], []
    t_gen = time.time()
    for bi, c0 in enumerate(range(0, args.n_mols, chunk_mols), 1):
        c1 = min(c0 + chunk_mols, args.n_mols)
        sl = slice(c0 * args.n_samples, c1 * args.n_samples)
        cn = n_nodes[sl]
        chunk_cubes = cubes[sl]
        n_real = len(cn)
        # Repeat the last entry; it is discarded below. Never the first, so a one-molecule
        # chunk still pads correctly.
        pad = (-n_real) % n_gpu
        if pad:
            cn = cn + [cn[-1]] * pad
            chunk_cubes = chunk_cubes + [chunk_cubes[-1]] * pad
        cb = len(cn); max_n = max(cn)
        assert cb % n_gpu == 0, (
            f'batch of {cb} does not divide across {n_gpu} GPUs. DMT.forward reshapes a '
            f'flattened edge_mask by the xh batch size (models/dmt.py:436), so an uneven '
            f'DataParallel split makes the two disagree. This is a bug in the padding above.')
        context = torch.stack(chunk_cubes, dim=0).to(config.device)   # build_dataset.py:570

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
        # mol_process trims each molecule to its own n_nodes (sampling.py:16-32), so chunks
        # with different max_n concatenate with no padding reconciliation.
        #
        # --- ADDED (eval): mol_process, NOT mol_process_2D. The 2D variant sets pos=None
        # (sampling.py:48), which throws away coordinates this loop has already computed one
        # line above and makes get_edm_metric -- the whole 3D stability/FCD suite -- impossible
        # to run on these samples. The tuple shape is identical, so every 2D consumer is
        # unaffected: get_2D_edm_metric unpacks pos and ignores it (evaluation/stability.py:211).
        # The one visible difference is that check_2D_stability now attaches a conformer
        # (evaluation/stability.py:107-111), so the returned mols carry 3D coordinates. That
        # changes no metric -- 2D stability reads edge_types and formal charges, never geometry
        # -- and canonical SMILES/InChIKey are conformer-invariant, so paper_metrics.py output
        # is unchanged.
        # [:n_real] drops the DataParallel padding added above. Everything downstream --
        # per-target aggregation, ranking, the pickles -- indexes as mi * n_samples + si, so a
        # padded entry surviving here would silently shift every later target's candidates.
        processed += mol_process(one_hot, pos, fc, cn, edge_types)[:n_real]
        for j, n in enumerate(cn[:n_real]):
            pos_list.append(pos[j, :n].cpu().numpy())
            atom_list.append(one_hot[j, :n].argmax(-1).cpu().numpy())
            edge_list.append(edge_types[j, :n, :n].cpu().numpy())
        if n_batches > 1:
            # --- ADDED (scale): elapsed/eta, because Run B is ~900 chunks over hours and the
            # old line carried no timing -- a stall was indistinguishable from work. Same
            # shape as the progress line tools/paper_metrics.py:130-134 already prints.
            el = time.time() - t_gen
            # peak GiB and max_n are here because they are what decides --batch-mols, and
            # guessing that number is how the training config once came out 7.7x wrong.
            # max_n matters as much as the batch does: EVERY molecule in a chunk is padded to
            # the chunk's largest, and STM sizes run median 23 / p95 35 / max 54, so a bigger
            # chunk is near-certain to contain a 54-atom molecule and pays for it on all of them.
            # That is the effect that eats into the gain from a bigger batch.
            mem = (torch.cuda.max_memory_allocated() / 1024 ** 3
                   if config.device.type == 'cuda' else 0.0)
            print(f'  batch {bi}/{n_batches}: targets {c0}-{c1 - 1} ({n_real} molecules'
                  f'{f", +{pad} pad" if pad else ""}, max_n {max_n})  '
                  f'{el:.0f}s elapsed  eta {el / bi * (n_batches - bi):.0f}s  '
                  f'{len(processed) / el:.1f} mol/s  peak {mem:.1f} GiB', flush=True)
        del context, node_mask, edge_mask, z, edge_z, x_node, x_edge, pos, one_hot, edge_types
        torch.cuda.empty_cache()
    # 2D only, deliberately, and this is the AUTHORITATIVE number: this model has
    # pred_edge=True, so bond orders are PREDICTED by the network and check_2D_stability reads
    # them directly instead of re-deriving them from geometry. It needs only atom_decoder and
    # atom_fc_num, both of which ters_hcno and stm_10elem define.
    #
    # NOTE for anyone comparing against an older log: the stability fields here were
    # STRUCTURALLY ZERO until D-11 was closed (evaluation/stability.py:143) -- 'TERS'/'STM'
    # hit an early return, so `mol_stable 0.0, atom_stable 0.0` printed alongside
    # `Validity 0.96` on the same molecules. Any stability number from a run before that fix
    # is an early return, not a score. Validity/Complete/Unique were always real.
    #
    # The 3D metric (get_edm_metric -> check_stability) is now reachable too -- see the
    # TERS/STM branch at evaluation/stability.py:50-56 -- but it is NOT run here. It re-infers
    # bonds from interatomic distances using margins tuned on QM9 (H/C/N/O/F only), so for a
    # corpus containing Si/P/S/Cl/Br it is approximate and uncalibrated. It belongs with the
    # rest of the distributional suite in tools/eval_cond.py, which reads the pickle written
    # below and states that caveat where the number is reported. What prints here is the
    # at-a-glance smoke signal for this run, nothing more.
    stab, rdkit_dict, rd_mols = get_2D_edm_metric(dataset_info)(processed)
    print(f'\nstability: {stab}')
    print(f'rdkit    : {rdkit_dict}')

    # --- ADDED (eval): the tensor tuples, for tools/eval_cond.py ---------------------------
    # Written rather than recomputed because regenerating them costs a full 250-step ancestral
    # run per cell. Small: at n~23 atoms a tuple is pos 23x3 + atom_type 23 + edge_type 23x23
    # + fc 23, i.e. a few kB -- three orders of magnitude under the geometry payload, which is
    # why this one is not behind a flag.
    with open(os.path.join(args.out, f'processed_{args.split}{tag}.pkl'), 'wb') as f:
        pickle.dump({'processed': processed,
                     'n_mols': args.n_mols, 'n_samples': args.n_samples,
                     'split': args.split, 'shuffled_cubes': bool(args.shuffle_cubes),
                     'modality': args.modality, 'config': args.config,
                     'config_qm9': args.config_qm9,
                     'info_name': config.data.info_name,
                     'ckpt': args.ckpt, 'step': int(ck['step'])}, f)

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

    if args.save_sdf:
        write_portable(args.out, args.split, tag, results, rd_mols, args.n_samples)


if __name__ == '__main__':
    main()
