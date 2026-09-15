"""Distributional metrics for cube-conditioned samples -- the half `paper_metrics.py` cannot see.

**This file computes no chemistry.** Every metric definition is imported from `evaluation/`,
the same package `run_lib.diffspectra_evaluate` imports. What is written here is the driver:
loading samples, building the reference sets once, and reporting.

Why it exists
-------------
Evaluation for the cube modalities runs through `tools/`, not through `main.py --mode eval`,
because `sampling.py` has no cube branch (defect D-8, open -- docs/05-progress-log.md). The
existing `tools/` path scores *similarity to the target*:

    tools/sample_cond.py   generation
    tools/paper_metrics.py top-k, Tanimoto, MCES, Fraggle, functional group

That leaves the *distributional* half of the DiffSpectra table unreachable -- FCD, SNN, Frag,
Scaf, IntDiv, novelty, and atom/molecule stability all live in `run_lib.diffspectra_evaluate`.
This file recovers them, from the `processed_<split>.pkl` that `sample_cond.py` already writes,
by importing the same three factories that `run_lib.py:373-377` uses:

    get_edm_metric      3D stability + validity/uniqueness/novelty   (evaluation/stability.py)
    get_2D_edm_metric   2D stability + validity/uniqueness/novelty   (evaluation/stability.py)
    get_moses_metrics   FCD, SNN, Frag, Scaf, IntDiv                 (evaluation/mose_metric.py)

Metric-3D does not work on this corpus. Use Metric-2D.
------------------------------------------------------
Not a suspicion -- MEASURED, 2026-09-10 (job 1228484), on 500 ground-truth STM test structures,
i.e. molecules that are correct by construction and that a perfect model would reproduce exactly:

    Metric-3D   mol_stable 0.250   atom_stable 0.823      <- the CEILING, not 1.0
    Metric-2D   mol_stable 0.974   atom_stable 0.999

Metric-3D calls three quarters of TRUE structures unstable, so on this corpus it reports the
bond tables rather than the model. Two candidate causes, not yet separated: its margins were
tuned on QM9 true samples (H/C/N/O/F only, bond_analyze.py:119-120) while this corpus contains
Si/P/S/Cl/Br; and `check_stability` scores against `allowed_bonds`, which is CHARGE-BLIND, while
`stm_10elem['atom_fc_num']` records ~24k O-1, ~14k N1 and ~13k S1. `check_2D_stability` uses
`allowed_fc_bonds` instead, which is precisely the difference.

Metric-2D is the authoritative one and lands where a correct metric should: these models set
`pred_edge=True`, so bond orders are PREDICTED by the network and `check_2D_stability` reads
them directly rather than re-deriving them from geometry. Metric-3D is still computed and
printed, because a relative movement across checkpoints of the same corpus may still carry
information -- but every output of this file carries CAVEAT_3D so the number cannot be lifted
out without it. Re-run the calibration with `--ground-truth N` on any new corpus.

Do not compare stability against a pre-D-11 log
-----------------------------------------------
2D stability for 'TERS'/'STM' was an early return -- a hardcoded `(0, 0)` -- until D-11 was
closed at evaluation/stability.py:143 (docs/05-progress-log.md:173-183). Every stability figure
printed by a run before that fix is meaningless, which is why it could sit next to a RDKit
`Validity 0.96` on the same molecules. Validity, Complete, Unique and Novelty were always real.
These are the first genuine stability numbers this project has produced.

Run
---
    # one cell
    python -m tools.eval_cond --samples $OUT/test/processed_test.pkl

    # all four cells, reference sets built once for all of them
    python -m tools.eval_cond --samples $OUT/*/processed_*.pkl --n-jobs 32

    # CALIBRATION -- run this first on any new corpus, before quoting Metric-3D
    python -m tools.eval_cond --config-qm9 configs/base_stm.py --ground-truth 500

    # smoke: bound the reference set (changes FCD -- recorded in the JSON as
    # reference_truncated, and never comparable to a full-reference run)
    python -m tools.eval_cond --samples <pkl> --max-ref 2000 --no-moses
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time

try:
    from paths import under                 # `python3 tools/eval_cond.py`
except ImportError:
    from tools.paths import under           # `python3 -m tools.eval_cond`

import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_dataset                                          # noqa: E402
from evaluation import (get_edm_metric, get_2D_edm_metric,                # noqa: E402
                        get_moses_metrics)

# MEASURED 2026-09-10, job 1228484, on 500 GROUND-TRUTH STM test structures -- molecules that
# are correct by construction, so this is the CEILING the metric can report on this corpus:
#
#     Metric-3D  mol_stable 0.250   atom_stable 0.823
#     Metric-2D  mol_stable 0.974   atom_stable 0.999
#
# The falsifier fired. Metric-3D calls three quarters of TRUE structures unstable, so on this
# corpus it is measuring the bond tables, not the model. Metric-2D lands where a correct metric
# should. Two candidate causes, not yet separated: (a) margin1/2/3 were tuned on QM9 true
# samples, i.e. H/C/N/O/F only (bond_analyze.py:119-120), while this corpus has Si/P/S/Cl/Br;
# (b) check_stability compares against `allowed_bonds`, which is CHARGE-BLIND, while
# stm_10elem['atom_fc_num'] records ~24k O-1, ~14k N1 and ~13k S1 -- every one of those atoms
# is scored against its neutral valence. check_2D_stability uses `allowed_fc_bonds` instead,
# which is exactly the difference.
CAVEAT_3D = ('Metric-3D is NOT trustworthy on this corpus: on 500 ground-truth test structures '
             'it reports mol_stable 0.250 (2D reports 0.974 on the same molecules), so its '
             'ceiling here is 0.25, not 1.0. It re-infers bonds from distances with QM9-tuned '
             'margins and a charge-blind valence table. Use Metric-2D, which reads the '
             'predicted bond orders directly (pred_edge=True). Do not quote Metric-3D without '
             'this line.')


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.get_config()


# --- reference sets -----------------------------------------------------------------------
# MEASURED 2026-09-10 (job 1228484), because the guess was wrong and worth correcting:
#     train (novelty, 184157 mols)   23 s
#     test  (MOSES,    23123 mols)    3 s
#     MOSES intermediate statistics   8 s   (on a GH200; `device` reaches FCD's ChemNet)
# 34 seconds, not the "minutes" this was written assuming. So caching is OPT-IN (--cache) and
# not the default: the cache file is 124 MB, /scratch on this project runs near its 250 GB
# quota, and this repo has already lost two jobs to quota kills (see tools/paths.py). Trading
# 124 MB of a scarce quota for 34 s of a GPU-node walk is not a trade worth making by default.
# Pass --cache when running many cells back to back and the seconds start to matter.
#
# The aux config (base_stm.py / base_ters.py) is the right one to build them from: it sets
# load_cubes=False, so the 917 KB z-stack per molecule is never read. Building the reference
# sets from the MAIN config instead would be ~169 GB of I/O for data discarded on the next
# line -- which is the entire reason base_stm.py exists (see its module docstring).

def _cache_path(cfg_aux_path, config, max_ref):
    key = '|'.join([os.path.abspath(cfg_aux_path),
                    str(getattr(config.data, 'graph_root', config.data.root)),
                    str(config.data.info_name), str(max_ref)])
    return under('eval-cache', f'refsets_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl')


def reference_sets(cfg_aux_path, max_ref, use_cache=False):
    """(train_smiles, test_mols, dataset_info) -- the novelty/MOSES references.

    Mirrors run_lib.py:367-369 in WHICH split feeds which reference (element 1 of the
    five-tuple is train, element 3 is test), but returns the train side as SMILES STRINGS.

    That is the fix for the OOM that killed job 1235060 at 217 GB. Novelty is a set-membership
    test over SMILES -- `eval_rdmol` immediately reduces train_mols to
    `[Chem.MolToSmiles(m) for m in train_mols]` (stability.py:165-167) and never touches the
    Mol objects again. Holding 184157 live RDKit Mols only to derive strings from them costs
    tens of GB, and it is still resident when `get_moses_metrics` forks its worker pool, so
    every worker inherits it. Converting once and dropping the Mols removes that entirely.
    test_mols must stay as Mol objects: MOSES embeds them through ChemNet for FCD.
    """
    config = load_config(cfg_aux_path)
    cache = _cache_path(cfg_aux_path, config, max_ref)

    _, train_ds, _, test_ds, dataset_info = get_dataset(config)

    if use_cache and os.path.exists(cache):
        print(f'reference sets: reusing {cache}', flush=True)
        with open(cache, 'rb') as f:
            blob = pickle.load(f)
        return blob['train_smiles'], blob['test_mols'], dataset_info

    def collect(ds, what):
        n = len(ds) if max_ref <= 0 else min(len(ds), max_ref)
        t0 = time.time()
        mols = []
        for i in range(n):
            m = ds[i].rdmol
            if m is not None:
                mols.append(m)
        print(f'  {what}: {len(mols)} mols of {len(ds)} in {time.time() - t0:.0f}s', flush=True)
        return mols

    print(f'building reference sets from {cfg_aux_path} '
          f'(max_ref={max_ref if max_ref > 0 else "all"})', flush=True)
    train_mols = collect(train_ds, 'train (novelty)')
    test_mols = collect(test_ds, 'test  (MOSES) ')

    # Reduce to SMILES and drop the Mols before anything forks a worker pool. See the
    # docstring: this is the difference between fitting in 217 GB and not.
    from rdkit import Chem
    t0 = time.time()
    train_smiles = [Chem.MolToSmiles(m) for m in train_mols]
    print(f'  train -> {len(train_smiles)} SMILES in {time.time() - t0:.0f}s '
          f'(Mol objects released)', flush=True)
    del train_mols

    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump({'train_smiles': train_smiles, 'test_mols': test_mols,
                         'max_ref': max_ref, 'config': os.path.abspath(cfg_aux_path)}, f)
        print(f'  cached -> {cache}', flush=True)
    return train_smiles, test_mols, dataset_info


# --- the calibration run -------------------------------------------------------------------
# Metric-3D re-infers bonds from distances with QM9-tuned margins, so before any generated
# number from it is quotable, the metric has to be shown to work on molecules that are known
# to be right. --ground-truth scores the corpus's OWN test structures: a perfect model would
# emit exactly these, so their score is the CEILING this metric can report on this corpus.
#
#   near 1.0   -> the margins transfer; Metric-3D on generated samples is meaningful
#   well below -> the margins are wrong for these elements, and Metric-3D is measuring the
#                 bond tables, not the model. That is the documented falsifier for the caveat
#                 in docs/09-research-summary.md.
def ground_truth_cells(cfg_aux_path, n, split='test', seed=0):
    import torch
    config = load_config(cfg_aux_path)
    _, train_ds, _, test_ds, _ = get_dataset(config)
    # --- ADDED: `split='train'` is the NULL CONTROL for the distributional half.
    # The corpus is one 8:1:1 random split, so train and test share a distribution BY
    # CONSTRUCTION. That means a model which learned only the training marginal -- and ignored
    # the conditioning entirely -- would still score well on FCD/Frag/Scaf against the test
    # reference. Scoring real TRAINING molecules against that same reference measures exactly
    # that floor. If the generated set does not beat it, the distributional metrics are
    # reporting corpus statistics rather than anything the model learned about a cube.
    # This is the distributional analogue of --shuffle-cubes for the similarity half.
    ds = train_ds if split == 'train' else test_ds
    n = len(ds) if n <= 0 else min(n, len(ds))
    # Random draw, not the first n: the sidecar is in corpus order and a prefix is not a
    # random sample of anything.
    sel = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    print(f'building {n} ground-truth molecules from the {split} split', flush=True)

    processed = []
    for i in sel:
        s = ds[i]
        k = int(s.num_atom)
        # The sidecar stores edges sparse; check_2D_stability wants the dense (k, k) matrix
        # that post_process would have produced.
        dense = torch.zeros(k, k, dtype=torch.long)
        ei, et = s.edge_index, s.edge_type
        for e in range(ei.shape[1]):
            a, b = int(ei[0, e]), int(ei[1, e])
            if a < k and b < k:
                dense[a, b] = dense[b, a] = int(et[e])
        processed.append((s.pos[:k].float(), s.atom_type[:k].long(),
                          dense, s.fc[:k].long()))
    return {'processed': processed, 'n_mols': len(processed), 'n_samples': 1,
            'split': split, 'shuffled_cubes': False, 'config_qm9': cfg_aux_path,
            'ckpt': f'GROUND-TRUTH-{split.upper()}', 'step': -1}


# --- parallel stability ---------------------------------------------------------------------
# `get_edm_metric` / `get_2D_edm_metric` loop over `processed` serially inside evaluation/, and
# check_stability is an O(n^2) Python double loop per molecule. At 115615 molecules that loop is
# the bottleneck, and it is embarrassingly parallel -- molecules do not interact.
#
# evaluation/ is NOT edited: it is shared with the QM9S path and the metric definitions must stay
# upstream's. Instead the SAME factory is called inside each worker on a slice, and the parts are
# recombined here. This is the pattern tools/paper_metrics.py:91-135 already uses -- a
# module-level `_SHARED` set before a fork-context Pool, so the (large) tensor list is inherited
# copy-on-write instead of pickled to every worker.
#
# Recombination, and why it is exact rather than an average of averages:
#   * the factory returns FRACTIONS, so each is multiplied back up by its own denominator --
#     mol_stable by the chunk's molecule count, atom_stable by the chunk's atom count (which is
#     recomputed here from the tuples, since the factory does not return it).
#   * Validity / Complete / Unique / Novelty are NOT per-chunk quantities -- Unique is a set over
#     the whole cell. So workers pass their rdkit mols back and `eval_rdmol` runs ONCE in the
#     parent over the combined list, exactly as the serial factory would.
# Workers are given train_mols=None so no worker re-derives SMILES for 184157 training molecules.
_SHARED_PROCESSED = None


def _stability_chunk(job):
    lo, hi, kind, info = job
    chunk = _SHARED_PROCESSED[lo:hi]
    factory = get_edm_metric if kind == '3d' else get_2D_edm_metric
    stab, _, mols = factory(info, None)(chunk)
    n_atoms = sum(len(t[1]) for t in chunk)
    return (round(stab['mol_stable'] * len(chunk)),      # back to counts
            stab['atom_stable'] * n_atoms,
            len(chunk), n_atoms, mols)


def stability_metrics(processed, dataset_info, kind, train_smiles, workers):
    """(stability_dict, rdkit_dict, rd_mols) -- identical to the serial factory's output."""
    global _SHARED_PROCESSED
    from evaluation.rdkit_metric import eval_rdmol

    factory = get_edm_metric if kind == '3d' else get_2D_edm_metric
    if workers <= 1 or len(processed) < 2 * workers:
        # Serial. The factory is given train_mols=None and eval_rdmol is re-run here on the
        # same mols with the SMILES -- identical to what the factory does internally
        # (stability.py:165-167), but without needing 184157 RDKit Mol objects alive.
        stab, _, mols = factory(dataset_info, None)(processed)
        return stab, eval_rdmol(mols, train_smiles), mols

    import multiprocessing as mp
    n = len(processed)
    # Enough chunks to keep every worker fed even when molecule sizes vary, but few enough that
    # the per-chunk factory setup stays negligible.
    n_chunks = min(n, workers * 4)
    edges = [round(i * n / n_chunks) for i in range(n_chunks + 1)]
    jobs = [(edges[i], edges[i + 1], kind, dataset_info)
            for i in range(n_chunks) if edges[i] < edges[i + 1]]

    _SHARED_PROCESSED = processed          # inherited by fork; never sent over a pipe
    ctx = mp.get_context('fork')
    t0 = time.time()
    mol_stable = atom_stable = n_mols = n_atoms = 0
    all_mols = []
    with ctx.Pool(workers) as pool:
        for i, (ms, ats, nm, na, mols) in enumerate(pool.imap(_stability_chunk, jobs), 1):
            mol_stable += ms; atom_stable += ats; n_mols += nm; n_atoms += na
            all_mols.extend(mols)
            if i == len(jobs) or i % max(1, len(jobs) // 4) == 0:
                el = time.time() - t0
                print(f'    {kind}: {n_mols}/{n} molecules  {el:.0f}s  '
                      f'eta {el / i * (len(jobs) - i):.0f}s', flush=True)

    stab = {'mol_stable': mol_stable / n_mols, 'atom_stable': atom_stable / n_atoms}
    return stab, eval_rdmol(all_mols, train_smiles), all_mols


def score_cell(name, payload, dataset_info, train_smiles, moses_fn, workers=1):
    """Every metric run_lib.diffspectra_evaluate reports, on one cell's samples."""
    processed = payload['processed']
    print(f'\n{"=" * 90}\n== {name}   {len(processed)} molecules   '
          f'{"SHUFFLED null" if payload.get("shuffled_cubes") else "matched"}   '
          f'ckpt step {payload.get("step")}\n{"=" * 90}', flush=True)

    out = {'cell': name, 'n_molecules': len(processed),
           'shuffled_cubes': bool(payload.get('shuffled_cubes')),
           'split': payload.get('split'), 'ckpt': payload.get('ckpt'),
           'step': payload.get('step'), 'caveat_3d': CAVEAT_3D}

    stab_3d, rdkit_3d, _ = stability_metrics(processed, dataset_info, '3d', train_smiles,
                                             workers)
    print('Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, '
          'complete: %.4f, unique & valid: %.4f, novelty: %.4f'
          % (stab_3d['atom_stable'], stab_3d['mol_stable'], rdkit_3d['Validity'],
             rdkit_3d['Complete'], rdkit_3d['Unique'], rdkit_3d['Novelty']))
    print(f'   ^ {CAVEAT_3D}')
    out['metric_3d'] = {**stab_3d, **rdkit_3d}

    stab_2d, rdkit_2d, mols_2d = stability_metrics(processed, dataset_info, '2d', train_smiles,
                                                   workers)
    print('Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, '
          'complete: %.4f, unique & valid: %.4f, novelty: %.4f'
          % (stab_2d['atom_stable'], stab_2d['mol_stable'], rdkit_2d['Validity'],
             rdkit_2d['Complete'], rdkit_2d['Unique'], rdkit_2d['Novelty']))
    out['metric_2d'] = {**stab_2d, **rdkit_2d}

    if moses_fn is not None:
        moses = moses_fn(mols_2d)
        print('Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f'
              % (moses['FCD'], moses['SNN'], moses['Frag'], moses['Scaf'], moses['IntDiv']))
        out['moses'] = {k: float(v) for k, v in moses.items()}
    else:
        print('MOSES metrics skipped (--no-moses)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', nargs='+', default=[],
                    help='processed_<split>[_shuf].pkl files written by tools/sample_cond.py. '
                         'Pass every cell at once so the reference sets are built once.')
    ap.add_argument('--ground-truth-split', default='test', choices=['test', 'train'],
                    help="which split --ground-truth draws from. 'train' is the NULL CONTROL "
                         "for the distributional metrics: train and test share a distribution "
                         "by construction (one 8:1:1 random split), so real training molecules "
                         "scored against the test reference are the floor any generated set "
                         "must beat to have shown anything.")
    ap.add_argument('--ground-truth', type=int, default=0, metavar='N',
                    help='ALSO score N ground-truth test structures as a calibration cell. '
                         'This is the ceiling Metric-3D can report on this corpus: if it is '
                         'not near 1.0 the QM9-tuned bond margins do not transfer and no '
                         'Metric-3D number here is quotable. Needs --config-qm9 if --samples '
                         'is empty.')
    ap.add_argument('--config-qm9', default=None,
                    help='aux config for the reference sets (default: the one recorded in the '
                         'sample pickle, i.e. base_stm.py or base_ters.py)')
    ap.add_argument('--max-ref', type=int, default=0,
                    help='cap each reference set at N molecules (0 = all). SMOKE RUNS ONLY: a '
                         'truncated reference changes FCD and inflates novelty, so such a run '
                         'is never comparable to a full one. Recorded in the output JSON.')
    ap.add_argument('--n-jobs', type=int, default=8,
                    help='MOSES worker processes (run_lib.py:377 hardcodes 32, which is wrong '
                         'on a login node)')
    ap.add_argument('--workers', type=int, default=0,
                    help='processes for the stability loops (default: os.cpu_count()). Those '
                         'loops are O(n^2) Python per molecule and dominate at 100k+ molecules. '
                         '1 forces the untouched serial path.')
    ap.add_argument('--no-moses', action='store_true',
                    help='skip FCD/SNN/Frag/Scaf/IntDiv -- much faster, for smoke runs')
    ap.add_argument('--cache', action='store_true',
                    help='cache the reference sets to $TERS_EXP_ROOT/eval-cache. OFF by '
                         'default: measured at 34s to build but 124 MB on disk, and /scratch '
                         'here runs near quota. Worth it only across many back-to-back cells.')
    ap.add_argument('--out', default='', help='write the JSON here (default: beside --samples[0])')
    args = ap.parse_args()

    if not args.samples and not args.ground_truth:
        raise SystemExit('nothing to score: pass --samples, --ground-truth, or both')
    if args.ground_truth and not args.samples and not args.config_qm9:
        raise SystemExit('--ground-truth without --samples needs --config-qm9')

    cells = []
    for path in args.samples:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if 'processed' not in payload:
            raise SystemExit(
                f'{path} has no "processed" key. It was probably written by the pre-rename '
                f'tools/sample_ters.py, which called mol_process_2D and discarded positions. '
                f'Re-run tools/sample_cond.py to produce it.')
        name = os.path.basename(os.path.dirname(os.path.abspath(path))) or payload['split']
        cells.append((f'{name}/{os.path.basename(path)}', path, payload))

    # Every cell must share an aux config, or the reference sets differ between them and the
    # matched-vs-shuffled comparison the protocol rests on is not like-for-like.
    aux = args.config_qm9 or (cells[0][2].get('config_qm9') if cells else None)
    if aux is None:
        raise SystemExit('no aux config recorded in the sample pickle; pass --config-qm9')
    others = {c[2].get('config_qm9') for c in cells if c[2].get('config_qm9')}
    if args.config_qm9 is None and len(others) > 1:
        raise SystemExit(f'cells disagree on the aux config {sorted(others)}; '
                         f'pass --config-qm9 explicitly if that is intended')

    # Placed AFTER the aux-config resolution so it uses the same one the cells do, and BEFORE
    # the reference sets so the (expensive) dataset build is shared with them.
    if args.ground_truth:
        _lbl = ('GROUND-TRUTH train (distributional NULL)' if args.ground_truth_split == 'train'
                else 'GROUND-TRUTH test (calibration ceiling)')
        cells.insert(0, (_lbl, None,
                         ground_truth_cells(aux, args.ground_truth, args.ground_truth_split)))

    train_smiles, test_mols, dataset_info = reference_sets(aux, args.max_ref, args.cache)

    moses_fn = None
    if not args.no_moses:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'building MOSES reference statistics over {len(test_mols)} test mols '
              f'(n_jobs={args.n_jobs}, device={device})', flush=True)
        t0 = time.time()
        moses_fn = get_moses_metrics(test_mols, n_jobs=args.n_jobs, device=device)
        print(f'  done in {time.time() - t0:.0f}s', flush=True)

    workers = args.workers or (os.cpu_count() or 1)
    print(f'stability loops: {workers} worker process(es)', flush=True)
    results = [score_cell(name, payload, dataset_info, train_smiles, moses_fn, workers)
               for name, _, payload in cells]

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.samples[0])) if args.samples
        else under('stm-eval'), 'eval_cond.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'cells': results,
                   'config_qm9': aux,
                   'max_ref': args.max_ref,
                   'reference_truncated': bool(args.max_ref > 0),
                   'n_train_ref': len(train_smiles), 'n_test_ref': len(test_mols),
                   'moses': not args.no_moses,
                   'workers': workers,
                   'caveat_3d': CAVEAT_3D}, f, indent=2, default=str)
    print(f'\nwrote {out_path}')
    if args.max_ref > 0:
        print(f'WARNING: reference sets truncated to {args.max_ref}; FCD and novelty from this '
              f'run are not comparable to a full-reference run.')


if __name__ == '__main__':
    main()
