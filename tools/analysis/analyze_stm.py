"""Per-split analysis of the STM corpus.

Reports composition, size, geometry and SMARTERS planarity, formal charges, image cube
statistics and split overlap. Writes ``analysis.json``, per-record, per-CID and summary CSVs,
and figures.

Usage
-----
    python -m tools.analysis.analyze_stm
    python -m tools.analysis.analyze_stm --self-test
"""

import argparse
import importlib.util
import json
import os
import os.path as osp
import sys
import types
from collections import Counter

import numpy as np

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO)
from env import data_root, output_root
from tools.analysis.planarity import SMARTERS_THRESHOLDS, planarity_of

SPLITS = ('train', 'val', 'test')

SPLIT_COLOR = {'train': '#2a78d6', 'val': '#eb6834', 'test': '#1baf7a'}
SPLIT_DASH = {'train': '-', 'val': '--', 'test': ':'}
# Colour of single-series plots.
ONE = '#2a78d6'
INK, MUTED = '#0b0b0b', '#52514e'

BOND_ORDER_NAME = {1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic'}


# --- importing the pipeline without importing the whole training stack -----------------

def load_bypassing_init(package_dir, module, alias):
    """Import ``<package_dir>/<module>.py`` without running the package's ``__init__``.

    The module is registered under a synthetic parent package ``alias`` so that its relative
    imports resolve.
    """
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [package_dir]
        sys.modules[alias] = pkg
    full = f'{alias}.{module}'
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, osp.join(package_dir, f'{module}.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[full]
        raise
    return mod


def stm_preprocess():
    """Return ``datasets.preprocess_stm``, importing it directly if ``datasets`` cannot load."""
    try:
        from datasets import preprocess_stm
        return preprocess_stm
    except Exception:
        return load_bypassing_init(osp.join(REPO, 'datasets'), 'preprocess_stm',
                                   '_stm_analysis_datasets')


def bond_symbol_fn():
    """Return ``evaluation.cal_geometry.get_bond_symbol``, or ``None`` if unavailable."""
    try:
        from evaluation.cal_geometry import get_bond_symbol
        return get_bond_symbol
    except Exception:
        pass
    try:
        # cal_geometry only imports tqdm; provide a no-op when it is unavailable.
        if 'tqdm' not in sys.modules:
            try:
                import tqdm  # noqa: F401
            except ImportError:
                shim = types.ModuleType('tqdm')
                shim.tqdm = lambda it=None, *a, **k: it
                sys.modules['tqdm'] = shim
        mod = load_bypassing_init(osp.join(REPO, 'evaluation'), 'cal_geometry',
                                  '_stm_analysis_evaluation')
        return mod.get_bond_symbol
    except Exception:
        return None


# --- the single pass -------------------------------------------------------------------

def molecule_table(records, symbols, want_smiles=True, progress_every=25000):
    """Compute every per-molecule quantity in one pass over the sidecar records.

    Returns ``(table, extras)``: ``table`` maps each column name to an array of length
    ``len(records)``; ``extras`` holds the remaining per-record lists and counters.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    get_bond_symbol = bond_symbol_fn()
    n = len(records)
    n_elem = len(symbols['symbol_list'])

    col = dict(
        n_atoms=np.zeros(n, np.int32),
        n_heavy=np.zeros(n, np.int32),
        n_bonds=np.zeros(n, np.int32),
        n_rings=np.zeros(n, np.int32),
        n_aromatic_rings=np.zeros(n, np.int32),
        max_ring_size=np.zeros(n, np.int32),
        mol_weight=np.zeros(n, np.float64),
        # Geometry in Angstrom, on mean-centred coordinates.
        radius=np.zeros(n, np.float64),
        gyration=np.zeros(n, np.float64),
        thickness=np.zeros(n, np.float64),        # extent along the thinnest principal axis
        z_extent=np.zeros(n, np.float64),         # extent along the file's own z axis
        # SMARTERS planarity; NaN for fewer than 3 atoms.
        plane_rmsd=np.full(n, np.nan),            # A, RMS distance from the best-fit plane
        planarity_pca=np.full(n, np.nan),         # %
        planarity_rms=np.full(n, np.nan),         # %
        # The same metrics over heavy atoms only.
        plane_rmsd_heavy=np.full(n, np.nan),
        planarity_pca_heavy=np.full(n, np.nan),
        planarity_rms_heavy=np.full(n, np.nan),
        n_fc=np.zeros(n, np.int32),               # atoms carrying a non-zero formal charge
        used_fallback=np.zeros(n, bool),
    )
    elem_counts = np.zeros((n, n_elem), np.int32)
    order_counts = np.zeros((n, 5), np.int32)     # index = bond order, 0 unused

    bond_symbols, formulas, fc_states = Counter(), Counter(), Counter()
    bond_lengths_by_order = {o: [] for o in (1, 2, 3, 4)}
    smiles, cids, sanitize_failures = [], [], 0
    formula_per_row, h5_index = [], np.zeros(n, np.int64)

    for i, r in enumerate(records):
        if progress_every and i and i % progress_every == 0:
            print(f'    ...{i}/{n}', flush=True)

        z = r['charge'].numpy()
        pos = r['pos'].numpy().astype(np.float64)
        mol = r['rdmol']

        col['n_atoms'][i] = len(z)
        col['n_heavy'][i] = int((z != 1).sum())
        col['used_fallback'][i] = bool(r.get('used_fallback', False))
        cids.append(r.get('cid'))
        h5_index[i] = int(r.get('h5_index', i))

        for zi, k in zip(*np.unique(z, return_counts=True)):
            elem_counts[i, symbols['index'][int(zi)]] = k

        fc = r['fc'].numpy()
        col['n_fc'][i] = int((fc != 0).sum())
        for zi, fci in zip(z, fc):
            if int(fci) != 0:
                fc_states[f'{symbols["symbol"][int(zi)]}{int(fci):+d}'] += 1

        # --- geometry ---------------------------------------------------------------
        centred = pos - pos.mean(axis=0, keepdims=True)
        d = np.linalg.norm(centred, axis=-1)
        col['radius'][i] = d.max()
        col['gyration'][i] = float(np.sqrt((d ** 2).mean()))
        col['z_extent'][i] = float(centred[:, 2].max() - centred[:, 2].min())
        if len(pos) >= 3:
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
            off = centred @ vh[-1]
            col['thickness'][i] = float(off.max() - off.min())

        pc, pr, rm = planarity_of(pos)
        col['planarity_pca'][i], col['planarity_rms'][i], col['plane_rmsd'][i] = pc, pr, rm
        heavy = pos[z != 1]
        pc, pr, rm = planarity_of(heavy) if len(heavy) >= 3 else (np.nan,) * 3
        (col['planarity_pca_heavy'][i], col['planarity_rms_heavy'][i],
         col['plane_rmsd_heavy'][i]) = pc, pr, rm

        # --- bonds ------------------------------------------------------------------
        # encode_graph stores each bond in both directions, hence the // 2.
        et = r['edge_type'].numpy()
        ei = r['edge_index'].numpy()
        col['n_bonds'][i] = len(et) // 2
        for o, k in zip(*np.unique(et, return_counts=True)):
            if 0 <= int(o) < 5:
                order_counts[i, int(o)] = k // 2
        if ei.size:
            lens = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=-1)
            for o in (1, 2, 3, 4):
                m = et == o
                if m.any():
                    bond_lengths_by_order[o].extend(lens[m][::2].tolist())

        if get_bond_symbol is not None:
            for b in mol.GetBonds():
                bond_symbols[get_bond_symbol(b)[0]] += 1

        # --- ring / formula / SMILES ------------------------------------------------
        try:
            rings = Chem.GetSymmSSSR(mol)
            col['n_rings'][i] = len(rings)
            if len(rings):
                col['max_ring_size'][i] = max(len(x) for x in rings)
        except Exception:
            pass

        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit.Chem.Descriptors import MolWt
        try:
            col['mol_weight'][i] = MolWt(mol)
            f = CalcMolFormula(mol)
            formulas[f] += 1
            formula_per_row.append(f)
        except Exception:
            formula_per_row.append(None)

        # The sidecar molecules are kekulized; sanitize a copy to recover aromaticity.
        if want_smiles:
            try:
                m = Chem.Mol(mol)
                Chem.SanitizeMol(m)
                smiles.append(Chem.MolToSmiles(m))
                ri = m.GetRingInfo()
                col['n_aromatic_rings'][i] = sum(
                    all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
                    for ring in ri.AtomRings())
            except Exception:
                sanitize_failures += 1
                smiles.append(None)
        else:
            # Keep one entry per record.
            smiles.append(None)

    extras = dict(
        elem_counts=elem_counts,
        order_counts=order_counts,
        bond_symbols=bond_symbols,
        bond_lengths_by_order=bond_lengths_by_order,
        formulas=formulas,
        fc_states=fc_states,
        smiles=smiles,
        cids=cids,
        h5_index=h5_index,
        formula_per_row=formula_per_row,
        sanitize_failures=sanitize_failures,
        have_bond_symbols=get_bond_symbol is not None,
    )
    return col, extras


def _pct(a, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    """Percentiles, mean, std and count of the finite values; ``{}`` if there are none."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {f'p{q}': float(np.percentile(a, q)) for q in qs} | {
        'mean': float(a.mean()), 'std': float(a.std()), 'n': int(a.size)}


def _threshold_table(values, thresholds):
    """Fraction of finite ``values`` at or below each threshold."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {f'{t:g}': (float((v <= t).mean()) if v.size else None) for t in thresholds}


def summarize(col, extras, symbols, thresholds=SMARTERS_THRESHOLDS):
    """Turn the per-molecule table into the JSON report."""
    n = len(col['n_atoms'])
    elem_counts = extras['elem_counts']
    total_atoms = int(elem_counts.sum())

    per_element = {}
    for j, sym in enumerate(symbols['symbol_list']):
        c = int(elem_counts[:, j].sum())
        per_element[sym] = dict(
            atoms=c,
            atom_fraction=c / total_atoms if total_atoms else 0.0,
            molecules=int((elem_counts[:, j] > 0).sum()),
            molecule_fraction=float((elem_counts[:, j] > 0).mean()),
        )

    order_counts = extras['order_counts']
    bonds_by_order = {BOND_ORDER_NAME[o]: int(order_counts[:, o].sum())
                      for o in (1, 2, 3, 4)}
    n_bond_total = sum(bonds_by_order.values())

    smiles = [s for s in extras['smiles'] if s]
    # A record is one image; a CID can recur in several orientations.
    cids = [c for c in extras['cids'] if c]
    cid_mult = Counter(Counter(cids).values())
    out = dict(
        n_molecules=n,
        n_atoms_total=total_atoms,
        n_bonds_total=n_bond_total,
        elements=per_element,
        size=dict(
            atoms=_pct(col['n_atoms']),
            heavy_atoms=_pct(col['n_heavy']),
            atoms_histogram={int(k): int(v) for k, v in
                             sorted(Counter(col['n_atoms'].tolist()).items())},
            heavy_atoms_histogram={int(k): int(v) for k, v in
                                   sorted(Counter(col['n_heavy'].tolist()).items())},
            hydrogen_fraction=float(
                elem_counts[:, symbols['index'][1]].sum() / total_atoms)
            if total_atoms else 0.0,
        ),
        molecular_weight=_pct(col['mol_weight']),
        bonds=dict(
            per_molecule=_pct(col['n_bonds']),
            by_order=bonds_by_order,
            by_order_fraction={k: (v / n_bond_total if n_bond_total else 0.0)
                               for k, v in bonds_by_order.items()},
            length_by_order={BOND_ORDER_NAME[o]: _pct(v)
                             for o, v in extras['bond_lengths_by_order'].items() if v},
            top_symbols=(dict(extras['bond_symbols'].most_common(20))
                         if extras['have_bond_symbols'] else None),
        ),
        rings=dict(
            per_molecule=_pct(col['n_rings']),
            histogram={int(k): int(v) for k, v in
                       sorted(Counter(col['n_rings'].tolist()).items())},
            acyclic_fraction=float((col['n_rings'] == 0).mean()),
            max_ring_size_histogram={int(k): int(v) for k, v in
                                     sorted(Counter(col['max_ring_size'].tolist()).items())},
            aromatic_rings_per_molecule=_pct(col['n_aromatic_rings']),
            fraction_with_aromatic_ring=float((col['n_aromatic_rings'] > 0).mean()),
        ),
        formal_charges=dict(
            states=dict(extras['fc_states'].most_common()),
            molecules_with_any=int((col['n_fc'] > 0).sum()),
            molecule_fraction=float((col['n_fc'] > 0).mean()),
        ),
        geometry=dict(
            centred_radius=_pct(col['radius']),
            radius_of_gyration=_pct(col['gyration']),
            thinnest_axis_extent=_pct(col['thickness']),
            file_z_extent=_pct(col['z_extent']),
            flat_in_file_z_frame=bool(
                np.median(col['z_extent']) <= 1.05 * np.median(col['thickness']))
            if n else None,
        ),
        planarity=dict(
            all_atoms=dict(
                plane_rmsd=_pct(col['plane_rmsd']),
                planarity_pca=_pct(col['planarity_pca']),
                planarity_rms=_pct(col['planarity_rms']),
                fraction_under_threshold=_threshold_table(col['plane_rmsd'], thresholds),
            ),
            heavy_atoms=dict(
                plane_rmsd=_pct(col['plane_rmsd_heavy']),
                planarity_pca=_pct(col['planarity_pca_heavy']),
                planarity_rms=_pct(col['planarity_rms_heavy']),
                fraction_under_threshold=_threshold_table(col['plane_rmsd_heavy'],
                                                          thresholds),
            ),
            thresholds_angstrom=list(thresholds),
            n_undefined=int((~np.isfinite(col['plane_rmsd'])).sum()),
            definition='SINGROUP/SMARTERS notebooks/utils/planarity.py; '
                       'rmsd = RMS distance from the best-fit plane, in Angstrom',
        ),
        quality=dict(
            covalent_fallback_molecules=int(col['used_fallback'].sum()),
            covalent_fallback_fraction=float(col['used_fallback'].mean()) if n else 0.0,
            sanitize_failures=extras['sanitize_failures'],
            unique_smiles=len(set(smiles)),
            duplicate_smiles_fraction=(1 - len(set(smiles)) / len(smiles)) if smiles else None,
            top_formulas=dict(extras['formulas'].most_common(15)),
        ),
        identity=dict(
            n_records=n,
            n_unique_cids=len(set(cids)),
            records_per_cid=(n / len(set(cids))) if cids else None,
            cid_multiplicity_histogram={int(k): int(v) for k, v in sorted(cid_mult.items())},
            max_records_per_cid=int(max(cid_mult)) if cid_mult else 0,
        ),
    )
    return out


def split_overlap(per_split):
    """CID and canonical-SMILES overlap between splits, with a severity verdict.

    A shared CID is leakage; a shared SMILES under distinct CIDs is reported as structure
    overlap.
    """
    names = list(per_split)
    out = {'pairs': {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca = {c for c in per_split[a]['cids'] if c}
            cb = {c for c in per_split[b]['cids'] if c}
            sa = {s for s in per_split[a]['smiles'] if s}
            sb = {s for s in per_split[b]['smiles'] if s}
            n_cid, n_smi = len(ca & cb), len(sa & sb)
            out['pairs'][f'{a}|{b}'] = dict(
                shared_cids=n_cid,
                shared_cid_fraction_of_smaller=n_cid / max(1, min(len(ca), len(cb))),
                shared_smiles=n_smi,
                shared_smiles_fraction_of_smaller=n_smi / max(1, min(len(sa), len(sb))),
            )
    cid_leaks = [k for k, v in out['pairs'].items() if v['shared_cids'] > 0]
    smiles_leaks = [k for k, v in out['pairs'].items() if v['shared_smiles'] > 0]
    worst = max((v['shared_smiles_fraction_of_smaller']
                 for v in out['pairs'].values()), default=0.0)
    if cid_leaks:
        out['severity'] = 'cid_leakage'
        out['verdict'] = (f'CID leakage: the same dataset entry appears in both splits of '
                          f'{", ".join(cid_leaks)}')
    elif smiles_leaks:
        out['severity'] = 'structure_overlap'
        out['verdict'] = (
            f'CID-disjoint, but {sum(v["shared_smiles"] for v in out["pairs"].values())} '
            f'structures recur across splits under different CIDs '
            f'(worst pair {100 * worst:.2f}% of the smaller split)')
    else:
        out['severity'] = 'ok'
        out['verdict'] = 'disjoint by both CID and structure'
    return out


def cube_stats(h5_path, split=None, n_sample=256, seed=0):
    """Intensity statistics of the image cubes on a random sample of ``n_sample`` molecules.

    Reported on the stored ``uint16`` values and in recovered units,
    ``X.astype(float32) * scale[i]``.
    """
    import h5py

    pp = stm_preprocess()
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, 'r') as f:
        name, g = pp._split_group(f, split)
        X = g[pp.CUBE_KEY]
        n_total, h, w, n_z = X.shape
        scale = np.asarray(g['scale']) if 'scale' in g else None
        z_slices = list(g.attrs.get('z_slices', []))
        attrs = {k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                 for k, v in g.attrs.items()}

        idx = np.sort(rng.choice(n_total, size=min(n_sample, n_total), replace=False))
        per_slice_mean = np.zeros((len(idx), n_z))
        per_slice_std = np.zeros((len(idx), n_z))
        frac_zero = np.zeros(len(idx))
        raw_all = []
        for k, i in enumerate(idx):
            cube = np.asarray(X[i]).astype(np.float32)
            s = float(scale[i]) if scale is not None else 1.0
            rec = cube * s
            per_slice_mean[k] = rec.mean(axis=(0, 1))
            per_slice_std[k] = rec.std(axis=(0, 1))
            frac_zero[k] = float((cube == 0).mean())
            raw_all.append(cube[::8, ::8, :].ravel())
        raw_all = np.concatenate(raw_all)

    env = per_slice_mean.mean(axis=0)
    return dict(
        split=name, n_molecules=int(n_total), n_sampled=int(len(idx)),
        cube_shape=[int(h), int(w), int(n_z)], z_slices=[float(v) for v in z_slices],
        attrs=attrs,
        per_slice_mean=env.tolist(),
        per_slice_mean_std=per_slice_mean.std(axis=0).tolist(),
        per_slice_std=per_slice_std.mean(axis=0).tolist(),
        envelope_monotone_decreasing=bool(np.all(env[:-1] > env[1:])),
        envelope_ratio_first_to_last=float(env[0] / env[-1]) if env[-1] else None,
        raw_uint16=_pct(raw_all),
        fraction_exactly_zero=float(frac_zero.mean()),
        scale=_pct(scale) if scale is not None else None,
        # (cube, slice) pairs with zero spatial variance.
        n_dead_slices=int((per_slice_std < 1e-12).sum()),
    )


# --- figures ---------------------------------------------------------------------------

def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=10, loc='left', fontweight='bold', color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
    return ax


def fig_composition(col, extras, summary, symbols, split, out_path):
    """What the molecules are made of: elements, bonds, rings, charges."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    idt = summary['identity']
    fig.suptitle(f'STM corpus composition - {split} split  '
                 f'({idt["n_records"]:,} records = {idt["n_unique_cids"]:,} distinct '
                 f'molecules x {idt["records_per_cid"]:.2f} images, '
                 f'{summary["n_atoms_total"]:,} atoms)',
                 fontsize=12, fontweight='bold', x=.008, ha='left', color=INK)

    syms = symbols['symbol_list']
    ax = _style(axes[0, 0], 'A - Atoms per element', ylabel='atoms (log)')
    counts = [summary['elements'][s]['atoms'] for s in syms]
    ax.bar(syms, counts, color=ONE, width=.68)
    ax.set_yscale('log')
    for x, c in enumerate(counts):
        ax.text(x, c * 1.35, f'{100 * summary["elements"][syms[x]]["atom_fraction"]:.2g}%',
                ha='center', fontsize=7, color=MUTED)
    ax.set_ylim(top=max(counts) * 6)

    ax = _style(axes[0, 1], 'B - Molecules containing each element',
                ylabel='% of molecules')
    fr = [100 * summary['elements'][s]['molecule_fraction'] for s in syms]
    ax.bar(syms, fr, color=ONE, width=.68)
    for x, v in enumerate(fr):
        ax.text(x, v + max(fr) * .02, f'{v:.3g}', ha='center', fontsize=7, color=MUTED)
    ax.set_ylim(0, max(fr) * 1.16)

    # 'aromatic' is empty because the sidecar is kekulized.
    ax = _style(axes[0, 2], 'C - Bond orders', ylabel='bonds')
    names = list(BOND_ORDER_NAME.values())
    bvals = [summary['bonds']['by_order'][k] for k in names]
    ax.bar(names, bvals, color=ONE, width=.62)
    for x, v in enumerate(bvals):
        pc = summary['bonds']['by_order_fraction'][names[x]] * 100
        ax.text(x, v + max(bvals) * .02, f'{v:,}\n{pc:.1f}%', ha='center', fontsize=7,
                color=MUTED)
    ax.set_ylim(0, max(bvals) * 1.25)
    if bvals[3] == 0:
        ax.text(.98, .82, 'aromatic = 0 is correct:\nthe sidecar is kekulized',
                transform=ax.transAxes, ha='right', fontsize=7.5, style='italic',
                color=MUTED)

    ax = _style(axes[1, 0], 'D - Most common bonds', xlabel='count')
    top = summary['bonds']['top_symbols']
    if top:
        items = list(top.items())[:12][::-1]
        ax.barh([k for k, _ in items], [v for _, v in items], color=ONE, height=.68)
        ax.set_xscale('log')
    else:
        ax.text(.5, .5, 'evaluation.cal_geometry unavailable', transform=ax.transAxes,
                ha='center', color=MUTED, fontsize=9)

    ax = _style(axes[1, 1], 'E - Rings per molecule', xlabel='rings', ylabel='records')
    rh = summary['rings']['histogram']
    ks = sorted(rh)[:12]
    ax.bar(ks, [rh[k] for k in ks], color=ONE, width=.68)
    ax.set_xticks(ks)
    ax.text(.97, .93,
            f'acyclic {100 * summary["rings"]["acyclic_fraction"]:.1f}%\n'
            f'has aromatic ring {100 * summary["rings"]["fraction_with_aromatic_ring"]:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color=MUTED)

    ax = _style(axes[1, 2], 'F - Formal charge states', ylabel='atoms (log)')
    st = summary['formal_charges']['states']
    if st:
        items = list(st.items())[:12]
        vals = [v for _, v in items]
        ax.bar([k for k, _ in items], vals, color=ONE, width=.68)
        ax.set_yscale('log')
        ax.set_ylim(top=max(vals) * 4)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(.5, .5, 'no non-zero formal charges', transform=ax.transAxes,
                ha='center', color=MUTED, fontsize=9)
    ax.text(.97, .93,
            f'{100 * summary["formal_charges"]["molecule_fraction"]:.1f}% of records\n'
            f'carry a charged atom', transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, .955])
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)
    return out_path


def fig_size_shape(col, extras, summary, split, out_path):
    """How big the molecules are and how flat they sit."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(19.5, 8), facecolor='white')
    fig.suptitle(f'STM molecule size and shape - {split} split',
                 fontsize=12, fontweight='bold', x=.008, ha='left', color=INK)

    def hist(ax, data, bins, title, xlabel, note=None):
        _style(ax, title, xlabel, 'records')
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]
        ax.hist(data, bins=bins, color=ONE, edgecolor='white', linewidth=.4)
        med = float(np.median(data))
        ax.axvline(med, color=INK, lw=1.2, ls='--')
        ax.text(med, .52, f'  median {med:.6g}', fontsize=7.5, color=INK, va='center',
                transform=ax.get_xaxis_transform())
        if note:
            ax.text(.97, .93, note, transform=ax.transAxes, ha='right', va='top',
                    fontsize=7.5, color=MUTED)

    a = col['n_atoms']
    hist(axes[0, 0], a, np.arange(a.min() - .5, a.max() + 1.5), 'A - Atoms per molecule',
         'atoms (incl. H)',
         f'max {a.max()} of 54 padded rows\np95 {summary["size"]["atoms"]["p95"]:.0f}')
    h = col['n_heavy']
    hist(axes[0, 1], h, np.arange(h.min() - .5, h.max() + 1.5),
         'B - Heavy atoms per molecule', 'atoms (excl. H)',
         f'H is {100 * summary["size"]["hydrogen_fraction"]:.1f}% of all atoms')
    hist(axes[0, 2], col['mol_weight'], 60, 'C - Molecular weight', 'g/mol')
    hist(axes[0, 3], col['n_bonds'], 60, 'D - Bonds per molecule', 'bonds')
    hist(axes[1, 0], col['radius'], 60, 'E - Centred radius',
         'max |r| from centroid (A)',
         'the quantity data.pos_norm\nhas to normalise')

    ax = _style(axes[1, 1], 'F - Out-of-plane extent', 'A', 'records')
    bins = np.linspace(0, max(1e-6, float(np.percentile(col['z_extent'], 99.5))), 60)
    ax.hist(col['thickness'], bins=bins, color=SPLIT_COLOR['train'], alpha=.72,
            label='thinnest principal axis', edgecolor='white', linewidth=.3)
    ax.hist(col['z_extent'], bins=bins, color=SPLIT_COLOR['val'], alpha=.62,
            label="file's own z axis", edgecolor='white', linewidth=.3)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=MUTED)
    ax.text(.97, .60,
            f'median {np.median(col["thickness"]):.2f} A vs '
            f'{np.median(col["z_extent"]):.2f} A\n'
            + ('the corpus is not stored\nflat in the z frame'
               if not summary['geometry']['flat_in_file_z_frame']
               else 'molecules lie flat in the\nfile z frame'),
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=MUTED)

    pl = summary['planarity']
    ax = _style(axes[1, 2], 'G - Planarity RMSD (SMARTERS)',
                'RMS distance from the best-fit plane (A)', 'records')
    hi = max(1e-6, float(np.nanpercentile(col['plane_rmsd'], 99.5)))
    bins = np.linspace(0, hi, 60)
    ax.hist(col['plane_rmsd'][np.isfinite(col['plane_rmsd'])], bins=bins,
            color=SPLIT_COLOR['train'], alpha=.72, label='all atoms',
            edgecolor='white', linewidth=.3)
    ax.hist(col['plane_rmsd_heavy'][np.isfinite(col['plane_rmsd_heavy'])], bins=bins,
            color=SPLIT_COLOR['val'], alpha=.62, label='heavy atoms only',
            edgecolor='white', linewidth=.3)
    for k, t in enumerate(pl['thresholds_angstrom']):
        if t <= hi:
            ax.axvline(t, color=INK, lw=1.0, ls=':')
            ax.text(t, .97 - .07 * k, f' {t:g}', fontsize=7, color=INK, va='top',
                    transform=ax.get_xaxis_transform())
    ax.legend(fontsize=7.5, frameon=False, labelcolor=MUTED, loc='upper right')
    frac = pl['all_atoms']['fraction_under_threshold']
    frac_h = pl['heavy_atoms']['fraction_under_threshold']
    ax.text(.97, .64,
            'under threshold (all / heavy)\n' + '\n'.join(
                f'  <= {t} A : {100 * frac[t]:.1f}% / {100 * frac_h[t]:.1f}%'
                for t in frac),
            transform=ax.transAxes, ha='right', va='top', fontsize=7, color=MUTED,
            family='monospace')

    ax = _style(axes[1, 3], 'H - Planarity (normalised)', 'percent', 'records')
    bins = np.linspace(
        float(min(np.nanpercentile(col['planarity_rms'], 0.5), 90.0)), 100.0, 60)
    ax.hist(col['planarity_pca'][np.isfinite(col['planarity_pca'])], bins=bins,
            color=SPLIT_COLOR['train'], alpha=.72, label='planarity_pca',
            edgecolor='white', linewidth=.3)
    ax.hist(col['planarity_rms'][np.isfinite(col['planarity_rms'])], bins=bins,
            color=SPLIT_COLOR['val'], alpha=.62, label='planarity_rms',
            edgecolor='white', linewidth=.3)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=MUTED, loc='upper left')
    ax.text(.97, .93,
            f'median {pl["all_atoms"]["planarity_pca"]["p50"]:.1f}% vs '
            f'{pl["all_atoms"]["planarity_rms"]["p50"]:.1f}%\n'
            f'pca saturates; rms discriminates',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=MUTED)

    fig.tight_layout(rect=[0, 0, 1, .955])
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)
    return out_path


def fig_splits(per_split, report, symbols, out_path):
    """Compare the splits' distributions and show their overlap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(per_split)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), facecolor='white')
    fig.suptitle('STM splits compared', fontsize=12, fontweight='bold', x=.008,
                 ha='left', color=INK)

    ax = _style(axes[0, 0], 'A - Atoms per molecule', 'atoms', 'density')
    bins = np.arange(0, 56)
    for s in names:
        a = per_split[s]['col']['n_atoms']
        ax.hist(a, bins=bins, density=True, histtype='step', lw=1.8,
                color=SPLIT_COLOR[s], ls=SPLIT_DASH[s], label=f'{s}  (n={len(a):,})')
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)

    ax = _style(axes[0, 1], 'B - Atom fraction per element', ylabel='% of atoms (log)')
    syms = symbols['symbol_list']
    x = np.arange(len(syms))
    w = .8 / len(names)
    for k, s in enumerate(names):
        vals = [100 * report[s]['elements'][e]['atom_fraction'] for e in syms]
        ax.bar(x + k * w - .4 + w / 2, vals, width=w * .9, color=SPLIT_COLOR[s], label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(syms)
    ax.set_yscale('log')
    ax.legend(fontsize=8, frameon=False, ncol=3, labelcolor=MUTED)

    ax = _style(axes[1, 0], 'C - Molecular weight', 'g/mol', 'density')
    lo = min(per_split[s]['col']['mol_weight'].min() for s in names)
    hi = max(np.percentile(per_split[s]['col']['mol_weight'], 99.5) for s in names)
    for s in names:
        ax.hist(per_split[s]['col']['mol_weight'], bins=np.linspace(lo, hi, 70),
                density=True, histtype='step', lw=1.8, color=SPLIT_COLOR[s],
                ls=SPLIT_DASH[s], label=s)
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)

    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('D - Split overlap', fontsize=10, loc='left', fontweight='bold',
                 color=INK)
    ov = report.get('split_overlap', {})
    lines = []
    for pair, v in ov.get('pairs', {}).items():
        lines.append(f'{pair.replace("|", "  vs  ")}')
        lines.append(f'    shared CIDs    {v["shared_cids"]:>8,}   '
                     f'({100 * v["shared_cid_fraction_of_smaller"]:.3f}% of the smaller)')
        lines.append(f'    shared SMILES  {v["shared_smiles"]:>8,}   '
                     f'({100 * v["shared_smiles_fraction_of_smaller"]:.3f}% of the smaller)')
        lines.append('')
    for s in names:
        q, idt = report[s]['quality'], report[s]['identity']
        lines.append(f'{s}: {idt["n_records"]:,} records = {idt["n_unique_cids"]:,} CIDs '
                     f'x {idt["records_per_cid"]:.2f} images each, '
                     f'{q["unique_smiles"]:,} unique SMILES')
    lines.append('')
    lines.append('a record is an image, not a molecule:')
    lines.append('the same CID recurs with a different orientation.')
    import textwrap
    sev = ov.get('severity', 'ok')
    headline = {
        'ok': 'PASS - no entry and no structure appears in two splits',
        'structure_overlap': 'PASS with a caveat - splits are CID-disjoint',
        'cid_leakage': 'FAIL - the same entry is in two splits; '
                       'held-out metrics include training molecules',
    }.get(sev, 'not computed')
    detail = '\n'.join(textwrap.wrap(ov.get('verdict', ''), 62))
    ax.text(0, .99, headline, transform=ax.transAxes, va='top', fontsize=9.5,
            fontweight='bold', color={'ok': '#008300', 'structure_overlap': '#eda100',
                                      'cid_leakage': '#e34948'}.get(sev, MUTED))
    ax.text(0, .93, detail, transform=ax.transAxes, va='top', fontsize=8,
            color=MUTED)
    ax.text(0, .80, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=8.5,
            family='monospace', color=INK)


    fig.tight_layout(rect=[0, 0, 1, .955])
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)
    return out_path


def fig_cubes(cubes, out_path):
    """Plot the per-slice intensity envelope, raw intensity percentiles and recovery scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(cubes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor='white')
    fig.suptitle('STM image cubes', fontsize=12, fontweight='bold', x=.008, ha='left',
                 color=INK)

    ax = _style(axes[0], 'A - Mean intensity vs tip height', 'z slice (A)',
                'mean recovered intensity')
    for s in names:
        c = cubes[s]
        z = c['z_slices'] or list(range(len(c['per_slice_mean'])))
        m = np.array(c['per_slice_mean'])
        sd = np.array(c['per_slice_mean_std'])
        ax.plot(z, m, SPLIT_DASH[s], color=SPLIT_COLOR[s], lw=2, marker='o', ms=4,
                label=f'{s} (n={c["n_sampled"]})')
        ax.fill_between(z, m - sd, m + sd, color=SPLIT_COLOR[s], alpha=.10, lw=0)
    ax.set_yscale('log')
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc='upper right')
    c0 = cubes[names[0]]
    ax.text(.03, .06,
            f'decays monotonically: {c0["envelope_monotone_decreasing"]}\n'
            f'first/last = {c0["envelope_ratio_first_to_last"]:.3g}x',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color=MUTED)

    ax = _style(axes[1], 'B - Raw uint16 intensity percentiles', 'percentile',
                'stored value')
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    for s in names:
        r = cubes[s]['raw_uint16']
        ax.plot(qs, [r[f'p{q}'] for q in qs], SPLIT_DASH[s], color=SPLIT_COLOR[s], lw=2,
                marker='o', ms=4, label=s)
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    ax.text(.03, .93,
            f'{100 * c0["fraction_exactly_zero"]:.1f}% of pixels are exactly 0\n'
            f'dead slices: {sum(cubes[s]["n_dead_slices"] for s in names)}',
            transform=ax.transAxes, va='top', fontsize=7.5, color=MUTED)

    ax = _style(axes[2], 'C - Per-sample recovery scale', 'scale (log10)', '')
    for k, sp in enumerate(names):
        sc = cubes[sp].get('scale')
        if not sc:
            continue
        lo, mid, hi = (np.log10(sc['p5']), np.log10(sc['p50']), np.log10(sc['p95']))
        y = len(names) - 1 - k
        ax.plot([lo, hi], [y, y], color=SPLIT_COLOR[sp], lw=3, solid_capstyle='round',
                alpha=.55)
        ax.plot([mid], [y], 'o', color=SPLIT_COLOR[sp], ms=7, zorder=3)
        ax.text(hi, y + .18, f'  {sp}  median 1e{mid:.1f}', fontsize=8, color=MUTED,
                va='bottom', ha='left')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([])
    ax.set_ylim(-.6, len(names) - .3)
    ax.text(.02, .06, 'bar = p5..p95, dot = median\n'
                      'X.astype(float32) * scale[i] recovers physical units\n'
                      '(group attribute "recover")', transform=ax.transAxes, ha='left',
            va='bottom', fontsize=7.5, color=MUTED, style='italic')

    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)
    return out_path


# --- CSV output ------------------------------------------------------------------------

# Rotation-invariant columns; checked for agreement across the records of a CID.
_MOL_COLS = ('n_atoms', 'n_heavy', 'n_bonds', 'n_rings', 'n_aromatic_rings',
             'max_ring_size', 'mol_weight', 'radius', 'gyration',
             'plane_rmsd', 'planarity_pca', 'planarity_rms',
             'plane_rmsd_heavy', 'planarity_pca_heavy', 'planarity_rms_heavy',
             'thickness', 'n_fc')

# Orientation-dependent columns, dropped from the per-CID CSV.
_IMAGE_COLS = ('h5_index', 'z_extent')

# Within-CID agreement tolerance; must exceed the rounding in _ROUND.
_AGREE_ATOL = 1e-3

_ROUND = {'mol_weight': 4, 'radius': 4, 'gyration': 4, 'thickness': 4, 'z_extent': 4,
          'plane_rmsd': 5, 'plane_rmsd_heavy': 5,
          'planarity_pca': 4, 'planarity_rms': 4,
          'planarity_pca_heavy': 4, 'planarity_rms_heavy': 4}


def _cell(key, value):
    """One CSV cell: rounded, with non-finite rendered as the empty string."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        return round(float(value), _ROUND.get(key, 6))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return '' if value is None else value


def _rows(col, extras, symbols, split):
    """Yield one CSV row per record."""
    n = len(col['n_atoms'])
    elem = extras['elem_counts']
    order = extras['order_counts']
    syms = symbols['symbol_list']
    for i in range(n):
        row = {'split': split, 'h5_index': int(extras['h5_index'][i]),
               'cid': extras['cids'][i] or ''}
        row.update({k: _cell(k, col[k][i]) for k in
                    ('n_atoms', 'n_heavy', 'n_bonds')})
        for o, name in BOND_ORDER_NAME.items():
            row[f'n_{name}'] = int(order[i, o])
        row.update({k: _cell(k, col[k][i]) for k in
                    ('n_rings', 'n_aromatic_rings', 'max_ring_size', 'mol_weight')})
        row['formula'] = extras['formula_per_row'][i] or ''
        for j, s in enumerate(syms):
            row[f'n_{s}'] = int(elem[i, j])
        row.update({k: _cell(k, col[k][i]) for k in
                    ('radius', 'gyration',
                     'plane_rmsd', 'planarity_pca', 'planarity_rms',
                     'plane_rmsd_heavy', 'planarity_pca_heavy', 'planarity_rms_heavy',
                     'thickness', 'z_extent', 'n_fc')})
        row['used_fallback'] = int(col['used_fallback'][i])
        row['smiles'] = extras['smiles'][i] or ''
        yield row


def write_csvs(col, extras, symbols, split, out_dir):
    """Write ``molecules_<split>.csv`` (one row per image) and ``per_cid_<split>.csv``.

    Use the per-CID file for per-molecule statistics; the per-record file weights each molecule
    by its number of images. The per-CID file keeps the first record's values, the count in
    ``n_records`` and, in ``geometry_spread``, the largest within-CID disagreement on a
    rotation-invariant column.
    """
    import csv

    rows = list(_rows(col, extras, symbols, split))
    paths = {}

    per_record = osp.join(out_dir, f'molecules_{split}.csv')
    with open(per_record, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    paths['per_record'] = per_record

    def _disagrees(a, b):
        """Difference beyond the CSV rounding tolerance: 0.0 within it, inf if non-numeric."""
        if a == b:
            return 0.0
        try:
            d = abs(float(a) - float(b))
        except (TypeError, ValueError):
            return float('inf')
        return d if d > _AGREE_ATOL else 0.0

    by_cid, drift, worst = {}, 0, 0.0
    for r in rows:
        cid = r['cid']
        if cid not in by_cid:
            by_cid[cid] = dict(r, n_records=1, geometry_spread=0.0)
        else:
            first = by_cid[cid]
            first['n_records'] += 1
            spread = max(_disagrees(first[k], r[k]) for k in _MOL_COLS)
            if spread > 0:
                drift += 1
                worst = max(worst, spread)
                first['geometry_spread'] = max(first['geometry_spread'], spread)

    per_cid_rows = []
    for r in by_cid.values():
        out = {k: v for k, v in r.items()
               if k not in _IMAGE_COLS and k not in ('n_records', 'geometry_spread')}
        out['n_records'] = r['n_records']
        out['geometry_spread'] = round(float(r['geometry_spread']), 5) \
            if np.isfinite(r['geometry_spread']) else ''
        per_cid_rows.append(out)

    per_cid = osp.join(out_dir, f'per_cid_{split}.csv')
    with open(per_cid, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cid_rows[0]))
        w.writeheader()
        w.writerows(per_cid_rows)
    paths['per_cid'] = per_cid
    paths['n_records'] = len(rows)
    paths['n_cids'] = len(per_cid_rows)
    paths['n_cid_drift'] = drift
    paths['n_cid_drift_worst'] = worst
    if drift:
        print(f'  WARNING: {drift} records disagree with the first record of their own CID '
              f'on a rotation-invariant column (worst spread {worst:.4g}), so the repeats are '
              f'not pure rotations and per_cid_{split}.csv holds the first record of each CID, '
              f'not values shared by all its records.')
    return paths


def write_summary_csv(report, splits, out_dir):
    """Write ``summary.csv``: one row of headline numbers per split."""
    import csv

    rows = []
    for split in splits:
        s = report.get(split)
        if not s:
            continue
        pa = s['planarity']['all_atoms']
        ph = s['planarity']['heavy_atoms']
        row = {
            'split': split,
            'n_records': s['identity']['n_records'],
            'n_cids': s['identity']['n_unique_cids'],
            'images_per_molecule': round(s['identity']['records_per_cid'], 3),
            'n_atoms_total': s['n_atoms_total'],
            'n_bonds_total': s['n_bonds_total'],
            'atoms_median': s['size']['atoms']['p50'],
            'atoms_p95': s['size']['atoms']['p95'],
            'atoms_max': s['size']['atoms']['p100'],
            'heavy_atoms_median': s['size']['heavy_atoms']['p50'],
            'hydrogen_fraction': round(s['size']['hydrogen_fraction'], 5),
            'mol_weight_median': round(s['molecular_weight']['p50'], 3),
            'rings_median': s['rings']['per_molecule']['p50'],
            'acyclic_fraction': round(s['rings']['acyclic_fraction'], 5),
            'aromatic_bearing_fraction': round(s['rings']['fraction_with_aromatic_ring'], 5),
            'plane_rmsd_median': round(pa['plane_rmsd']['p50'], 5),
            'plane_rmsd_p95': round(pa['plane_rmsd']['p95'], 5),
            'planarity_pca_median': round(pa['planarity_pca']['p50'], 4),
            'planarity_rms_median': round(pa['planarity_rms']['p50'], 4),
            'plane_rmsd_heavy_median': round(ph['plane_rmsd']['p50'], 5),
        }
        for t, v in pa['fraction_under_threshold'].items():
            row[f'frac_rmsd_le_{t}'] = round(v, 5) if v is not None else ''
        for t, v in ph['fraction_under_threshold'].items():
            row[f'frac_rmsd_heavy_le_{t}'] = round(v, 5) if v is not None else ''
        row['covalent_fallback_fraction'] = round(
            s['quality']['covalent_fallback_fraction'], 6)
        row['unique_smiles'] = s['quality']['unique_smiles']
        rows.append(row)

    if not rows:
        return None
    path = osp.join(out_dir, 'summary.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


# --- driver -----------------------------------------------------------------------------

def element_symbols():
    """Return ``{'symbol_list', 'symbol', 'index'}`` from the STM element map."""
    pp = stm_preprocess()
    return dict(symbol_list=list(pp.STM_SYMBOLS),
                symbol={z: pp.STM_SYMBOLS[i] for z, i in pp.STM_TYPES.items()},
                index=dict(pp.STM_TYPES))


def analyze_split(graphs_path, symbols, limit=None, want_smiles=True,
                  thresholds=SMARTERS_THRESHOLDS):
    import torch

    print(f'  loading {graphs_path}', flush=True)
    records = torch.load(graphs_path, weights_only=False)
    if limit:
        records = records[:limit]
    print(f'  {len(records):,} records', flush=True)

    col, extras = molecule_table(records, symbols, want_smiles=want_smiles)
    summary = summarize(col, extras, symbols, thresholds=thresholds)

    stats_path = graphs_path + '.stats.json'
    if osp.exists(stats_path):
        with open(stats_path) as fh:
            summary['preprocess_stats'] = json.load(fh)
    fail_path = graphs_path + '.failures.json'
    if osp.exists(fail_path):
        with open(fail_path) as fh:
            fails = json.load(fh)
        summary['preprocess_failures'] = dict(
            n_recorded=len(fails),
            by_type=dict(Counter(f['error'].split(':')[0] for f in fails).most_common()))
    return col, extras, summary


def print_summary(name, s):
    e = s['elements']
    print(f'\n  {name}: {s["n_molecules"]:,} records  {s["n_atoms_total"]:,} atoms  '
          f'{s["n_bonds_total"]:,} bonds')
    print(f'    atoms/molecule   median {s["size"]["atoms"]["p50"]:.0f}  '
          f'p95 {s["size"]["atoms"]["p95"]:.0f}  max {s["size"]["atoms"]["p100"]:.0f}')
    print(f'    heavy atoms      median {s["size"]["heavy_atoms"]["p50"]:.0f}  '
          f'H is {100 * s["size"]["hydrogen_fraction"]:.1f}% of all atoms')
    print(f'    weight           median {s["molecular_weight"]["p50"]:.0f} g/mol')
    print(f'    rings            median {s["rings"]["per_molecule"]["p50"]:.0f}  '
          f'acyclic {100 * s["rings"]["acyclic_fraction"]:.1f}%  '
          f'aromatic-bearing {100 * s["rings"]["fraction_with_aromatic_ring"]:.1f}%')
    print(f'    elements         ' + '  '.join(
        f'{k} {100 * v["atom_fraction"]:.2f}%' for k, v in e.items()))
    print(f'    quality          fallback {100 * s["quality"]["covalent_fallback_fraction"]:.2f}%  '
          f'unique SMILES {s["quality"]["unique_smiles"]:,}')
    pa = s['planarity']['all_atoms']
    ph = s['planarity']['heavy_atoms']
    fr = pa['fraction_under_threshold']
    frh = ph['fraction_under_threshold']
    print(f'    planarity        RMSD median {pa["plane_rmsd"]["p50"]:.3f} A all-atom / '
          f'{ph["plane_rmsd"]["p50"]:.3f} A heavy   '
          f'planarity_rms {pa["planarity_rms"]["p50"]:.1f}%')
    print(f'    under threshold  ' + '   '.join(
        f'<={t}A {100 * fr[t]:.1f}%/{100 * frh[t]:.1f}%' for t in fr)
        + '   (all/heavy)')
    i = s['identity']
    print(f'    identity         {i["n_records"]:,} records but only {i["n_unique_cids"]:,} '
          f'distinct CIDs ({i["records_per_cid"]:.2f} images per molecule, '
          f'max {i["max_records_per_cid"]})')


def _benzene(plane='xy'):
    """Planar benzene as ``(z, pos)``, in the xy plane or stood up in the xz plane."""
    ang = np.arange(6) * np.pi / 3
    rc, rh = 1.39, 2.48
    xs = np.concatenate([rc * np.cos(ang), rh * np.cos(ang)])
    ys = np.concatenate([rc * np.sin(ang), rh * np.sin(ang)])
    zs = np.zeros(12)
    z = np.array([6] * 6 + [1] * 6)
    pos = np.stack([xs, ys, zs] if plane == 'xy' else [xs, zs, ys], axis=1)
    return z, pos


def self_test():
    """Check the measurements, CSVs and figures on planted molecules; needs no data."""
    import tempfile

    ok = failed = 0

    def check(name, cond, detail=''):
        nonlocal ok, failed
        print(f'  {"PASS" if cond else "FAIL"}  {name}{"  " + detail if detail else ""}')
        if cond:
            ok += 1
        else:
            failed += 1

    pp = stm_preprocess()
    symbols = element_symbols()
    check('element map is the ten-element STM map',
          symbols['symbol_list'] == ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br'],
          str(symbols['symbol_list']))

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    methane = (np.array([6, 1, 1, 1, 1]),
               np.array([(0, 0, 0), (1.09, 0, 0), (-.36, 1.03, 0), (-.36, -.51, .89),
                         (-.36, -.51, -.89)], dtype=float))
    chloromethane = (np.array([6, 17, 1, 1, 1]),
                     np.array([(0, 0, 0), (1.78, 0, 0), (-.37, 1.03, 0),
                               (-.37, -.51, .89), (-.37, -.51, -.89)], dtype=float))

    def record(z, pos, cid):
        mol, fallback, _ = pp.mol_from_geometry(z, pos, allow_fallback=True, charge=0)
        rec = pp.encode_graph(mol, h5_index=0, types=pp.STM_TYPES)
        rec['used_fallback'] = fallback
        rec['cid'] = cid
        return rec

    recs = [record(*methane, 'M1'),
            record(*chloromethane, 'C1'),
            record(*_benzene('xy'), 'B1'),
            record(*_benzene('xz'), 'B2')]

    col, extras = molecule_table(recs, symbols, want_smiles=True, progress_every=0)

    check('methane: 5 atoms, 1 heavy, 4 bonds, 0 rings',
          (col['n_atoms'][0], col['n_heavy'][0], col['n_bonds'][0], col['n_rings'][0])
          == (5, 1, 4, 0),
          f'{col["n_atoms"][0]}/{col["n_heavy"][0]}/{col["n_bonds"][0]}/{col["n_rings"][0]}')
    check('methane: 4 single bonds and nothing else',
          extras['order_counts'][0].tolist() == [0, 4, 0, 0, 0],
          str(extras['order_counts'][0].tolist()))
    check('chloromethane: Cl counted in the element column the TERS map lacks',
          extras['elem_counts'][1, symbols['index'][17]] == 1)
    check('benzene: one ring of size 6', (col['n_rings'][2], col['max_ring_size'][2]) == (1, 6),
          f'{col["n_rings"][2]} rings, max size {col["max_ring_size"][2]}')
    check('benzene: kekulized in the sidecar, so both single and double bonds are stored',
          extras['order_counts'][2][1] > 0 and extras['order_counts'][2][2] == 3,
          str(extras['order_counts'][2].tolist()))
    check('benzene: aromaticity recovered by sanitizing a copy (1 aromatic ring)',
          col['n_aromatic_rings'][2] == 1, str(col['n_aromatic_rings'][2]))
    check('benzene in xy is flat by both measures',
          col['plane_rmsd'][2] < 1e-6 and col['z_extent'][2] < 1e-6,
          f'plane_rmsd {col["plane_rmsd"][2]:.2e}  z_extent {col["z_extent"][2]:.2e}')
    check('benzene stood on edge: still flat, but not in the file z frame',
          col['plane_rmsd'][3] < 1e-6 and col['z_extent'][3] > 2.5,
          f'plane_rmsd {col["plane_rmsd"][3]:.2e}  z_extent {col["z_extent"][3]:.2f}')
    check('SMARTERS percentages reach 100% on planar benzene',
          abs(col['planarity_pca'][2] - 100) < 1e-9
          and abs(col['planarity_rms'][2] - 100) < 1e-9,
          f'pca {col["planarity_pca"][2]:.4f}%  rms {col["planarity_rms"][2]:.4f}%')
    check('methane is measured as non-planar',
          col['plane_rmsd'][0] > 0.1 and col['planarity_rms'][0] < 90,
          f'rmsd {col["plane_rmsd"][0]:.3f} A  rms {col["planarity_rms"][0]:.1f}%')
    # Methane has one heavy atom, so its heavy-atom planarity is NaN.
    check('heavy-atom planarity is computed alongside the all-atom one',
          abs(col['plane_rmsd_heavy'][2]) < 1e-6
          and np.isnan(col['plane_rmsd_heavy'][0]),
          f'benzene heavy {col["plane_rmsd_heavy"][2]:.2e}, '
          f'methane heavy {col["plane_rmsd_heavy"][0]}')
    from rdkit import Chem
    check('the two benzenes are the same molecule to SMILES, with explicit H kept',
          (extras['smiles'][2] == extras['smiles'][3]
           and '[H]' in extras['smiles'][2]
           and Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(extras['smiles'][2])))
           == 'c1ccccc1'),
          extras['smiles'][2])

    s = summarize(col, extras, symbols)
    check('summary totals match the table',
          s['n_molecules'] == 4
          and s['n_atoms_total'] == int(col['n_atoms'].sum())
          and s['n_bonds_total'] == int(col['n_bonds'].sum()),
          f'{s["n_atoms_total"]} atoms, {s["n_bonds_total"]} bonds')
    check('per-element molecule_fraction counts molecules, not atoms',
          s['elements']['Cl']['atoms'] == 1
          and abs(s['elements']['Cl']['molecule_fraction'] - .25) < 1e-9)
    check('no bond is reported as aromatic (the sidecar is kekulized)',
          s['bonds']['by_order']['aromatic'] == 0)
    check('bond symbols come from evaluation/cal_geometry.py, not from a local format',
          s['bonds']['top_symbols'] is not None
          and s['bonds']['top_symbols'].get('H1C', 0) == 19,   # 4 + 3 + 6 + 6
          str(dict(list((s['bonds']['top_symbols'] or {}).items())[:4])))
    check('duplicate SMILES are detected (2 benzenes -> 3 unique of 4)',
          s['quality']['unique_smiles'] == 3, str(s['quality']['unique_smiles']))
    check('identity block separates records from distinct CIDs',
          (s['identity']['n_records'], s['identity']['n_unique_cids']) == (4, 4)
          and s['identity']['max_records_per_cid'] == 1,
          f'{s["identity"]["n_records"]} records / '
          f'{s["identity"]["n_unique_cids"]} CIDs')

    # CID leakage must outrank structure overlap.
    ov = split_overlap({
        'a': dict(cids=['M1', 'C1'], smiles=['C', 'CCl']),
        'b': dict(cids=['B1'], smiles=['c1ccccc1']),
        'c': dict(cids=['C1'], smiles=['CCl']),
    })
    check('a shared CID is graded as CID leakage',
          ov['pairs']['a|c']['shared_cids'] == 1 and ov['severity'] == 'cid_leakage',
          ov['verdict'])
    check('overlap check reports no shared structure for the disjoint pair',
          ov['pairs']['a|b']['shared_smiles'] == 0)
    ov2 = split_overlap({
        'a': dict(cids=['1'], smiles=['CCl']),
        'b': dict(cids=['2'], smiles=['CCl']),
    })
    check('same structure under different CIDs is a caveat, not CID leakage',
          ov2['severity'] == 'structure_overlap'
          and ov2['pairs']['a|b']['shared_cids'] == 0, ov2['verdict'])
    ov3 = split_overlap({'a': dict(cids=['1'], smiles=['C']),
                         'b': dict(cids=['2'], smiles=['CCl'])})
    check('fully disjoint splits are graded ok', ov3['severity'] == 'ok', ov3['verdict'])

    check('planarity summary carries the SMARTERS thresholds',
          s['planarity']['thresholds_angstrom'] == list(SMARTERS_THRESHOLDS)
          and set(s['planarity']['all_atoms']['fraction_under_threshold']) ==
              {f'{t:g}' for t in SMARTERS_THRESHOLDS},
          str(s['planarity']['all_atoms']['fraction_under_threshold']))
    check('NaN planarity stays out of the percentiles',
          s['planarity']['heavy_atoms']['plane_rmsd']['n'] == 2,   # only the 2 benzenes
          f'n={s["planarity"]["heavy_atoms"]["plane_rmsd"]["n"]} of 4 records')

    with tempfile.TemporaryDirectory() as td:
        # --- CSV round trip -------------------------------------------------------
        import csv as _csv
        paths = write_csvs(col, extras, symbols, 'selftest', td)
        with open(paths['per_record']) as fh:
            got = list(_csv.DictReader(fh))
        check('per-record CSV has one row per record',
              len(got) == 4 and paths['n_records'] == 4, f'{len(got)} rows')
        check('per-record CSV carries the planarity columns',
              all(c in got[0] for c in ('plane_rmsd', 'planarity_pca', 'planarity_rms',
                                        'plane_rmsd_heavy', 'cid', 'smiles', 'formula')),
              ', '.join(list(got[0])[:6]) + ' ...')
        check('CSV values round-trip',
              int(got[0]['n_atoms']) == 5 and got[2]['formula'] == 'C6H6',
              f'row0 n_atoms={got[0]["n_atoms"]}  row2 formula={got[2]["formula"]}')
        check('undefined planarity is written as an empty cell, not the text "nan"',
              got[0]['plane_rmsd_heavy'] == '',
              repr(got[0]['plane_rmsd_heavy']))
        with open(paths['per_cid']) as fh:
            got_cid = list(_csv.DictReader(fh))
        check('per-CID CSV deduplicates and keeps the multiplicity',
              len(got_cid) == 4 and 'n_records' in got_cid[0]
              and 'h5_index' not in got_cid[0], f'{len(got_cid)} rows')
        check('per-CID CSV drops orientation-dependent columns',
              'z_extent' not in got_cid[0] and 'z_extent' in got[0]
              and 'thickness' in got_cid[0],
              f'per-record has z_extent: {"z_extent" in got[0]}; '
              f'per-CID has it: {"z_extent" in got_cid[0]}')

        p1 = fig_composition(col, extras, s, symbols, 'selftest',
                             osp.join(td, 'composition.png'))
        p2 = fig_size_shape(col, extras, s, 'selftest', osp.join(td, 'size_shape.png'))
        check('composition figure renders', osp.getsize(p1) > 10_000,
              f'{osp.getsize(p1) / 1024:.0f} KB')
        check('size/shape figure renders', osp.getsize(p2) > 10_000,
              f'{osp.getsize(p2) / 1024:.0f} KB')

        try:
            import h5py
        except ImportError:
            print('  SKIP  cube statistics (h5py not installed)')
        else:
            # Synthetic HDF5 with a planted monotone z-decay envelope.
            n_z = 6
            env = np.exp(-np.arange(n_z) / 1.5)
            rng = np.random.default_rng(0)
            cube = (rng.random((4, 16, 16, n_z)) * env * 30000).astype(np.uint16)
            h5 = osp.join(td, 'synthetic.h5')
            with h5py.File(h5, 'w') as f:
                g = f.create_group('val')
                g.attrs['z_slices'] = np.arange(6, 6 + n_z)
                g.create_dataset('X', data=cube)
                g.create_dataset('scale', data=np.ones(4, np.float32))
                g.create_dataset('xyz', data=np.zeros((4, 54, 5), np.float32))
            c = cube_stats(h5, n_sample=4)
            check('cube_stats finds the split group without being told',
                  c['split'] == 'val' and c['cube_shape'] == [16, 16, n_z],
                  f'{c["split"]} {c["cube_shape"]}')
            check('planted z-decay envelope is detected as monotone',
                  c['envelope_monotone_decreasing'] is True,
                  f'ratio {c["envelope_ratio_first_to_last"]:.2f}x')
            check('no dead slices in a live synthetic cube', c['n_dead_slices'] == 0)
            p3 = fig_cubes({'val': c}, osp.join(td, 'cubes.png'))
            check('cube figure renders', osp.getsize(p3) > 10_000,
                  f'{osp.getsize(p3) / 1024:.0f} KB')

    return ok, failed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true', help='no data required')
    ap.add_argument('--graphs-dir', default=None, help='holds graphs_{split}.pt (default: DATA_ROOT/stm-graphs)')
    ap.add_argument('--data-dir', default=None, help='holds {split}.h5 (default: DATA_ROOT/stm-data)')
    ap.add_argument('--out-dir', default=None,
                    help='default: $OUTPUT_ROOT/stm-eval/analysis')
    ap.add_argument('--splits', nargs='+', default=list(SPLITS))
    ap.add_argument('--limit', type=int, default=None,
                    help='first N molecules per split (smoke runs)')
    ap.add_argument('--cube-samples', type=int, default=256,
                    help='cubes read per split for the intensity statistics')
    ap.add_argument('--no-cubes', action='store_true', help='skip the HDF5 pass entirely')
    ap.add_argument('--no-smiles', action='store_true',
                    help='skip canonical SMILES (drops the duplicate/leakage checks)')
    ap.add_argument('--no-csv', action='store_true',
                    help='skip the per-record / per-CID / summary CSVs')
    ap.add_argument('--planarity-thresholds', type=float, nargs='+',
                    default=list(SMARTERS_THRESHOLDS),
                    help=f'RMSD cut-offs to report pass rates against, in Angstrom '
                         f'(default: the SMARTERS thresholds {list(SMARTERS_THRESHOLDS)})')
    a = ap.parse_args()

    if a.self_test:
        ok, failed = self_test()
        print(f'\nself-test: {ok} passed, {failed} failed')
        raise SystemExit(1 if failed else 0)

    a.graphs_dir = a.graphs_dir or data_root('stm-graphs')
    a.data_dir = a.data_dir or data_root('stm-data')
    out_dir = a.out_dir or output_root('stm-eval', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    symbols = element_symbols()
    report, per_split, figures, csvs = {}, {}, [], []

    for split in a.splits:
        gp = osp.join(a.graphs_dir, f'graphs_{split}.pt')
        if not osp.exists(gp):
            print(f'skipping {split}: no {gp}')
            continue
        print(f'\n=== {split} ===', flush=True)
        col, extras, summary = analyze_split(gp, symbols, limit=a.limit,
                                             want_smiles=not a.no_smiles,
                                             thresholds=a.planarity_thresholds)
        report[split] = summary
        per_split[split] = dict(col=col, cids=extras['cids'], smiles=extras['smiles'])
        print_summary(split, summary)

        if not a.no_csv:
            paths = write_csvs(col, extras, symbols, split, out_dir)
            summary['csv'] = {k: v for k, v in paths.items() if k.startswith('n_')}
            csvs.extend([paths['per_record'], paths['per_cid']])
            print(f'    csv              {paths["n_records"]:,} rows -> '
                  f'{osp.basename(paths["per_record"])}   '
                  f'{paths["n_cids"]:,} rows -> {osp.basename(paths["per_cid"])}')

        figures.append(fig_composition(col, extras, summary, symbols, split,
                                       osp.join(out_dir, f'composition_{split}.png')))
        figures.append(fig_size_shape(col, extras, summary, split,
                                      osp.join(out_dir, f'size_shape_{split}.png')))

    if len(per_split) > 1:
        report['split_overlap'] = split_overlap(per_split)
        print(f'\nsplit overlap: {report["split_overlap"]["verdict"]}')
        figures.append(fig_splits(per_split, report, symbols,
                                  osp.join(out_dir, 'splits_comparison.png')))

    if not a.no_cubes:
        cubes = {}
        for split in a.splits:
            h5 = osp.join(a.data_dir, f'{split}.h5')
            if not osp.exists(h5):
                print(f'skipping cubes for {split}: no {h5}')
                continue
            print(f'\nreading {a.cube_samples} cubes from {h5}', flush=True)
            cubes[split] = cube_stats(h5, split=None, n_sample=a.cube_samples)
            c = cubes[split]
            print(f'  {c["cube_shape"]}  z={c["z_slices"]}  '
                  f'monotone={c["envelope_monotone_decreasing"]}  '
                  f'dead slices={c["n_dead_slices"]}')
        if cubes:
            report['cubes'] = cubes
            figures.append(fig_cubes(cubes, osp.join(out_dir, 'cubes.png')))

    report['_meta'] = dict(graphs_dir=a.graphs_dir, data_dir=a.data_dir,
                           limit=a.limit, splits=a.splits,
                           smiles=not a.no_smiles)
    if not a.no_csv:
        sp = write_summary_csv(report, a.splits, out_dir)
        if sp:
            csvs.append(sp)

    json_path = osp.join(out_dir, 'analysis.json')
    with open(json_path, 'w') as fh:
        json.dump(report, fh, indent=2)

    print(f'\nwrote {json_path}')
    for f in csvs:
        print(f'      {f}  ({os.path.getsize(f) / 1024:.0f} KB)')
    for f in figures:
        print(f'      {f}  ({os.path.getsize(f) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
