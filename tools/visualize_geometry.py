"""Plot the model's ACTUAL predicted coordinates against the TERS cube that conditioned them.

Layout, one row per held-out spectrum, rows ranked best -> worst:

    [ TERS cube ] [ ground truth ] [ sample 0 ] ... [ sample 4 ]

REUSE : ters_hcno['colors_dic'] / ['radius_dic'] / ['atom_decoder']
        (datasets/datasets_config.py:44-59). Those three keys exist in the dataset_info for
        exactly this purpose -- EDM-lineage repos read them to render molecules -- and this is
        the first thing in this project to actually use them.
NEW   : the plotting itself. Nothing in ters-2-graph or ters-ae draws molecules from
        coordinates; visualize.py only calls RDKit's depiction engine, which THROWS AWAY the
        predicted geometry and lays atoms out afresh with Compute2DCoords. Everything drawn
        here is the network's own output.

The 2D panels are not a redrawing either: they are the real 3D coordinates projected onto
each molecule's own best-fit plane (PCA). The training set is planar_hdf5_0.5, so ground
truth is genuinely flat; the out-of-plane RMS printed on every 3D panel says whether the
model reproduced that.
"""
import argparse, os, pickle

# --- ADDED (stm): output paths resolve to /scratch, never inside the repo ($HOME is a
# 15 GB quota and one run is 3.7 GB). Both import forms, because these are run BOTH as
# `python3 tools/x.py` (sibling import) and as `python3 -m tools.x` (package import).
try:
    from paths import generated
except ImportError:
    from tools.paths import generated
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from datasets.datasets_config import get_dataset_info

# --- ADDED (stm): element tables come from the dataset info, not a fixed 4-element list ---
# These were `ters_hcno['colors_dic']` etc., which is 4 entries (H/C/N/O). The STM corpus has
# 10, so `COLORS[a]` raised IndexError on the first Cl/Br atom -- a crash, thankfully, rather
# than a silently mis-coloured plot. Set by --info-name so the same script renders either
# corpus; the default keeps TERS behaviour byte-for-byte.
COLORS, RADII, DECODER = None, None, None


def set_palette(info_name):
    global COLORS, RADII, DECODER
    info = get_dataset_info(info_name)
    COLORS = info['colors_dic']
    RADII = info['radius_dic']
    DECODER = info['atom_decoder']
    if not (len(COLORS) == len(RADII) == len(DECODER)):
        raise ValueError(
            f'{info_name}: colors_dic/radius_dic/atom_decoder disagree '
            f'({len(COLORS)}/{len(RADII)}/{len(DECODER)}) -- a plot would mislabel elements')


CUBE_LABEL = 'TERS cube'          # --- ADDED (stm): overridden by --info-name below

set_palette('ters_hcno')          # import-time default, unchanged from before
BOND_LW = {1: 1.6, 2: 2.8, 3: 3.8, 4: 2.2}


def plane_project(pos):
    """Project onto the molecule's own best-fit plane; also return out-of-plane RMS."""
    c = pos - pos.mean(0)
    if len(c) < 3:
        return c[:, :2], 0.0
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ vt[:2].T
    oop = c @ vt[2]
    return xy, float(np.sqrt((oop ** 2).mean()))


def draw_2d(ax, pos, atom_type, edges):
    xy, oop = plane_project(np.asarray(pos, dtype=float))
    at = np.asarray(atom_type).astype(int)
    n = len(at)
    for i in range(n):
        for j in range(i + 1, n):
            o = int(edges[i, j])
            if o > 0:
                ax.plot(xy[[i, j], 0], xy[[i, j], 1], '-', color='#444444',
                        lw=BOND_LW.get(o, 1.6), zorder=1, solid_capstyle='round')
    heavy = at > 0
    ax.scatter(xy[heavy, 0], xy[heavy, 1], s=[RADII[a] * 190 for a in at[heavy]],
               c=[COLORS[a] for a in at[heavy]], edgecolors='#222222', linewidths=.7, zorder=3)
    if (~heavy).any():
        ax.scatter(xy[~heavy, 0], xy[~heavy, 1], s=28, c='#DDDDDD',
                   edgecolors='#999999', linewidths=.4, zorder=2)
    ax.set_aspect('equal'); ax.axis('off')
    return oop


def draw_3d(ax, pos, atom_type, edges):
    p = np.asarray(pos, dtype=float); p = p - p.mean(0)
    at = np.asarray(atom_type).astype(int)
    n = len(at)
    for i in range(n):
        for j in range(i + 1, n):
            o = int(edges[i, j])
            if o > 0:
                ax.plot(p[[i, j], 0], p[[i, j], 1], p[[i, j], 2], '-',
                        color='#555555', lw=BOND_LW.get(o, 1.6) * .8)
    heavy = at > 0
    ax.scatter(p[heavy, 0], p[heavy, 1], p[heavy, 2],
               s=[RADII[a] * 130 for a in at[heavy]], c=[COLORS[a] for a in at[heavy]],
               edgecolors='#222222', linewidths=.5, depthshade=True)
    if (~heavy).any():
        ax.scatter(p[~heavy, 0], p[~heavy, 1], p[~heavy, 2], s=14, c='#DDDDDD',
                   edgecolors='none', depthshade=True)
    r = np.abs(p).max() * 1.05 + 1e-6
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1)); ax.grid(False); ax.set_axis_off()
    ax.view_init(elev=22, azim=35)
    _, oop = plane_project(p)
    return oop


def figure(geo, scores, split, mode, path):
    nm, ns = geo['n_mols'], geo['n_samples']
    ncol = 2 + ns
    fig = plt.figure(figsize=(2.35 * ncol, 2.5 * nm), facecolor='white')
    # Drive the layout off the SCORE records, never off geo's own ordering: the geometry
    # pickle was ranked by the explicit-hydrogen Tanimoto that sample_ters.py computed, and
    # summary.pkl was re-ranked on heavy atoms. Indexing one by the other's rank silently
    # paired each molecule with a different molecule's scores.
    for rank, sc in enumerate(scores):
        mi = sc['mol']                    # molecule index, from the same record as the scores
        # --- column 0: the TERS cube that conditioned this row ---------------------------
        ax = fig.add_subplot(nm, ncol, rank * ncol + 1)
        ax.imshow(geo['cube_sum'][mi], cmap='inferno', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#BBBBBB')
        ax.set_ylabel(f"#{sc['ds_index']}", fontsize=9, color='#333')
        if rank == 0:
            # --- ADDED (stm): name the modality and the REAL channel count. Was hardcoded
            # 'TERS cube / sum over 100 ch', which on an STM z-stack mislabels both the
            # modality and the depth (14 slices, not 100 channels) -- a caption that is
            # confidently wrong is worse than none on a figure that gets pasted into a note.
            # cube_sum is already summed over channels, so the count only exists if the
            # sampler recorded it; older payloads have no 'n_channels' key. Omit rather
            # than guess -- an unlabelled panel beats a wrong channel count.
            n_ch = geo.get('n_channels')
            ax.set_title(f'{CUBE_LABEL}\n(sum over {n_ch} ch)' if n_ch else CUBE_LABEL,
                         fontsize=9, color='#111')
        # --- column 1: ground truth ------------------------------------------------------
        t = geo['truth'][mi]
        proj = '3d' if mode == '3d' else None
        ax = fig.add_subplot(nm, ncol, rank * ncol + 2, projection=proj)
        oop = (draw_3d if mode == '3d' else draw_2d)(ax, t['pos'], t['atom_type'], t['edges'])
        ax.set_title(('ground truth' if rank == 0 else '') +
                     f"\n{len(t['atom_type'])} atoms · flat {oop:.2f} Å", fontsize=8, color='#111')
        # --- columns 2..: the five samples ------------------------------------------------
        for si in range(ns):
            g = geo['gen'][mi * ns + si]
            ax = fig.add_subplot(nm, ncol, rank * ncol + 3 + si, projection=proj)
            oop = (draw_3d if mode == '3d' else draw_2d)(ax, g['pos'], g['atom_type'], g['edges'])
            s = sc['samples'][si]
            tag = '  EXACT' if s['exact'] else ('  formula' if s['formula_match'] else '')
            colr = '#1B7F4B' if s['exact'] else ('#8A6A1F' if s['formula_match'] else '#333333')
            ax.set_title((f'sample {si}' if rank == 0 else '') +
                         f"\nTan {s['tanimoto']:.2f}{tag} · flat {oop:.2f} Å",
                         fontsize=8, color=colr)
    handles = [plt.Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=COLORS[i],
                          markeredgecolor='#222', label=DECODER[i]) for i in range(len(DECODER))]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f'{split.upper()} — {CUBE_LABEL} → predicted geometry ({mode.upper()}), '
                 f'rows ranked best to worst', fontsize=12, y=.995)
    fig.tight_layout(rect=[0, .025, 1, .975])
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=generated('ters-runs/ters_20260828_925597'))
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    ap.add_argument('--info-name', default='ters_hcno',
                    help='datasets_config entry supplying the element palette '
                         "(use 'stm_10elem' for the STM corpus)")
    args = ap.parse_args()
    set_palette(args.info_name)          # --- ADDED (stm)
    global CUBE_LABEL
    CUBE_LABEL = 'STM z-stack' if args.info_name == 'stm_10elem' else 'TERS cube'
    figdir = os.path.join(args.dir, 'figures'); os.makedirs(figdir, exist_ok=True)
    with open(os.path.join(args.dir, 'summary.pkl'), 'rb') as f:
        summary = pickle.load(f)
    for split in args.splits:
        with open(os.path.join(args.dir, f'geometry_{split}.pkl'), 'rb') as f:
            geo = pickle.load(f)
        for mode in ('2d', '3d'):
            p = os.path.join(figdir, f'{split}_cube_truth_samples_{mode}.png')
            figure(geo, summary[split], split, mode, p)
            print(f'{split} {mode}: {p}  ({os.path.getsize(p)/1024:.0f} KB)')


if __name__ == '__main__':
    main()
