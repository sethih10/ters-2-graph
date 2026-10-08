"""Plot generated geometries next to the conditioning cube and the target molecule.

The 2D views project each molecule's coordinates onto its own best-fit plane; the 3D views
show the predicted coordinates directly.
"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.analysis.planarity import plane_project
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from datasets.datasets_config import get_dataset_info

# Element colours, radii and symbols; set by set_palette from the dataset info.
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


CUBE_LABEL = 'TERS cube'          # Overridden by --info-name.

set_palette('ters_hcno')
BOND_LW = {1: 1.6, 2: 2.8, 3: 3.8, 4: 2.2}


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
    # Rows follow summary.pkl's ranking; index geo by sc['mol'], not by rank.
    for rank, sc in enumerate(scores):
        mi = sc['mol']
        # --- column 0: the conditioning cube ---------------------------------------------
        ax = fig.add_subplot(nm, ncol, rank * ncol + 1)
        ax.imshow(geo['cube_sum'][mi], cmap='inferno', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#BBBBBB')
        ax.set_ylabel(f"#{sc['ds_index']}", fontsize=9, color='#333')
        if rank == 0:
            # Absent from older geometry payloads.
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
        # --- columns 2..: generated samples -----------------------------------------------
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
    ap.add_argument('--dir', required=True, help='a <workdir>/generated directory')
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    ap.add_argument('--info-name', default='ters_hcno',
                    help='datasets_config entry supplying the element palette '
                         "(use 'stm_10elem' for the STM corpus)")
    args = ap.parse_args()
    set_palette(args.info_name)
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
