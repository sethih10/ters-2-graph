"""The DiffSpectra-style results figure for TERS, with the null control the paper has no need of.

The paper's headline is a top-k accuracy curve (top-1 40.76% -> top-10 99.49% on QM9S). That
number is only interpretable because QM9S conditioning is a real spectrum of the target. Here the
model is additionally handed the true atom count and the chemical space is one narrow family, so a
matched-only curve would flatter the model. Every panel therefore carries the shuffled-cube null:
same targets, same atom counts, another molecule's cube.

REUSE : tools/paper_metrics.py for the scoring (upstream compute_metrics.py definitions).
NEW   : the figure.
"""
import argparse, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.paper_metrics import pair_metrics, SIMS

MATCH_C, NULL_C = '#1B5E9B', '#B03030'


def cell_curves(path, ks):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    nm, ns = d['n_mols'], d['n_samples']
    per_mol = [[pair_metrics(d['truth'][mi], d['generated'][mi * ns + si], with_mces=False)
                for si in range(ns)] for mi in range(nm)]
    topk = [sum(any(c['exact'] for c in cs[:k]) for cs in per_mol) / nm for k in ks]
    best = {m: np.array([max(c[m] for c in cs) for cs in per_mol]) for m in SIMS}
    return topk, best, nm, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matched', required=True)
    ap.add_argument('--null', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ks = list(range(1, 11))
    tk_m, best_m, nm, ns = cell_curves(args.matched, ks)
    tk_n, best_n, _, _ = cell_curves(args.null, ks)

    fig = plt.figure(figsize=(13.5, 4.4), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.05], wspace=.32)

    # --- A: top-k accuracy, the paper's headline plot ---------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ks, tk_m, 'o-', color=MATCH_C, lw=2, ms=5, label='matched cube', zorder=3)
    ax.plot(ks, tk_n, 's--', color=NULL_C, lw=2, ms=4, label='shuffled cube (null)', zorder=3)
    ax.fill_between(ks, tk_n, tk_m, color=MATCH_C, alpha=.10, zorder=1)
    ax.set_xlabel('k (candidates per spectrum)'); ax.set_ylabel('top-k accuracy (InChIKey)')
    ax.set_title('A · Top-k accuracy', fontsize=10, loc='left', fontweight='bold')
    ax.set_xticks(ks); ax.set_ylim(-.02, max(.55, max(tk_m) * 1.35))
    ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=8, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.annotate(f'DiffSpectra on QM9S: 0.9949 @ k=10', xy=(.5, .93), xycoords='axes fraction',
                fontsize=7.5, color='#666', ha='center', style='italic')

    # --- B: best-of-k similarity, matched vs null ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(SIMS))[::-1]
    mm = [best_m[m].mean() for m in SIMS]
    nn = [best_n[m].mean() for m in SIMS]
    ax.barh(y + .19, mm, .36, color=MATCH_C, label='matched')
    ax.barh(y - .19, nn, .36, color=NULL_C, label='null')
    for yi, (a, b) in enumerate(zip(mm, nn)):
        ax.text(a + .012, y[yi] + .19, f'{a:.2f}', va='center', fontsize=7.5, color=MATCH_C)
        ax.text(b + .012, y[yi] - .19, f'{b:.2f}', va='center', fontsize=7.5, color=NULL_C)
    ax.set_yticks(y); ax.set_yticklabels(SIMS, fontsize=8.5)
    ax.set_xlim(0, 1.12); ax.set_xlabel(f'best of {ns} samples (mean over {nm} targets)')
    ax.set_title('B · Similarity to the true molecule', fontsize=10, loc='left', fontweight='bold')
    ax.grid(axis='x', alpha=.25, lw=.6); ax.legend(fontsize=8, frameon=False, loc='lower right')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

    # --- C: per-target paired effect --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    a, b = best_n['Tanimoto (Morgan)'], best_m['Tanimoto (Morgan)']
    order = np.argsort(b)
    for i, o in enumerate(order):
        ax.plot([0, 1], [a[o], b[o]], '-', color='#999', lw=.8, alpha=.7, zorder=1)
    ax.scatter(np.zeros(len(a)), a[order], s=26, color=NULL_C, zorder=3, label='null')
    ax.scatter(np.ones(len(b)), b[order], s=26, color=MATCH_C, zorder=3, label='matched')
    won = int((b > a).sum())
    ax.set_xlim(-.35, 1.35); ax.set_xticks([0, 1]); ax.set_xticklabels(['shuffled', 'matched'])
    ax.set_ylabel('best-of-k Tanimoto (Morgan)')
    ax.set_title('C · Paired, per target', fontsize=10, loc='left', fontweight='bold')
    ax.grid(axis='y', alpha=.25, lw=.6)
    ax.text(.5, .02, f'{won}/{len(b)} targets improve with the correct cube',
            transform=ax.transAxes, ha='center', fontsize=8, color='#333')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ttl = f'TERS → molecule: DiffSpectra metrics, {nm} spectra × {ns} samples'
    if args.label:
        ttl += f'  ({args.label})'
    fig.suptitle(ttl, fontsize=12, y=1.01)
    fig.savefig(args.out, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'wrote {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)')
    print(f'  top-1  matched {tk_m[0]:.3f} vs null {tk_n[0]:.3f}')
    print(f'  top-10 matched {tk_m[-1]:.3f} vs null {tk_n[-1]:.3f}')


if __name__ == '__main__':
    main()
