"""Linear probe of molecular composition from the conditioning encoder output.

Prints held-out ridge-probe R^2 per element, size-controlled retrieval against permutation
chance and mean pairwise cosine, for the trained EMA model and the encoder at initialisation.

Usage
-----
    python -m tools.diagnostics.probe_conditioning --ckpt CKPT [--no-control] [--set KEY=VALUE]
"""

import argparse

import numpy as np
import torch

from datasets import get_dataset, get_dataloader
from models import create_model
from models.ema import ExponentialMovingAverage
from config_utils import add_set_arg, load_config


def unwrap(m):
    return m.module if hasattr(m, 'module') else m


@torch.no_grad()
def collect(model, loader, n_target, device):
    """Return encoder outputs ``(n, nf)`` and per-element atom counts ``(n, n_elem)``."""
    enc = unwrap(model).cond_encoder
    H, C = [], []
    seen = 0
    for batch in loader:
        ctx = batch['context'].to(device)
        h = enc(ctx)                                   # (B, nf), the input to cond_lin
        aoh = batch['atom_one_hot'].to(device)         # (B, N, n_elem)
        mask = batch['atom_mask'].to(device)           # (B, N)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)                  # (B, N, 1)
        counts = (aoh * mask).sum(dim=1)               # (B, n_elem)
        H.append(h.float().cpu().numpy())
        C.append(counts.float().cpu().numpy())
        seen += h.shape[0]
        if seen >= n_target:
            break
    return np.concatenate(H)[:n_target], np.concatenate(C)[:n_target]


def ridge_fit(X, Y, lam):
    """Closed-form ridge with an intercept. X:(n,d) Y:(n,k) -> W:(d+1,k)."""
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    d = X1.shape[1]
    A = X1.T @ X1 + lam * np.eye(d)
    A[-1, -1] -= lam                                   # do not penalise the intercept
    return np.linalg.solve(A, X1.T @ Y)


def ridge_pred(X, W):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ W


def r2(Y, P):
    ss_res = ((Y - P) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)


def size_controlled_retrieval(P, Y, n_heavy, rng, n_perm=200):
    """Mean reciprocal rank of the true composition within equal-heavy-atom-count groups.

    Chance is the same statistic on shuffled predictions; 1/group_size would understate it.
    Returns ``(mrr, chance_mrr, mrr / chance_mrr)``.
    """
    def mrr(pred):
        vals = []
        for size in np.unique(n_heavy):
            idx = np.where(n_heavy == size)[0]
            if len(idx) < 3:
                continue
            for pos, i in enumerate(idx):
                d = np.linalg.norm(Y[idx] - pred[i][None, :], axis=1)
                vals.append(1.0 / (1 + int((d < d[pos]).sum())))
        return float(np.mean(vals)) if vals else float('nan')

    real = mrr(P)
    chance = np.mean([mrr(P[rng.permutation(len(P))]) for _ in range(n_perm)])
    return real, float(chance), (real / chance if chance else float('nan'))


def probe_one(model, loader, device, n, lam, seed, label):
    H, C = collect(model, loader, n, device)
    n_heavy = C[:, 1:].sum(axis=1) if C.shape[1] > 1 else C.sum(axis=1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(H))
    cut = int(0.7 * len(H))
    tr, te = perm[:cut], perm[cut:]

    mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-8
    Xtr, Xte = (H[tr] - mu) / sd, (H[te] - mu) / sd
    W = ridge_fit(Xtr, C[tr], lam)
    P = ridge_pred(Xte, W)

    scores = r2(C[te], P)
    mrr_real, mrr_chance, ret = size_controlled_retrieval(P, C[te], n_heavy[te], rng)
    # Mean pairwise cosine similarity.
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    S = Hn @ Hn.T
    off = S[~np.eye(len(S), dtype=bool)]
    print(f'\n--- {label}')
    print(f'    n={len(H)} (train {len(tr)} / test {len(te)}), dim={H.shape[1]}')
    print(f'    mean pairwise cosine      : {off.mean():.4f}   (1.0 => total collapse)')
    print(f'    R^2 per element (held out): ' +
          ' '.join(f'{v:+.3f}' for v in scores))
    print(f'    mean R^2 over elements    : {np.nanmean(scores):+.4f}')
    print(f'    size-controlled MRR       : {mrr_real:.4f}  (permutation chance '
          f'{mrr_chance:.4f})  =  {ret:.3f}x chance')
    return dict(cone=float(off.mean()), r2=float(np.nanmean(scores)), retrieval=ret)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/diffspectra_stm.py')
    ap.add_argument('--config-qm9', default='configs/base_stm.py')
    ap.add_argument('--n', type=int, default=1024, help='molecules to probe')
    ap.add_argument('--lam', type=float, default=1.0, help='ridge penalty')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-control', action='store_true',
                    help='skip the control run with the encoder at initialisation; without it '
                         'the trained numbers have no baseline')
    add_set_arg(ap)
    args = ap.parse_args()

    config = load_config(args.config, args.overrides)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_q = load_config(args.config_qm9)
    _, _, val_ds, test_ds, _ = get_dataset(config)
    get_dataset(cfg_q)
    _, val_loader, _ = get_dataloader(val_ds, val_ds, test_ds, config)

    print('=' * 78)
    print('Linear probe: is molecular composition decodable from the conditioning vector?')
    print('=' * 78)

    out = {}
    if not args.no_control:
        ctrl = create_model(config)
        ctrl.eval()
        out['control'] = probe_one(ctrl, val_loader, config.device, args.n,
                                       args.lam, args.seed, 'control: encoder at initialisation (pretrained if the config sets a path)')
        del ctrl
        torch.cuda.empty_cache()

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    ck = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    ema.load_state_dict(ck['ema'])
    ema.copy_to(model.parameters())
    model.eval()
    out['trained'] = probe_one(model, val_loader, config.device, args.n, args.lam,
                               args.seed, f'trained: step {int(ck["step"])} (EMA)')

    if 'control' in out:
        t, c = out['trained'], out['control']
        print('\n' + '=' * 78)
        print(f'  retrieval  : {t["retrieval"]:.3f}x trained vs {c["retrieval"]:.3f}x at init')
        print(f'  mean R^2   : {t["r2"]:+.4f} trained vs {c["r2"]:+.4f} at init')
        print(f'  cone       : {t["cone"]:.4f} trained vs {c["cone"]:.4f} at init')
        print('  A trained number at or below the control means denoising training did not add '
              'decodable\n  molecular information to the conditioning vector.')
        print('=' * 78)


if __name__ == '__main__':
    main()
