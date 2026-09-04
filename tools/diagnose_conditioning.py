"""Is the trained model actually using the TERS cube?

The training loss went flat after ~4000 steps (job 925597). That is consistent with two
very different stories:
  (a) the model learned an unconditional prior over molecules and ignores the cube, or
  (b) the model is conditioning fine and the total loss is dominated by the irreducible
      high-noise timesteps, which hide real progress.

`sampling.py` has no TERS branch (D-8), so the usual answer -- generate molecules and score
them -- is not available. These four tests distinguish (a) from (b) without a sampler.

REUSE: get_dataset/get_dataloader (datasets/build_dataset.py), create_model (models/utils.py),
       get_sde_graph_loss_fn + process_edge_batch (losses.py), NoiseScheduleVP, get_data_scaler.
NEW:   only the shuffling and the paired-seed harness.

Run on a GPU node:  python -m tools.diagnose_conditioning --ckpt <path> [--batches 5]
"""
import argparse, importlib.util, random, sys
import numpy as np
import torch

from datasets import get_dataset, get_dataloader
from models.ema import ExponentialMovingAverage
from models import create_model
import losses
from utils import get_data_scaler
from diffusion import NoiseScheduleVP


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/diffspectra_ters.py')
    ap.add_argument('--config-qm9', default='configs/base_ters.py')
    ap.add_argument('--batches', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=6)
    args = ap.parse_args()

    config = load_config(args.config)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_qm9 = load_config(args.config_qm9)

    print(f'device: {config.device}')
    _, _, val_ds, test_ds, _ = get_dataset(config)
    _, _, val_o, test_o, _ = get_dataset(cfg_qm9)
    # get_dataloader wants (train, val, test); the val split is what we want to read.
    _, val_loader, _ = get_dataloader(val_ds, val_ds, test_ds, config)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    ck = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    ema.load_state_dict(ck['ema'])
    ema.copy_to(model.parameters())          # evaluate the EMA weights, as --mode eval would
    model.eval()
    print(f"loaded step {ck['step']} (EMA weights)")

    net = unwrap(model)
    noise_scheduler = NoiseScheduleVP(config.sde.schedule,
                                      continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)
    scaler = get_data_scaler(config)
    loss_fn = losses.get_sde_graph_loss_fn(noise_scheduler, False, scaler, config)

    batches = []
    it = iter(val_loader)
    for _ in range(args.batches):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    print(f'val batches held: {len(batches)}  (batch size {batches[0]["positions"].shape[0]})')

    # ---------------------------------------------------------------- A: encoder collapse
    print('\n' + '=' * 72)
    print('A. Does the TERS encoder produce a DIFFERENT vector per molecule?')
    print('=' * 72)
    embs = []
    with torch.no_grad():
        for b in batches:
            embs.append(net.cond_encoder(b['context'].to(config.device)).float().cpu())
    E = torch.cat(embs)                                   # [N, 256]
    across = E.std(dim=0)                                 # spread across molecules, per dim
    mag = E.abs().mean()
    print(f'  embeddings: {tuple(E.shape)}')
    print(f'  mean |activation|            : {mag:.6f}')
    print(f'  mean across-sample std       : {across.mean():.6f}')
    print(f'  ratio std/|activation|       : {across.mean() / (mag + 1e-12):.6f}')
    print(f'  dims with std < 1e-4         : {(across < 1e-4).sum().item()} / {E.shape[1]}')
    cos = torch.nn.functional.cosine_similarity(E[:, None, :], E[None, :, :], dim=-1)
    off = cos[~torch.eye(len(E), dtype=bool)]
    print(f'  mean pairwise cosine sim     : {off.mean():.6f}   (1.0 => identical vectors)')

    # ---------------------------------------------------------------- B: signal magnitude
    print('\n' + '=' * 72)
    print('B. How big is the cube signal next to the time embedding it is ADDED to?')
    print('   (dmt.py:428  time_emb = time_mlp(noise_level) + context)')
    print('=' * 72)
    with torch.no_grad():
        ctx = net.cond_lin(net.cond_encoder(batches[0]['context'].to(config.device)))
        t = torch.rand(ctx.shape[0], device=config.device) * 0.99999 + 1e-5
        a, s = noise_scheduler.marginal_prob(t)
        nl = torch.log(a ** 2 / s ** 2)
        temb = net.time_mlp(nl)
    print(f'  ||context||      mean {ctx.norm(dim=-1).mean():10.4f}   std across batch {ctx.norm(dim=-1).std():.4f}')
    print(f'  ||time_emb||     mean {temb.norm(dim=-1).mean():10.4f}')
    print(f'  context / time   ratio {(ctx.norm(dim=-1).mean() / temb.norm(dim=-1).mean()):.6f}')

    # ------------------------------------------------- C: paired matched vs shuffled vs zero
    print('\n' + '=' * 72)
    print('C. DECISIVE: same molecules, same t, same noise -- only the cube pairing changes.')
    print('   If matched == shuffled, the cube is contributing nothing.')
    print('=' * 72)
    rows = []
    for bi, b in enumerate(batches):
        n = b['positions'].shape[0]
        perm = torch.roll(torch.arange(n), 1)      # every molecule gets someone else's cube
        for sd in range(args.seeds):
            out = {}
            for name in ('matched', 'shuffled', 'zeroed'):
                bb = dict(b)
                if name == 'shuffled':
                    bb['context'] = b['context'][perm].clone()
                elif name == 'zeroed':
                    bb['context'] = torch.zeros_like(b['context'])
                torch.manual_seed(1000 * bi + sd); random.seed(1000 * bi + sd)
                np.random.seed(1000 * bi + sd)
                with torch.no_grad():
                    out[name] = float(loss_fn(model, bb))
            rows.append(out)
        print(f'  batch {bi}: matched {np.mean([r["matched"] for r in rows[-args.seeds:]]):.5f}  '
              f'shuffled {np.mean([r["shuffled"] for r in rows[-args.seeds:]]):.5f}  '
              f'zeroed {np.mean([r["zeroed"] for r in rows[-args.seeds:]]):.5f}')

    m = np.array([r['matched'] for r in rows]); sh = np.array([r['shuffled'] for r in rows])
    z = np.array([r['zeroed'] for r in rows])
    d = sh - m
    print(f'\n  n paired samples            : {len(m)}')
    print(f'  matched  mean loss          : {m.mean():.6f}  (sd {m.std():.6f})')
    print(f'  shuffled mean loss          : {sh.mean():.6f}  (sd {sh.std():.6f})')
    print(f'  zeroed   mean loss          : {z.mean():.6f}  (sd {z.std():.6f})')
    print(f'  paired diff (shuffled-match): {d.mean():+.8f}  (sd {d.std():.8f})')
    if d.std() > 0:
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f'  paired t statistic          : {tstat:+.3f}   (|t| < 2 => indistinguishable)')
    print(f'  relative effect             : {100 * d.mean() / m.mean():+.5f}% of the loss')

    print('\n' + '=' * 72)
    if abs(d.mean()) < 1e-4 or (d.std() > 0 and abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) < 2):
        print('VERDICT: giving the model the WRONG spectrum costs it nothing.')
        print('         The cube is not being used. The model is an unconditional prior.')
    else:
        print('VERDICT: the wrong spectrum measurably hurts the loss -- conditioning IS active.')
    print('=' * 72)


if __name__ == '__main__':
    main()
