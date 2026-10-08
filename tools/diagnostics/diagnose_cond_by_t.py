"""Matched versus shuffled-cube loss at fixed noise levels.

Prints, for each t, both losses of the EMA model on the same batches and seeds, their difference
and the paired t statistic.

Usage
-----
    python -m tools.diagnostics.diagnose_cond_by_t --ckpt CKPT [--set KEY=VALUE]
"""
import argparse, importlib.util, random
import numpy as np
import torch

from datasets import get_dataset, get_dataloader
from models.ema import ExponentialMovingAverage
from models import create_model
import losses
from utils import get_data_scaler
from diffusion import NoiseScheduleVP
from config_utils import add_set_arg, load_config
from tools.diagnostics.diagnose_progress import FixedT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/diffspectra_ters.py')
    ap.add_argument('--config-qm9', default='configs/base_ters.py')
    ap.add_argument('--batches', type=int, default=2)
    ap.add_argument('--seeds', type=int, default=4)
    add_set_arg(ap)
    args = ap.parse_args()

    config = load_config(args.config, args.overrides); config.device = torch.device('cuda')
    cfg_q = load_config(args.config_qm9)
    _, _, val_ds, test_ds, _ = get_dataset(config)
    get_dataset(cfg_q)
    _, val_loader, _ = get_dataloader(val_ds, val_ds, test_ds, config)

    batches = [next(iter(val_loader))] if args.batches == 1 else None
    if batches is None:
        it = iter(val_loader); batches = [next(it) for _ in range(args.batches)]

    noise_scheduler = NoiseScheduleVP(config.sde.schedule,
                                      continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)
    loss_fn = losses.get_sde_graph_loss_fn(noise_scheduler, False, get_data_scaler(config), config)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    ck = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    ema.load_state_dict(ck['ema']); ema.copy_to(model.parameters()); model.eval()
    print(f"loaded step {int(ck['step'])} (EMA)")

    t_grid = [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]
    print('\n' + '=' * 82)
    print('Loss penalty for a mismatched cube, by noise level')
    print('=' * 82)
    print(f"{'t':>6}{'matched':>12}{'shuffled':>12}{'penalty':>12}{'% of loss':>12}{'paired t':>11}")
    for tv in t_grid:
        m, s = [], []
        for bi, b in enumerate(batches):
            n = b['positions'].shape[0]
            perm = torch.roll(torch.arange(n), 1)
            for sd in range(args.seeds):
                bb = dict(b); bb['context'] = b['context'][perm].clone()
                torch.manual_seed(sd + 77 * bi); random.seed(sd + 77 * bi)
                with FixedT(tv), torch.no_grad():
                    m.append(float(loss_fn(model, b)))
                torch.manual_seed(sd + 77 * bi); random.seed(sd + 77 * bi)
                with FixedT(tv), torch.no_grad():
                    s.append(float(loss_fn(model, bb)))
        m, s = np.array(m), np.array(s); d = s - m
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0.0
        print(f'{tv:6.2f}{m.mean():12.4f}{s.mean():12.4f}{d.mean():+12.4f}'
              f'{100 * d.mean() / m.mean():+12.2f}{tstat:+11.2f}')
    print('=' * 82)


if __name__ == '__main__':
    main()
