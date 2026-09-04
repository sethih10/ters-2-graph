"""Run the conditioning tests on BOTH held-out splits.

Everything reported so far came from the validation split only. The test split
(782 molecules, verified disjoint from train and val) had never been touched. This
runs the same paired matched/shuffled comparison at fixed noise on each, so val and
test can be compared directly.
"""
import argparse, random
import numpy as np
import torch

from datasets import get_dataset, get_dataloader
from models.ema import ExponentialMovingAverage
from models import create_model
import losses
from utils import get_data_scaler
from diffusion import NoiseScheduleVP
from tools.diagnose_progress import FixedT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/diffspectra_ters.py')
    ap.add_argument('--config-qm9', default='configs/base_ters.py')
    ap.add_argument('--batches', type=int, default=2)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    config = load_config(args.config); config.device = torch.device('cuda')
    cfg_q = load_config(args.config_qm9)
    _, _, val_ds, test_ds, _ = get_dataset(config)
    get_dataset(cfg_q)

    noise_scheduler = NoiseScheduleVP(config.sde.schedule,
                                      continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)
    loss_fn = losses.get_sde_graph_loss_fn(noise_scheduler, False, get_data_scaler(config), config)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    ck = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    ema.load_state_dict(ck['ema']); ema.copy_to(model.parameters()); model.eval()
    print(f"checkpoint step {int(ck['step'])} (EMA)\n")

    t_grid = [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]
    table = {}
    for name, ds in (('val', val_ds), ('test', test_ds)):
        # get_dataloader(train, val, test); we read the middle loader for whichever ds we pass.
        _, loader, _ = get_dataloader(ds, ds, ds, config)
        it = iter(loader)
        batches = [next(it) for _ in range(args.batches)]
        print(f'{name}: {len(ds)} molecules, {len(batches)} batches of '
              f'{batches[0]["positions"].shape[0]}')
        per_t = {}
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
            m, s = np.array(m), np.array(s)
            per_t[tv] = (m.mean(), s.mean(), (s - m).mean())
        table[name] = per_t

    print('\n' + '=' * 92)
    print('Validation vs test: loss, and the cost of the WRONG spectrum, at fixed noise')
    print('=' * 92)
    print(f"{'t':>6}{'val loss':>11}{'test loss':>11}{'val pen':>10}{'test pen':>10}"
          f"{'val %':>9}{'test %':>9}{'gap':>9}")
    for tv in t_grid:
        v, te = table['val'][tv], table['test'][tv]
        print(f'{tv:6.2f}{v[0]:11.4f}{te[0]:11.4f}{v[2]:+10.4f}{te[2]:+10.4f}'
              f'{100*v[2]/v[0]:+9.1f}{100*te[2]/te[0]:+9.1f}{100*(te[0]-v[0])/v[0]:+9.2f}')
    print('=' * 92)
    print('gap = how much worse test is than val at that noise level; a large positive gap')
    print('would be the overfitting signal the run has no validation loss to show.')


if __name__ == '__main__':
    main()
