"""Did the last 24000 steps learn anything, and where in the noise schedule?

The total training loss is an average over a random timestep t per sample, so it is
dominated by the high-noise end where the loss floor is irreducible. A flat total loss can
therefore hide real progress at low noise. This resolves that by holding t FIXED and
comparing two checkpoints on identical noise.

torch.rand is used exactly once in losses.get_sde_graph_loss_fn's loss_fn -- to draw t
(losses.py:327). The noise itself comes from torch.randn via sample_*_noise. So patching
torch.rand alone pins t and leaves everything else untouched.
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


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.get_config()


class FixedT:
    """Pin the single torch.rand call inside loss_fn to a constant t."""
    def __init__(self, val): self.val = val
    def __enter__(self):
        self.orig = torch.rand
        def patched(*a, **kw):
            out = self.orig(*a, **kw)
            return torch.full_like(out, self.val)
        torch.rand = patched
        return self
    def __exit__(self, *a):
        torch.rand = self.orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--config', default='configs/diffspectra_ters.py')
    ap.add_argument('--config-qm9', default='configs/base_ters.py')
    ap.add_argument('--batches', type=int, default=2)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    config = load_config(args.config); config.device = torch.device('cuda')
    cfg_q = load_config(args.config_qm9)
    _, _, val_ds, test_ds, _ = get_dataset(config)
    _, _, _, _, _ = get_dataset(cfg_q)
    _, val_loader, _ = get_dataloader(val_ds, val_ds, test_ds, config)

    batches = []
    it = iter(val_loader)
    for _ in range(args.batches):
        batches.append(next(it))

    noise_scheduler = NoiseScheduleVP(config.sde.schedule,
                                      continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)
    scaler = get_data_scaler(config)
    loss_fn = losses.get_sde_graph_loss_fn(noise_scheduler, False, scaler, config)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)

    t_grid = [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]
    results = {}
    for ck_path in args.ckpts:
        ck = torch.load(ck_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ck['model'], strict=True)
        ema.load_state_dict(ck['ema']); ema.copy_to(model.parameters()); model.eval()
        step = int(ck['step'])
        per_t = {}
        for tv in t_grid:
            vals = []
            for bi, b in enumerate(batches):
                for sd in range(args.seeds):
                    torch.manual_seed(sd + 77 * bi); random.seed(sd + 77 * bi)
                    with FixedT(tv), torch.no_grad():
                        vals.append(float(loss_fn(model, b)))
            per_t[tv] = (float(np.mean(vals)), float(np.std(vals)))
        results[step] = per_t
        print(f'evaluated {ck_path} (step {step})')

    steps = sorted(results)
    print('\n' + '=' * 78)
    print('Loss at FIXED noise level, identical noise across checkpoints')
    print('=' * 78)
    hdr = f"{'t':>6} {'alpha^2/sig^2':>14}" + ''.join(f"{'step '+str(s):>14}" for s in steps)
    if len(steps) >= 2:
        hdr += f"{'change':>12}{'%':>9}"
    print(hdr)
    for tv in t_grid:
        a, s = noise_scheduler.marginal_prob(torch.tensor([tv]))
        snr = float((a ** 2 / s ** 2))
        row = f'{tv:6.2f} {snr:14.4f}'
        for st in steps:
            row += f'{results[st][tv][0]:14.4f}'
        if len(steps) >= 2:
            first, last = results[steps[0]][tv][0], results[steps[-1]][tv][0]
            row += f'{last-first:+12.4f}{100*(last-first)/first:+9.2f}'
        print(row)
    print('=' * 78)
    print('t -> 0 is nearly-clean data (structure detail); t -> 1 is nearly pure noise.')


if __name__ == '__main__':
    main()
