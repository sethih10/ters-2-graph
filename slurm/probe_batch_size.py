"""Find the largest TERS batch size that fits on one GH200, by measuring.

The only honest way to answer "how much can we raise the batch size" is to run a
real training step at each candidate size and watch the allocator. This does that,
reusing the exact objects run_lib.train() builds (run_lib.py:187-222) so the memory
figure is the training loop's, not an approximation of it:

    create_model / get_optimizer / NoiseScheduleVP / get_data_scaler
    losses.optimization_manager -> losses.get_step_fn -> train_step_fn(state, batch)

run_lib.train() is not called directly because it is monolithic -- it would build
checkpoints, samplers and MOSES reference sets before reaching a single step. This
module rebuilds only the pieces a step needs.

Usage:
    # sweep batch size (memory question)
    python3 slurm/probe_batch_size.py --sizes 16,32,64,128,256 --steps 4
    # sweep loader workers at fixed batch (throughput question)
    python3 slurm/probe_batch_size.py --sizes 64 --workers 2,4,8,16 --steps 6
"""

import argparse
import time

import torch

import losses
from datasets import get_dataset, get_dataloader, inf_iterator
from diffusion import NoiseScheduleVP
from models import create_model
from models.ema import ExponentialMovingAverage
from utils import get_data_scaler


# --- ADDED (stm): which config pair to probe -------------------------------------------
# Was hardcoded to the TERS pair. A module-level pair rather than an argument threaded through
# probe() because every caller wants the same pair for a whole sweep, and --config sets it once
# in main(). Default is unchanged, so an existing `probe_batch_size.py --sizes ...` invocation
# probes exactly what it used to.
CONFIG_PAIRS = {
    'ters': ('configs.diffspectra_ters', 'configs.base_ters'),
    'stm': ('configs.diffspectra_stm', 'configs.base_stm'),
}
_PAIR = 'ters'


def build_config(batch_size, num_workers=None):
    """A fresh config at one batch size. ConfigDict from get_config() is unlocked,
    unlike the one main.py hands absl (lock_config=True), so this can be mutated."""
    import importlib

    main_mod, orig_mod = CONFIG_PAIRS[_PAIR]
    get_config = importlib.import_module(main_mod).get_config
    get_config_orig = importlib.import_module(orig_mod).get_config

    config = get_config()
    config.training.batch_size = batch_size
    config.training.eval_batch_size = batch_size
    if num_workers is not None:
        config.data.num_workers = num_workers
    # snapshot sampling would drag in the evaluation package; not part of a memory probe
    config.training.snapshot_sampling = False
    return config, get_config_orig()


def probe(batch_size, steps, num_workers=None):
    config, config_orig = build_config(batch_size, num_workers)
    torch.cuda.reset_peak_memory_stats()

    _, train_ds, val_ds, test_ds, _ = get_dataset(config)
    train_loader, _, _ = get_dataloader(train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1)
    scaler = get_data_scaler(config)
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    losses_seen, t_data, t_step = [], 0.0, 0.0
    for i in range(steps):
        t0 = time.perf_counter()
        batch = next(train_iter)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = train_step_fn(state, batch)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i:                      # step 0 includes cuDNN autotune + allocator warmup
            t_data += t1 - t0
            t_step += t2 - t1
        losses_seen.append(float(loss))

    n = max(steps - 1, 1)
    return dict(
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
        reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
        data_s=t_data / n,
        step_s=t_step / n,
        loss_mean=sum(losses_seen) / len(losses_seen),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='16,32,64,128',
                    help='comma-separated per-process batch sizes to try')
    ap.add_argument('--steps', type=int, default=4,
                    help='training steps per size; step 0 is discarded as warmup')
    ap.add_argument('--config', default='ters', choices=sorted(CONFIG_PAIRS),
                    help='which config pair to probe (default: ters)')
    ap.add_argument('--workers', default=None,
                    help='comma-separated DataLoader worker counts. When given, batch is '
                         'held at the FIRST value of --sizes and workers are swept instead '
                         '-- the useful sweep once the loader is known to be the bottleneck.')
    args = ap.parse_args()

    global _PAIR
    _PAIR = args.config
    print(f'probing config pair: {_PAIR}  ({CONFIG_PAIRS[_PAIR][0]})')

    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device visible -- submit this to gputest, do not run it '
                         'on the login node')

    n_gpu = torch.cuda.device_count()
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f'{n_gpu} x {torch.cuda.get_device_name(0)}  ({total:.0f} GiB each)')
    print('NOTE: models/utils.py:27 wraps the model in DataParallel, so a batch is split\n'
          f'      across all {n_gpu} visible GPU(s); "batch" below is the whole batch.\n')

    sizes = [int(x) for x in args.sizes.split(',')]
    if args.workers:
        combos = [(sizes[0], int(w)) for w in args.workers.split(',')]
        sweep = 'workers'
    else:
        combos = [(b, None) for b in sizes]
        sweep = 'batch'

    header = (f'{"batch":>6} {"wrk":>4} {"peak GiB":>9} {"resv GiB":>9} '
              f'{"data s":>8} {"step s":>8} {"mol/s":>8}  result')
    print(header); print('-' * len(header))

    ok = []
    for bs, nw in combos:
        try:
            r = probe(bs, args.steps, nw)
        except torch.cuda.OutOfMemoryError:
            print(f'{bs:>6} {str(nw):>4} {"":>9} {"":>9} {"":>8} {"":>8} {"":>8}  OOM')
            torch.cuda.empty_cache()
            break
        except Exception as exc:                       # noqa: BLE001 - report and keep going
            print(f'{bs:>6} {str(nw):>4} {"":>9} {"":>9} {"":>8} {"":>8} {"":>8}  FAILED: '
                  f'{type(exc).__name__}: {exc}')
            torch.cuda.empty_cache()
            break
        rate = r['batch_size'] / (r['data_s'] + r['step_s'])
        print(f'{r["batch_size"]:>6} {r["num_workers"]:>4} {r["peak_gib"]:>9.1f} '
              f'{r["reserved_gib"]:>9.1f} {r["data_s"]:>8.2f} {r["step_s"]:>8.2f} '
              f'{rate:>8.1f}  ok')
        ok.append((r, rate))
        torch.cuda.empty_cache()

    if ok:
        best, best_rate = max(ok, key=lambda x: x[1])
        largest = ok[-1][0]
        print(f'\nlargest that fits : batch {largest["batch_size"]} @ '
              f'{largest["num_workers"]} workers ({largest["peak_gib"]:.1f} GiB peak of {total:.0f})')
        print(f'best throughput   : batch {best["batch_size"]} @ '
              f'{best["num_workers"]} workers ({best_rate:.1f} molecules/s)')
        # data_s >> step_s means the GPU is idling on I/O and a bigger batch will not help;
        # more DataLoader workers will.
        if best['data_s'] > 2 * best['step_s']:
            print(f'\nI/O BOUND: data {best["data_s"]:.2f}s vs step {best["step_s"]:.2f}s. '
                  f'Raise data.num_workers, not the batch.')
        if sweep == 'workers':
            print(f'\nSet data.num_workers = {best["num_workers"]} in '
                  f'configs/diffspectra_ters.py')
        else:
            print(f'\nSet base_batch_size = {best["batch_size"] // max(n_gpu, 1)} in '
                  f'configs/diffspectra_ters.py:153\n'
                  f'  (multiplied by training.num_gpus at line 154; num_gpus is {n_gpu} here)')


if __name__ == '__main__':
    main()
