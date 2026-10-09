"""Distributed data parallel (DDP) helpers for multi-node training.

Enabled with ``config.training.ddp = True`` and launched with one ``srun`` task per GPU;
otherwise training uses ``DataParallel``. Under DDP ``training.batch_size`` is the per-rank
batch, not the global batch.
"""

import os
import datetime

import torch
import torch.distributed as dist


def enabled(config):
    """True when ``config.training.ddp`` is set (default False)."""
    return bool(getattr(config.training, 'ddp', False))


def slurm_topology():
    """Return ``(rank, world_size, local_rank)`` from Slurm, or ``None`` outside ``srun``."""
    if 'SLURM_PROCID' not in os.environ:
        return None
    return (int(os.environ['SLURM_PROCID']),
            int(os.environ.get('SLURM_NTASKS', 1)),
            int(os.environ.get('SLURM_LOCALID', 0)))


def init(config):
    """Join the process group and pin this rank's GPU. Returns ``(rank, world_size)``.

    When DDP is off, returns ``(0, 1)`` and does nothing else.
    """
    if not enabled(config):
        return 0, 1

    topo = slurm_topology()
    if topo is None:
        raise RuntimeError(
            'config.training.ddp=True but SLURM_PROCID is unset, so there is no process group '
            'to join. DDP runs must be launched with `srun --ntasks=<total ranks> '
            '--ntasks-per-node=<gpus per node>`; a plain `python main.py` cannot run DDP.')
    rank, world_size, local_rank = topo

    if world_size > 1 and not all(k in os.environ for k in ('MASTER_ADDR', 'MASTER_PORT')):
        raise RuntimeError(
            'MASTER_ADDR/MASTER_PORT are not set, so ranks cannot find each other. The job '
            'script must export them, e.g. MASTER_ADDR=$(scontrol show hostnames '
            '"$SLURM_JOB_NODELIST" | head -n1).')

    n_visible = torch.cuda.device_count()
    if local_rank >= n_visible:
        raise RuntimeError(
            f'SLURM_LOCALID={local_rank} but this node exposes only {n_visible} GPU(s). '
            f'--ntasks-per-node must not exceed the GPUs per node (4 on gh200 nodes).')

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Long timeout: other ranks may wait while rank 0 builds the MOSES reference sets.
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank,
                                timeout=datetime.timedelta(minutes=60))
    return rank, world_size


def is_main(config=None):
    """True on the rank that writes checkpoints, logs and wandb (always True without DDP)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Hold every rank until all arrive. No-op when not distributed."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value, device):
    """Average a Python float across ranks; returns ``value`` unchanged without DDP."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def describe(config):
    """Return one line naming the parallelism in force and what ``training.batch_size`` means."""
    if not enabled(config):
        n = max(torch.cuda.device_count(), 1)
        return (f'parallelism: DataParallel over {n} visible GPU(s) | '
                f'training.batch_size={config.training.batch_size} is the global batch, '
                f'split {n} ways by one process')
    ws = get_world_size()
    per = config.training.batch_size
    return (f'parallelism: DDP, world_size={ws}, rank={get_rank()} | '
            f'training.batch_size={per} is per rank -> global batch {per * ws}')
