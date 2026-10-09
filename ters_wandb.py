"""Weights & Biases logging for training runs.

Configured by `config.training.wandb`. Errors are logged and never stop training. The API key
is read from WANDB_API_KEY in `.env`; offline runs are uploaded later with
`wandb sync <workdir>/wandb/offline-run-*`.

Usage
-----
    python3 ters_wandb.py --self-test
"""

import hashlib
import logging
import os
import time

from env import DOTENV_PATH, load_dotenv


def _json_safe(obj):
    """Convert a ConfigDict.to_dict() result to JSON-safe values; unknown leaves become str."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


class _NoOp:
    """Logger with no effect, returned when wandb is disabled or unavailable."""

    enabled = False

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class WandbLogger:
    """Thin wrapper over a live wandb run. Never raises."""

    enabled = True

    def __init__(self, run, batch_size):
        self._run = run
        self._batch = batch_size
        self._last_t = time.time()
        self._last_step = None
        self._dead = False

    def log(self, metrics, step):
        """Log `metrics` at `step`, adding `perf/steps_per_s` and `perf/mol_per_s`."""
        if self._dead:
            return
        try:
            now = time.time()
            payload = dict(metrics)
            if self._last_step is not None:
                dt = now - self._last_t
                ds = step - self._last_step
                if dt > 0 and ds > 0:
                    payload['perf/steps_per_s'] = ds / dt
                    if self._batch:
                        payload['perf/mol_per_s'] = ds * self._batch / dt
            self._last_t, self._last_step = now, step
            self._run.log(payload, step=step)
        except Exception as exc:
            self._dead = True
            logging.warning('wandb: logging disabled after error: %s', exc)

    def finish(self):
        if self._dead:
            return
        try:
            self._run.finish()
        except Exception as exc:
            logging.warning('wandb: finish failed: %s', exc)


def init(config, workdir):
    """Start a wandb run, or return a no-op if it is disabled or unavailable.

    Returns an object with .log(metrics, step) and .finish().
    """
    cfg = getattr(config.training, 'wandb', None)
    if cfg is None or not getattr(cfg, 'enabled', False):
        return _NoOp()

    if int(os.environ.get('LOCAL_RANK', getattr(config.training, 'local_rank', 0))) != 0:
        return _NoOp()

    load_dotenv(getattr(cfg, 'dotenv_path', DOTENV_PATH))

    mode = getattr(cfg, 'mode', 'offline')
    if mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        logging.warning('wandb: mode=online but WANDB_API_KEY is unset (looked in %s); '
                        'falling back to offline', getattr(cfg, 'dotenv_path', DOTENV_PATH))
        mode = 'offline'

    try:
        import wandb
    except Exception as exc:
        logging.warning('wandb: unavailable (%s); training continues without it', exc)
        return _NoOp()

    # Set through the environment: wandb.Settings keyword names vary across versions.
    os.environ.setdefault('WANDB_INIT_TIMEOUT',
                          str(getattr(cfg, 'init_timeout', 60)))

    # A run id derived from the workdir lets a resumed job rejoin its run (resume='allow').
    run_id = getattr(cfg, 'run_id', '') or None
    if run_id is None:
        digest = hashlib.sha1(os.path.abspath(workdir).encode()).hexdigest()
        run_id = f'ters{digest[:12]}'

    try:
        run_dir = os.path.join(workdir, 'wandb')
        os.makedirs(run_dir, exist_ok=True)
        run = wandb.init(
            project=getattr(cfg, 'project', '') or 'ters-diffspectra',
            entity=getattr(cfg, 'entity', '') or None,
            name=getattr(cfg, 'run_name', '') or None,
            group=getattr(cfg, 'group', '') or None,
            id=run_id,
            mode=mode,
            dir=workdir,
            resume='allow',
            config=_json_safe(config.to_dict()),
        )
        logging.info('wandb: mode=%s project=%s id=%s dir=%s', mode,
                     getattr(cfg, 'project', ''), run_id, run_dir)
        if mode == 'offline':
            logging.info('wandb: sync afterwards with  wandb sync %s/offline-run-*', run_dir)
        return WandbLogger(run, getattr(config.training, 'batch_size', None))
    except Exception as exc:
        logging.warning('wandb: init failed (%s); training continues without it', exc)
        return _NoOp()


def _self_test():
    """Exercise the module offline, including its failure paths."""
    import shutil
    import tempfile
    import ml_collections

    def make_cfg(**over):
        c = ml_collections.ConfigDict()
        c.training = ml_collections.ConfigDict()
        c.training.batch_size = 128
        c.training.local_rank = 0
        c.device = 'cpu-sentinel-object'
        c.training.wandb = w = ml_collections.ConfigDict()
        w.enabled = True
        w.project = 'ters-selftest'
        w.entity = ''
        w.run_name = ''
        w.group = ''
        w.run_id = ''
        w.mode = 'offline'
        w.dotenv_path = '/nonexistent/.env'
        for k, v in over.items():
            setattr(w, k, v)
        return c

    passed = failed = 0

    def check(name, cond):
        nonlocal passed, failed
        ok = bool(cond)
        passed, failed = passed + ok, failed + (not ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    tmp = tempfile.mkdtemp(prefix='ters-wandb-selftest-')
    try:
        # disabled config must not touch wandb at all
        check('enabled=False returns a no-op', not init(make_cfg(enabled=False), tmp).enabled)

        # non-serialisable leaves must survive config capture
        safe = _json_safe(make_cfg().to_dict())
        check('torch.device-like leaf coerced to str', isinstance(safe['device'], str))

        # missing .env must be silent, not fatal
        load_dotenv('/nonexistent/.env')
        check('missing .env is not an error', True)

        # a dead backend must degrade, not raise
        class Boom:
            def log(self, *a, **k):
                raise RuntimeError('simulated wandb outage')

            def finish(self):
                raise RuntimeError('simulated wandb outage')

        dead = WandbLogger(Boom(), 128)
        dead.log({'train/loss': 1.0}, step=0)
        dead.log({'train/loss': 0.9}, step=50)      # second call must be a silent no-op
        dead.finish()
        check('backend failure degrades to no-op', dead._dead)

        # a real offline run must start, log, and leave a syncable directory behind
        real = init(make_cfg(), tmp)
        if real.enabled:
            real.log({'train/loss': 1.0, 'train/lr': 1e-4}, step=0)
            real.log({'train/loss': 0.5, 'train/lr': 2e-4}, step=50)
            real.finish()
            offline = [d for d in os.listdir(os.path.join(tmp, 'wandb'))
                       if d.startswith('offline-run-')]
            check('offline run directory written', offline)
            check('throughput derived on the second call', real._last_step == 50)
        else:
            check('offline run started', False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if '--self-test' in sys.argv:
        sys.exit(0 if _self_test() else 1)
    print(__doc__)
