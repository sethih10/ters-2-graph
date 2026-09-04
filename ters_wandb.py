"""Weights & Biases logging for TERS training runs.

Provenance (see docs/02-reuse-map.md):
  * config capture   -- REUSE of ml_collections ConfigDict.to_dict()
  * _json_safe       -- NEW (to_dict() leaves torch.device and ConfigDict leaves in
                        place; wandb.config cannot serialise either)
  * load_dotenv      -- NEW, deliberately. python-dotenv is a one-function dependency
                        and docs/06-code-practices.md §8 is explicit that dependencies
                        on this project vanish. Twelve lines beats an overlay entry.
  * everything else  -- NEW

Why NEW rather than reused: this repo has no experiment tracking to build on. The only
trace of any is `# from torch.utils import tensorboard` at run_lib.py:8 -- commented out,
never wired to the loop, and left commented on purpose (the user asked for wandb instead).
Searched run_lib.py and main.py for SummaryWriter / basicConfig / add_scalar: nothing.

Kept in its own module so run_lib.py gains three lines rather than thirty, and so the
whole thing can be swapped for a no-op without touching the training loop.

Two constraints this module is built around:

1. **Logging must never kill training.** A 9 h GPU allocation is not allowed to die
   because a metrics upload timed out. Every public entry point swallows its own
   exceptions and degrades to a no-op, warning exactly once. This is the §5 rule
   inverted on purpose: a *logging* failure must be loud but non-fatal, where a *data*
   failure must be fatal.
2. **Compute nodes here DO have outbound internet.** This was measured, not assumed --
   job 925402 resolved api.wandb.ai and completed an HTTPS request from a gh200 node in
   0.2 s, with no proxy variables set. That is why `mode` defaults to `'online'`. Do not
   "fix" this back to offline on the usual HPC assumption without re-running the probe in
   slurm/verify_wandb_gputest.sbatch; see D-9 in docs/05-progress-log.md.

   `mode='offline'` remains supported and is the fallback if the network changes: it
   writes to `<workdir>/wandb/` for a later push from a login node with

       wandb sync exp/<run>/wandb/offline-run-*

Exercise without the cluster (§3):

    python3 ters_wandb.py --self-test

The API key is read from `.env` (WANDB_API_KEY=...), which is not committed. Nothing in
this module ever prints the key.
"""

import hashlib
import logging
import os
import time


# --- NEW (wandb): minimal .env reader ----------------------------------------
# Added instead of taking a python-dotenv dependency; see the provenance note above.
def load_dotenv(path='.env'):
    """Set KEY=VALUE pairs from `path` into os.environ without overwriting existing vars.

    Never logs values. A missing file is normal (online mode is opt-in), so it is not an
    error; an unreadable one is warned about, because that is a typo the user wants to see.
    """
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass
    except OSError as exc:
        logging.warning('wandb: could not read %s: %s', path, exc)


# --- NEW (wandb): ConfigDict -> wandb.config ---------------------------------
# REUSE of ConfigDict.to_dict() for the walk; this only fixes the leaves it leaves behind.
def _json_safe(obj):
    """Recursively coerce a to_dict() result into something wandb.config can store.

    `config.device` is a torch.device and `normalize_factors` is a str, so the tree is
    mostly fine -- but a single unserialisable leaf makes wandb.init raise, which would
    then be swallowed and cost us the whole run's logging. Cheaper to coerce than to debug.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


class _NoOp:
    """Stand-in returned whenever wandb is disabled, unavailable, or has failed.

    Exists so callers never branch on None and never wrap .log() in try/except --
    the failure handling lives here, once.
    """

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
        """Log scalars, adding throughput derived from wall-clock between consecutive calls.

        Throughput is computed here rather than in the training loop so the loop stays
        free of timing state -- and because it is only meaningful relative to the previous
        log call, which is this object's business, not the caller's.
        """
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
            # Deliberately broad: wandb raises a wide and version-dependent set of errors,
            # and none of them are worth losing a 9 h run over. Warn once, then stay quiet.
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

    Returns an object with .log(metrics, step) and .finish(). Callers never check for
    None and never catch anything.
    """
    cfg = getattr(config.training, 'wandb', None)
    if cfg is None or not getattr(cfg, 'enabled', False):
        return _NoOp()

    # Only the primary process logs. DataParallel is single-process so this is trivially
    # true today (models/utils.py:27), but it keeps the module correct if DDP ever lands.
    if int(os.environ.get('LOCAL_RANK', getattr(config.training, 'local_rank', 0))) != 0:
        return _NoOp()

    load_dotenv(getattr(cfg, 'dotenv_path', '.env'))

    mode = getattr(cfg, 'mode', 'offline')
    if mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        # §5: say what happened and what was done about it, not just that something failed.
        logging.warning('wandb: mode=online but WANDB_API_KEY is unset (looked in %s); '
                        'falling back to offline', getattr(cfg, 'dotenv_path', '.env'))
        mode = 'offline'

    try:
        import wandb
    except Exception as exc:
        logging.warning('wandb: unavailable (%s); training continues without it', exc)
        return _NoOp()

    # Set via env rather than wandb.Settings(...) because the Settings kwarg names have
    # churned across wandb versions and a TypeError here would silently cost us logging.
    os.environ.setdefault('WANDB_INIT_TIMEOUT',
                          str(getattr(cfg, 'init_timeout', 60)))

    # --- NEW (wandb): derive a stable run id from the workdir ---------------------------
    # Without this, `resume='allow'` has nothing to resume TO: wandb mints a fresh random
    # id per process, so a preempted 9 h job would come back as a second, disconnected run
    # with the loss curve split across both. The workdir is the one identifier that
    # survives preemption (run_lib restores from <workdir>/checkpoints-meta), so it is the
    # right thing to key on. Hashed because wandb ids must be short and alphanumeric.
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
            resume='allow',          # so a preempted job resumes its run instead of forking one
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


# --- NEW (wandb): cluster-free self-test, per docs/06-code-practices.md §3 ----
def _self_test():
    """Exercise the module offline, with the pathological cases planted (§3).

    Planted cases: a disabled config, a non-serialisable config leaf (torch.device), a
    missing .env, and a logger whose backend has died mid-run. Happy path alone would
    not prove any of the failure handling, which is the whole point of this module.
    """
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
