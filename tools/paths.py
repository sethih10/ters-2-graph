"""Where outputs go: the repo holds code, /scratch holds data, runs and figures.

--- NEW (stm) -------------------------------------------------------------------------
The repo lives on a **15 GB $HOME quota** and the run outputs do not fit in it: one TERS run
is 3.7 GB, one STM checkpoint is 587 MB. Before this module the tools defaulted to
``exp/<run>/generated`` *inside the repo*, and `exp/ters_20260828_925597` alone had taken
$HOME to 100 % full (76 MB free) -- which is a real failure mode here, not tidiness: a
training job that hits the quota mid-write dies with a truncated checkpoint, which is exactly
how STM job 1134521 lost its resume file, and how ters-ae lost job 989420.

So every output path resolves through here, and the default root is project scratch.

Override with ``TERS_EXP_ROOT`` (mirrors the ``TERS_WORKDIR`` convention the sbatch scripts
already use), e.g. to write somewhere else for a one-off:

    TERS_EXP_ROOT=/scratch/project_2001912/scratch-me python3 tools/sample_cond.py ...
"""

import os

DEFAULT_ROOT = '/scratch/project_2001912'


def exp_root():
    """Root under which runs, evaluations and figures are written."""
    return os.environ.get('TERS_EXP_ROOT', DEFAULT_ROOT)


def under(*parts):
    """A path under the output root."""
    return os.path.join(exp_root(), *parts)


def generated(run, subdir='generated'):
    """The generated-samples directory for a run, e.g. generated('ters-runs/ters_20260828_925597')."""
    return os.path.join(exp_root(), run, subdir)


if __name__ == '__main__':
    print(f'TERS_EXP_ROOT = {exp_root()}'
          f'{"  (default)" if "TERS_EXP_ROOT" not in os.environ else "  (from environment)"}')
    for name, p in [
        ('TERS baseline generated', generated('ters-runs/ters_20260828_925597')),
        ('STM eval', under('stm-eval')),
        ('STM eval figures', under('stm-eval', 'figures')),
        ('STM runs', under('stm-runs')),
    ]:
        print(f'  {name:26s} {p}   {"exists" if os.path.exists(p) else "-"}')
