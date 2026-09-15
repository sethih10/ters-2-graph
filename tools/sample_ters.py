"""DEPRECATED shim -- this file moved to tools/sample_cond.py.

The name was wrong: the script has handled STM z-stacks as well as TERS cubes since the STM
retrofit (`cube_of` probes `.ters` and `.stm`), and `metric_evaluation/run_metrics.py` drives
it with STM defaults. Renamed to `sample_cond.py`; kept here so sbatch scripts and notebooks
holding the old path keep working.

Delegates to the real `main()`, so behaviour is identical -- including the argparse defaults,
which stay TERS-shaped (`--modality ters` is the default in the new file for exactly this
reason). Prefer the new path in anything written from now on:

    python -m tools.sample_cond --modality stm --ckpt <path> --split test
"""
import sys

try:
    from sample_cond import main            # `python3 tools/sample_ters.py`
except ImportError:
    from tools.sample_cond import main      # `python3 -m tools.sample_ters`

if __name__ == '__main__':
    print('tools/sample_ters.py is deprecated; use tools/sample_cond.py '
          '(same arguments, plus --modality).', file=sys.stderr)
    main()
