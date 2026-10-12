"""Machine-specific settings read from ``.env`` in the repository root.

See ``.env.example`` for the variables. Values already set in the environment take precedence.
"""

import logging
import os

from dotenv import load_dotenv as _load_dotenv

DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')


def load_dotenv(path=DOTENV_PATH):
    """Load ``path`` without overwriting existing environment variables."""
    try:
        return _load_dotenv(dotenv_path=path, override=False)
    except OSError as exc:
        logging.warning('could not read %s: %s', path, exc)
        return False


def _require(name):
    load_dotenv()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'{name} is not set. Add it to {DOTENV_PATH} '
                           f'(see .env.example) or export it.')
    return value


def data_root(*parts):
    """Path under DATA_ROOT, the directory holding datasets and graph files."""
    return os.path.join(_require('DATA_ROOT'), *parts)


def output_root(*parts):
    """Path under OUTPUT_ROOT, the directory for runs, samples, evaluations and figures."""
    return os.path.join(_require('OUTPUT_ROOT'), *parts)
