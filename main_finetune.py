"""Train or evaluate the frozen DiffSpectra denoiser with a trainable TERS encoder.

Use configs/diffspectra_ters_frozen.py with this entry point. It selects the frozen
model factory under the DMT architecture name expected by the unchanged dataset
and loss functions, then delegates flags, logging, training, and evaluation to main.py.
"""

from absl import app

from models.dmt_frozen import configure_frozen_runtime

# Defines every flag main.py defines, because they are declared at its module scope.
import main as _main


def main(argv):
    # The data and loss functions dispatch on the DMT architecture name. Select
    # the frozen factory for this config; ordinary models remain unchanged.
    configure_frozen_runtime(_main.FLAGS.config)
    _main.main(argv)


if __name__ == '__main__':
    app.run(main)
