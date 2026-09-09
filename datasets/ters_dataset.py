"""Phase 2 -- pair the TERS image cubes in the HDF5 with the Phase-1 graph sidecar.

``preprocess_ters.py`` wrote one record per molecule carrying the graph fields and an
``h5_index`` back-pointer.  This module joins the two: the sidecar is small enough to hold
in RAM, the image cubes are read from the HDF5 one at a time.

Provenance (see docs/02-reuse-map.md 5, 7):
  * ``NormalizeVectorized``  -- ADAPT of Multimodal-ters/src/transforms/normalize.py:19-35
                                (per-channel mean/std over the H,W axes + zero-std guard)
  * ``TersGraphDataset``     -- ADAPT of Multimodal-ters/src/datasets/ters_hdf5.py:24-134
                                (key selection, permute, transform hook; changed: lazy file
                                open for fork-safety, returns a PyG ``Data``, graph fields
                                come from the sidecar rather than the unused ``targets``)

``CollateTers`` lives in ``build_dataset.py`` beside the ``CollateSpectra`` it subclasses;
keeping it there means this module imports nothing from ``build_dataset`` at module level
and there is no import cycle.

Usage
-----
    # standalone smoke test against a real split (no training loop involved)
    python -m datasets.ters_dataset --h5 .../val.h5 --graphs datasets/ters/graphs_val.pt
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class NormalizeVectorized:
    """ADAPT of Multimodal-ters/src/transforms/normalize.py:19-35.

    Per-channel standardisation over the spatial axes of a ``(C, H, W)`` cube.

    **This is not optional.** Raw cube values run 0 to ~2556 (docs/02-reuse-map.md 5) and
    ``EdgeComSpectraTransform`` will not touch them -- it log-normalises only ``ir``/``uv``/
    ``raman``, each guarded by ``hasattr`` (datasets/build_dataset.py:145-151).  Skipping
    this is the most likely cause of NaN in the first hundred training steps.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, cube):
        mean = cube.mean(dim=(1, 2), keepdim=True)
        std = cube.std(dim=(1, 2), keepdim=True)
        # Zero-std guard.  TERS cubes contain genuinely dead channels, and dividing by a
        # near-zero std amplifies float32 cancellation noise (a constant channel's mean is
        # only exact to ~1e-9) into O(0.1) garbage.  Substituting 1.0 leaves such a channel
        # at ~0 instead.
        std = torch.where(std < self.eps, torch.ones_like(std), std)
        return (cube - mean) / std


class TersGraphDataset(Dataset):
    """One TERS split: graph targets from the sidecar, conditioning cube from the HDF5.

    Deliberately *not* built on ``QM9SDataset``.  That class subclasses PyG
    ``InMemoryDataset``, which ``torch.load``s the whole corpus into RAM
    (datasets/qm9s_dataset.py:153), and wraps ``__getitem__`` in ``@lru_cache(maxsize=16)``
    (line 357) -- 16 cached TERS samples is ~1.7 GB *per worker*.  Both disqualifying.
    See docs/02-reuse-map.md 7.
    """

    cube_attr = 'ters'          # --- ADDED (stm): overridden by StmGraphDataset

    def __init__(self, h5_path, graphs_path, channels_key='channels_100',
                 transform=None, normalize=True, load_cubes=True):
        import h5py

        self.h5_path = h5_path
        self.graphs_path = graphs_path
        self.channels_key = channels_key
        self.transform = transform
        # --- ADDED (stm): accept a callable normaliser, not only a bool -------------------
        # STM z-stacks need a different transform (per-cube, not per-channel -- the TERS one
        # deletes their decay envelope; see models/stm_encoder.NormalizeStmCube). Accepting a
        # callable lets StmGraphDataset subclass this instead of copying 120 lines of
        # fork-safe HDF5 access. `True`/`False` behave exactly as before.
        if callable(normalize):
            self.normalize = normalize
        else:
            self.normalize = NormalizeVectorized() if normalize else None
        self.load_cubes = load_cubes

        # The sidecar holds RDKit Mol objects under 'rdmol', so it is not a weights-only
        # payload.  torch>=2.6 defaults weights_only=True and would refuse to load it.
        self.graphs = torch.load(graphs_path, weights_only=False)
        if not self.graphs:
            raise ValueError(f'{graphs_path} contains no records')

        self._hf = None

        if not load_cubes:
            # Graph-only mode. run_lib.py:339-341 walks the whole dataset purely to collect
            # `.rdmol` for the MOSES reference sets; reading a cube per molecule there is
            # tens of GB of I/O for data that is immediately discarded. In this mode the
            # HDF5 is never opened, so the sidecar alone is enough to construct the dataset.
            self.cube_shape = None
            self.in_channels = None
            return

        # Validate the cube key once here, in the parent process, rather than letting it
        # fail inside a DataLoader worker where the traceback is far less readable.
        with h5py.File(h5_path, 'r') as f:
            if channels_key not in f:
                raise KeyError(
                    f'{h5_path} has no dataset {channels_key!r}. '
                    f'Available top-level keys: {list(f.keys())}')
            shape = f[channels_key].shape

        if len(shape) != 4:
            raise ValueError(
                f'{channels_key} has shape {shape}; expected 4 dims (N, H, W, C)')
        self.cube_shape = tuple(shape[1:])          # (H, W, C) as stored
        self.in_channels = int(shape[3])            # feeds TersEncoder(in_channels=)

        worst = max(int(g['h5_index']) for g in self.graphs)
        if worst >= shape[0]:
            raise IndexError(
                f'sidecar references h5_index {worst} but {channels_key} holds only '
                f'{shape[0]} cubes -- {graphs_path} and {h5_path} are not the same split')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        data = Data(**g)

        if not self.load_cubes:
            # No `ters` attribute is set, so CollateTers would raise on this dataset -- which
            # is correct: a graph-only dataset must never be handed to a training loader.
            return data if self.transform is None else self.transform(data)

        # Opened on first touch, so the handle is created *inside* the worker.  An h5py
        # handle inherited across fork() is not safe to read from.
        if self._hf is None:
            import h5py
            self._hf = h5py.File(self.h5_path, 'r')

        cube = self._hf[self.channels_key][int(g['h5_index'])]      # (H, W, C)
        # num_nodes is inferred by PyG from pos, which EdgeComSpectraTransform relies on at
        # datasets/build_dataset.py:130.  Setting pos (done in the sidecar) is sufficient.
        ters = torch.from_numpy(np.asarray(cube)).float().permute(2, 0, 1).contiguous()
        # --- ADDED (stm): attribute name is a class constant ------------------------------
        # So a subclass can store the cube under its own name and CollateStm can read it.
        # Rejected reusing the name 'ters' for STM cubes: it would make CollateTers work
        # with zero changes, but every downstream `data.ters` would then be lying about
        # which modality it holds.
        setattr(data, self.cube_attr, ters if self.normalize is None
                else self.normalize(ters))

        return data if self.transform is None else self.transform(data)

    def __getstate__(self):
        # Never pickle a live HDF5 handle to a worker (spawn start method, or a parent that
        # already read a sample before the loader forked).
        state = self.__dict__.copy()
        state['_hf'] = None
        return state

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _smoke_test(h5_path, graphs_path, channels_key):
    """Load a few samples through the real transform and collate, and report shapes."""
    from .build_dataset import CollateTers, EdgeComSpectraTransform
    from .datasets_config import get_dataset_info

    info = get_dataset_info('qm9_second_half')
    transform = EdgeComSpectraTransform(info['atom_encoder'].values(),
                                        include_aromatic=False, use_normalize=True)
    ds = TersGraphDataset(h5_path, graphs_path, channels_key=channels_key,
                          transform=transform)
    print(f'{len(ds)} molecules   cube (H,W,C)={ds.cube_shape}   '
          f'in_channels={ds.in_channels}')

    sample = ds[0]
    print(f'sample 0: {sample.num_nodes} atoms  ters={tuple(sample.ters.shape)}  '
          f'mean={sample.ters.mean():+.3e}  std={sample.ters.std():.3f}')

    batch = CollateTers()([ds[i] for i in range(min(4, len(ds)))])
    for k, v in batch.items():
        print(f'  {k:15s} {tuple(v.shape)}')
    return ds, batch


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--channels-key', default='channels_100')
    a = ap.parse_args()
    _smoke_test(a.h5, a.graphs, a.channels_key)
