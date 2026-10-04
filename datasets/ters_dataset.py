"""Pair TERS image cubes (HDF5) with the graph sidecar written by ``preprocess_ters.py``.

Each sample is a PyG ``Data`` with the sidecar's graph fields and the cube as ``(C, H, W)``;
batches are collated by ``build_dataset.CollateTers``.

Usage
-----
    python -m datasets.ters_dataset --h5 .../val.h5 --graphs datasets/ters/graphs_val.pt
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class NormalizeVectorized:
    """Standardise each channel of a ``(C, H, W)`` cube over its spatial axes.

    Channels with std below ``eps`` are only mean-centred.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, cube):
        mean = cube.mean(dim=(1, 2), keepdim=True)
        std = cube.std(dim=(1, 2), keepdim=True)
        std =torch.where(std < self.eps, torch.ones_like(std), std)
        return (cube - mean) / std


class AddCubeNoise:
    """Add i.i.d. Gaussian noise to a normalised cube.

    Apply after normalisation, so ``std`` is a fraction of the per-channel signal std. The
    noise is seeded from ``seed`` and ``index``, so a molecule gets the same noise on every read.
    """

    def __init__(self, std, seed=0):
        self.std = float(std)
        self.seed = int(seed)

    def __call__(self, cube, index):
        if self.std <= 0.0:
            return cube
        gen =torch.Generator().manual_seed((self.seed * 1000003 + int(index)) & 0x7FFFFFFF)
        noise = torch.empty_like(cube).normal_(mean=0.0, std=self.std, generator=gen)
        return cube + noise

    def __repr__(self):
        return f'AddCubeNoise(std={self.std}, seed={self.seed})'


class TersGraphDataset(Dataset):
    """One TERS split: graph targets from the sidecar, conditioning cube from the HDF5."""

    cube_attr = 'ters'          # Data attribute that holds the cube

    def __init__(self, h5_path, graphs_path, channels_key='channels_100',
                 transform=None, normalize=True, load_cubes=True, noise=None):
        import h5py

        self.h5_path = h5_path
        self.graphs_path = graphs_path
        self.channels_key = channels_key
        self.transform = transform
        if callable(normalize):
            self.normalize = normalize
        else:
            self.normalize = NormalizeVectorized() if normalize else None
        # Callable (cube, index) -> cube applied after normalisation, e.g. AddCubeNoise.
        self.noise = noise
        self.load_cubes = load_cubes

        # The sidecar holds RDKit Mols, which weights_only=True cannot load.
        self.graphs = torch.load(graphs_path, weights_only=False)
        if not self.graphs:
            raise ValueError(f'{graphs_path} contains no records')

        self._hf = None

        if not load_cubes:
            self.cube_shape = None
            self.in_channels = None
            return

        # Validate here so errors surface outside DataLoader workers.
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
        self.in_channels = int(shape[3])

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
            # Graph-only sample: no cube, so it cannot go through CollateTers.
            return data if self.transform is None else self.transform(data)

        # Open lazily, inside the worker: h5py handles are not fork-safe.
        if self._hf is None:
            import h5py
            self._hf = h5py.File(self.h5_path, 'r')

        cube = self._hf[self.channels_key][int(g['h5_index'])]      # (H, W, C)
        ters = torch.from_numpy(np.asarray(cube)).float().permute(2, 0, 1).contiguous()
        cube_t = ters if self.normalize is None else self.normalize(ters)
        if self.noise is not None:
            cube_t = self.noise(cube_t, int(g['h5_index']))
        setattr(data, self.cube_attr, cube_t)

        return data if self.transform is None else self.transform(data)

    def __getstate__(self):
        # Never pickle a live HDF5 handle.
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
