"""Pair STM image cubes (HDF5) with the graph sidecar written by ``preprocess_stm.py``.

``StmGraphDataset`` is a ``TersGraphDataset`` that reads the z-stack from ``'<split>/X'``,
normalises it with ``NormalizeStmCube`` and stores it as ``data.stm`` for ``CollateStm``.

Usage
-----
    python -m datasets.stm_dataset \\
        --h5 $DATA_ROOT/stm-data/val.h5 \\
        --graphs $DATA_ROOT/stm-graphs/graphs_val.pt
"""

import torch

from models.stm_encoder import NormalizeStmCube
from .ters_dataset import TersGraphDataset

CUBE_KEY = 'X'


def resolve_cube_key(h5_path, split=None, cube_key=CUBE_KEY):
    """Return the cube dataset path ``'<split>/X'``, inferring the split when not given."""
    import h5py

    with h5py.File(h5_path, 'r') as f:
        groups = [k for k in f.keys() if hasattr(f[k], 'keys')]
        if split is None:
            if cube_key in f:                       # flat layout, no split group
                return cube_key
            if len(groups) != 1:
                raise KeyError(
                    f'{h5_path}: cannot infer the split group, found {groups}. '
                    f'Pass split= explicitly.')
            split = groups[0]
        key = f'{split}/{cube_key}'
        if key not in f:
            raise KeyError(
                f'{h5_path} has no dataset {key!r}. Top-level keys: {list(f.keys())}; '
                f'group {split!r} holds '
                f'{list(f[split].keys()) if split in f else "<missing>"}.')
        return key


class StmGraphDataset(TersGraphDataset):
    """One STM split: graph targets from the sidecar, conditioning z-stack from the HDF5.

    Parameters
    ----------
    h5_path, graphs_path
        As ``TersGraphDataset``.
    split : str, optional
        Group name inside the HDF5; inferred when the file holds only one.
    normalize : bool or callable
        ``True`` uses ``NormalizeStmCube``; not ``NormalizeVectorized``, which removes the
        decay along z. A callable overrides it, ``False`` disables it.
    noise : callable, optional
        As ``TersGraphDataset``, applied after ``normalize``.
    """

    cube_attr = 'stm'

    def __init__(self, h5_path, graphs_path, split=None, transform=None, normalize=True,
                 load_cubes=True, cube_key=CUBE_KEY, noise=None):
        channels_key = (resolve_cube_key(h5_path, split, cube_key)
                        if load_cubes else f'{split or "train"}/{cube_key}')
        if normalize is True:
            normalize = NormalizeStmCube()
        super().__init__(h5_path=h5_path, graphs_path=graphs_path,
                         channels_key=channels_key, transform=transform,
                         normalize=normalize, load_cubes=load_cubes, noise=noise)

        # in_channels is the z-slice count; check it against the file's z_slices attribute.
        if load_cubes:
            self.n_z_slices = self.in_channels
            self._check_z_slices(h5_path, split)

    def _check_z_slices(self, h5_path, split):
        import h5py

        with h5py.File(h5_path, 'r') as f:
            group = split or self.channels_key.split('/')[0]
            attrs = f[group].attrs if group in f else {}
            z_slices = attrs.get('z_slices')
        if z_slices is not None and len(z_slices) != self.in_channels:
            raise ValueError(
                f'{h5_path}: attrs["z_slices"] lists {len(z_slices)} heights but the cube '
                f'has {self.in_channels} channels. One of them is stale; the encoder would '
                f'be built for the wrong depth.')
        self.z_slices = None if z_slices is None else list(z_slices)


def _smoke_test(h5_path, graphs_path, split=None):
    """Load a few samples through the real transform and collate, and report shapes."""
    from .build_dataset import CollateStm, EdgeComSpectraTransform
    from .datasets_config import get_dataset_info

    info = get_dataset_info('stm_10elem')
    transform = EdgeComSpectraTransform(info['atom_encoder'].values(),
                                        include_aromatic=False, use_normalize=True)
    ds = StmGraphDataset(h5_path, graphs_path, split=split, transform=transform)
    print(f'{len(ds)} molecules   cube (H,W,C)={ds.cube_shape}   '
          f'in_channels={ds.in_channels}   z_slices={ds.z_slices}')

    sample = ds[0]
    print(f'sample 0: {sample.num_nodes} atoms  stm={tuple(sample.stm.shape)}  '
          f'mean={sample.stm.mean():+.3e}  std={sample.stm.std():.3f}')

    batch = CollateStm()([ds[i] for i in range(min(4, len(ds)))])
    for k, v in batch.items():
        print(f'  {k:15s} {tuple(v.shape)}')

    # Per-slice means should decrease along z after normalisation.
    per_slice = batch['context'].mean(dim=(0, 2, 3))
    print(f'  per-slice mean {per_slice[0]:+.3f} -> {per_slice[-1]:+.3f}  '
          f'monotone={bool(torch.all(per_slice[:-1] > per_slice[1:]))}')
    return ds, batch


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--split', default=None)
    a = ap.parse_args()
    _smoke_test(a.h5, a.graphs, a.split)
