"""Pair the STM image cubes in the HDF5 with the graph sidecar from ``preprocess_stm.py``.

The STM corpus is stored the same way the TERS one is -- a big lazily-read cube array plus a
small in-RAM sidecar of graph targets carrying an ``h5_index`` back-pointer -- so this is a
thin subclass of :class:`~datasets.ters_dataset.TersGraphDataset` rather than a second
implementation.

Provenance (ladder in docs/README.md):
  * ``TersGraphDataset``  -- **REUSE** by subclassing.  The fork-safe lazy HDF5 open, the
        ``__getstate__`` handle-stripping, the sidecar/cube length validation and the
        ``load_cubes=False`` graph-only mode are inherited, not copied.  Those encode
        conventions the whole pipeline depends on, and a divergent second copy is a bug
        waiting to happen (the same argument ``CollateTers`` makes for subclassing
        ``CollateSpectra``).
  * ``NormalizeStmCube`` -- **REUSE**, imported from models/stm_encoder.py, so the encoder
        and the dataset cannot disagree about what normalisation the encoder was designed
        for.  It lives next to the encoder deliberately: the reason it exists is a property
        of the encoder's 3D stem.
  * key/split resolution, the three-way shape guard -- **NEW**.

Three things differ from TERS, and they are the whole content of this file:

1. **The cube lives one group down.**  TERS: ``f['channels_100']``.  STM: ``f['val']['X']``.
   h5py resolves ``'val/X'`` as a path, so the inherited key handling works unchanged once
   the split name is known -- and it is discoverable, so the caller does not have to pass it.
2. **The normaliser.**  ``NormalizeVectorized`` standardises each channel over ``(H, W)``,
   which on a z-stack flattens every slice to mean 0 and deletes the decay envelope along z.
   Measured, and asserted in ``models/stm_encoder --self-test``.  See that module.
3. **The attribute name.**  The cube is ``data.stm``, read by ``CollateStm``.

Usage
-----
    python -m datasets.stm_dataset \\
        --h5 /scratch/project_2001912/stm-data/val.h5 \\
        --graphs /scratch/project_2001912/stm-graphs/graphs_val.pt
"""

import torch

from models.stm_encoder import NormalizeStmCube
from .ters_dataset import TersGraphDataset

CUBE_KEY = 'X'


def resolve_cube_key(h5_path, split=None, cube_key=CUBE_KEY):
    """``'<split>/X'``, with the split discovered when not given.

    Done in the parent process on purpose (docs/06-code-practices.md 6): a wrong key inside
    a DataLoader worker surfaces as a mangled traceback from a subprocess, and the fix is
    always the same one sentence.
    """
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
        Group name inside the HDF5 (``'train'``/``'val'``/``'test'``).  Inferred from the
        file when omitted -- each STM file holds exactly one.
    normalize : bool or callable
        ``True`` (default) uses ``NormalizeStmCube``.  **Not** ``NormalizeVectorized``:
        see the module docstring.  Pass a callable to override, ``False`` to disable.
    """

    cube_attr = 'stm'

    def __init__(self, h5_path, graphs_path, split=None, transform=None, normalize=True,
                 load_cubes=True, cube_key=CUBE_KEY):
        channels_key = (resolve_cube_key(h5_path, split, cube_key)
                        if load_cubes else f'{split or "train"}/{cube_key}')
        if normalize is True:
            normalize = NormalizeStmCube()
        super().__init__(h5_path=h5_path, graphs_path=graphs_path,
                         channels_key=channels_key, transform=transform,
                         normalize=normalize, load_cubes=load_cubes)

        # The parent validates 4 dims and reads in_channels from shape[3]; for STM that is
        # the z-slice count, which is what StmEncoder(in_channels=) needs. Cross-check it
        # against the file's own record of the slices so a corpus regenerated with a
        # different z range cannot silently train a mismatched encoder.
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

    # The property the encoder depends on, checked on the real path rather than assumed.
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
