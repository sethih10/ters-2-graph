"""Build a synthetic TERS corpus so every phase can be exercised without the cluster.

The real HDF5 lives on another cluster (docs/05-progress-log.md D-1). This writes files with
the same *shape* as the real ones -- zero-padded atom arrays, an (N, H, W, C) cube block --
and plants the pathological cases we care about:

  * an all-zero channel and a constant channel  -> exercises the zero-std guard in
    NormalizeVectorized, which is where a silent NaN would come from
  * cube values on the real 0-2556 scale        -> a normalisation no-op would be visible
  * aromatic and charged molecules              -> exercises the kekulisation trap (D-4)
  * ragged atom counts within a split           -> exercises padding and masking

Usage:
    python -m tests.make_synthetic /path/to/outdir
"""

import os
import os.path as osp
import sys

import numpy as np

SMILES = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCCO', 'CC(C)N', 'c1ccncc1', 'CCC=O', 'C[N+](=O)[O-]']
SPLITS = {'train': SMILES, 'val': SMILES[:4], 'test': SMILES[4:]}

H = W = 8
N_CHANNELS = 100
DEAD_CHANNEL = 7          # all-zero
CONST_CHANNEL = 9         # constant, non-zero
RAW_MAX = 2556.0          # the real scale, per docs/02-reuse-map.md 5


def build(out_dir, h=H, w=W, n_channels=N_CHANNELS, seed=0):
    import h5py
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for split, smiles in SPLITS.items():
        mols = []
        for i, smi in enumerate(smiles):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            if AllChem.EmbedMolecule(m, randomSeed=i + 1) != 0:
                raise RuntimeError(f'could not embed {smi}')
            AllChem.MMFFOptimizeMolecule(m)
            mols.append(m)

        max_atoms = max(m.GetNumAtoms() for m in mols)
        z = np.zeros((len(mols), max_atoms), dtype=np.int32)
        pos = np.zeros((len(mols), max_atoms, 3), dtype=np.float32)
        for i, m in enumerate(mols):
            n = m.GetNumAtoms()
            z[i, :n] = [a.GetAtomicNum() for a in m.GetAtoms()]
            pos[i, :n] = m.GetConformer().GetPositions()

        rng = np.random.default_rng(seed)
        cubes = rng.random((len(mols), h, w, n_channels)).astype(np.float32) * RAW_MAX
        cubes[:, :, :, DEAD_CHANNEL] = 0.0
        cubes[:, :, :, CONST_CHANNEL] = 42.0

        path = osp.join(out_dir, f'{split}.h5')
        with h5py.File(path, 'w') as f:
            f['atomic_numbers'] = z
            f['atom_positions'] = pos
            f[f'channels_{n_channels}'] = cubes
        written.append(path)
        print(f'wrote {path}  {len(mols)} molecules, padded to {max_atoms} atoms')
    return written


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'tests/_synthetic')
