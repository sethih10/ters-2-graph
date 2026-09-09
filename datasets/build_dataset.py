from torch_geometric.transforms import Compose, ToDevice
from .qm9s_dataset import QM9SDataset
from .datasets_config import get_dataset_info
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torch
from scipy.spatial.transform import Rotation
import numpy as np


prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11}


def get_dataset(config, transform=True):
    """Create dataset for training and evaluation."""

    # Obtain dataset info
    dataset_info = get_dataset_info(config.data.info_name)  # 'qm9_second_half' -> qm9_second_half object

    # get transform
    if transform:         # True
        name_transform = getattr(config.data, 'transform', 'EdgeComSpectra')   # 'EdgeComSpectra'
        if name_transform == 'EdgeComSpectra':
            transform = EdgeComSpectraTransform(dataset_info['atom_encoder'].values(), config.data.include_aromatic, use_normalize=config.data.use_normalize)
        elif name_transform == 'EdgeComCond':
            prop2idx = dataset_info['prop2idx']
            transform = EdgeComCondTransform(dataset_info['atom_encoder'].values(), config.data.include_aromatic,
                                                prop2idx[config.cond_property])
        else:
            raise ValueError('Invalid data transform name')
    else:
        transform = None

    # Build up dataset
    if config.data.name == 'TERS':
        return _build_ters(config, transform, dataset_info)

    # --- ADDED (stm): STM splits are on disk too, same five-tuple contract -----------------
    if config.data.name == 'STM':
        return _build_stm(config, transform, dataset_info)

    if config.data.name == 'QM9S':
        dataset = QM9SDataset(config.data.root, transform=transform, spectra_version=config.data.spectra_version)
    else:
        raise ValueError('Undefined dataset name.')

    # Split dataset
    if 'cond' in config.exp_type or 'diffspectra' in config.exp_type:
        split_idx = dataset.get_cond_idx_split()
        train_idx = np.concatenate([split_idx['first_train'], split_idx['second_train']])
        first_train_dataset = dataset.index_select(split_idx['first_train'])
        second_train_dataset = dataset.index_select(train_idx)
        val_dataset = dataset.index_select(split_idx['valid'])
        test_dataset = dataset.index_select(split_idx['test'])
        return first_train_dataset, second_train_dataset, val_dataset, test_dataset, dataset_info

    split_idx = dataset.get_idx_split()
    train_dataset = dataset.index_select(split_idx['train'])
    val_dataset = dataset.index_select(split_idx['valid'])
    test_dataset = dataset.index_select(split_idx['test'])

    return train_dataset, val_dataset, test_dataset, dataset_info


def _build_ters(config, transform, dataset_info):
    """Phase 2 -- three TersGraphDataset splits, one per HDF5 file.

    Returns the five-tuple the ``'diffspectra'`` path expects, with the train set in both
    slots: ``first_train`` feeds only the unconditional pre-training branch, and ``run_lib``
    consumes ``second_train``.  Splitting is already done on disk (one HDF5 + one sidecar
    per split), so this deliberately bypasses ``index_select`` and ``get_cond_idx_split`` --
    those are ``InMemoryDataset`` APIs that a lazily-read HDF5 dataset does not have.

    Config keys (see configs/):
        data.root        directory holding {train,val,test}.h5
        data.graph_root  directory holding graphs_{split}.pt   (default: data.root)
        data.ters_key    cube dataset name                     (default: channels_100)
        data.load_cubes  read the image cubes at all           (default: True)

    ``load_cubes=False`` is for the auxiliary config passed as ``--config_original_qm9``,
    whose datasets are consumed only for ``.rdmol`` (run_lib.py:339-341). See configs/base_ters.py.
    """
    import os.path as osp
    from .ters_dataset import TersGraphDataset

    root = config.data.root
    graph_root = getattr(config.data, 'graph_root', root)
    channels_key = getattr(config.data, 'ters_key', 'channels_100')
    load_cubes = getattr(config.data, 'load_cubes', True)

    def split(name):
        return TersGraphDataset(
            h5_path=osp.join(root, f'{name}.h5'),
            graphs_path=osp.join(graph_root, f'graphs_{name}.pt'),
            channels_key=channels_key,
            transform=transform,
            load_cubes=load_cubes,
        )

    train, val, test = split('train'), split('val'), split('test')
    return train, train, val, test, dataset_info


def _build_stm(config, transform, dataset_info):
    """--- ADDED (stm): three StmGraphDataset splits, one per HDF5 file.

    Same five-tuple contract and the same reasoning as ``_build_ters`` above: splitting is
    already done on disk (one HDF5 + one sidecar per split), so ``index_select`` and
    ``get_cond_idx_split`` -- ``InMemoryDataset`` APIs a lazily-read HDF5 dataset does not
    have -- are deliberately bypassed.

    Config keys (see configs/diffspectra_stm.py):
        data.root        directory holding {train,val,test}.h5
        data.graph_root  directory holding graphs_{split}.pt   (default: data.root)
        data.stm_key     cube dataset name inside the split group  (default: X)
        data.load_cubes  read the image cubes at all           (default: True)
    """
    import os.path as osp
    from .stm_dataset import StmGraphDataset

    root = config.data.root
    graph_root = getattr(config.data, 'graph_root', root)
    cube_key = getattr(config.data, 'stm_key', 'X')
    load_cubes = getattr(config.data, 'load_cubes', True)

    def split(name):
        return StmGraphDataset(
            h5_path=osp.join(root, f'{name}.h5'),
            graphs_path=osp.join(graph_root, f'graphs_{name}.pt'),
            # Passed explicitly rather than inferred: the group inside train.h5 happens to
            # be named 'train', but relying on that would break silently on a file whose
            # group is named differently, and the caller always knows the split here.
            split=name,
            transform=transform,
            load_cubes=load_cubes,
            cube_key=cube_key,
        )

    train, val, test = split('train'), split('val'), split('test')
    return train, train, val, test, dataset_info


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# setup dataloader
def get_dataloader(train_ds, val_ds, test_ds, config):
    if config.model.name == 'DMT':
        aug_rotation = False
        aug_translation = False
    else:
        raise ValueError(f"Model {config.model.name} not recognized.")

    is_ters = config.data.name == 'TERS'
    # --- ADDED (stm): STM takes the same cube path as TERS, different collate --------------
    is_stm = config.data.name == 'STM'

    if is_stm:
        if config.only_2D:
            raise ValueError('only_2D has no STM collate; the z-stack conditions 3D positions.')
        collate_fn = CollateStm(
            aug_rotation=aug_rotation,
            aug_translation=aug_translation,
            aug_translation_scale=config.data.aug_translation_scale
        )
    elif is_ters:
        if config.only_2D:
            raise ValueError('only_2D has no TERS collate; the cube conditions 3D positions.')
        collate_fn = CollateTers(
            aug_rotation=aug_rotation,
            aug_translation=aug_translation,
            aug_translation_scale=config.data.aug_translation_scale
        )
    elif config.only_2D:
        collate_fn = CollateSpectr2D(
            spectra_version=config.data.spectra_version,
        )
    else:
        collate_fn = CollateSpectra(
            spectra_version=config.data.spectra_version,
            aug_rotation=aug_rotation,
            aug_translation=aug_translation,
            aug_translation_scale=config.data.aug_translation_scale
        )

    # A TERS sample carries a whole image cube (256x256x100 float32 = 26.2 MB), not three
    # 1-D spectra, so in-flight RAM is roughly
    #     num_workers x prefetch_factor x batch_size x 26.2 MB
    # This used to be hard-clamped to 2 workers on the assumption that RAM was scarce. It is
    # not: a Roihu GH200 node gives ~211 GiB per reserved GPU, and measurement on gputest
    # (job 923883) showed the loader -- not the GPU -- is the bottleneck. Throughput was flat
    # at ~25 molecules/s from batch 16 to 256 while data-fetch time grew to 10x step time,
    # i.e. the GPU sat idle waiting for cubes. So honour config.data.num_workers instead of
    # overriding it, and keep prefetch_factor=1 so the formula above stays easy to reason
    # about. Budget it before raising: at batch 64 and 16 workers this holds ~27 GiB.
    loader_kwargs = {}
    # --- ADDED (stm): STM deliberately takes the `else` branch below ----------------------
    # Not an oversight. The prefetch_factor=1 clamp above exists because a TERS cube is
    # 256x256x100 float32 = 26.2 MB; an STM z-stack is 128x128x14 = 917 KB, 28x smaller, so
    # at batch 128 / 8 workers / default prefetch 2 the in-flight cubes are ~1.9 GiB of the
    # ~211 GiB a reserved GH200 gets. Clamping prefetch there would throttle the loader for
    # no reason. STM still honours config.data.num_workers, because `else` reads it directly.
    if is_ters:
        num_workers = getattr(config.data, 'num_workers', 2)
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = 1
    else:
        num_workers = config.data.num_workers

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                                num_workers=num_workers, collate_fn=collate_fn, drop_last=config.training.dataloader_drop_last, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, **loader_kwargs)
    return train_loader, val_loader, test_loader


# transform data
class EdgeComSpectraTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, use_normalize=True):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.use_normalize = use_normalize

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        # edge feature channel [edge_exist; bond_order; aromatic_exist]
        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        # log(x+1) normalization for molecular spectral data
        if self.use_normalize:
            if hasattr(data, 'ir'):
                data.ir = torch.log10(data.ir + 1)
            if hasattr(data, 'uv'):
                data.uv = torch.log10(data.uv + 1)
            if hasattr(data, 'raman'):
                data.raman = torch.log10(data.raman + 1)
        return data


class EdgeComCondTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx = property_idx

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        properties = data.y
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

            data.property = properties[0, self.property_idx:self.property_idx+1] - \
                            torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
            property = properties[0, self.property_idx:self.property_idx+1]  # 从 data.y 中取出对应的那一列作为 data.property
            data.property = property

        return data


def pad_node_feature(x, pad_len):
    x_len, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, x_dim], dtype=x.dtype)
        new_x[:x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_feature(x, pad_len):
    # x: [N_node, N_node, ch]
    x_len, _, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, pad_len, x_dim])
        new_x[:x_len, :x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def get_node_mask(node_num, pad_len, dtype):
    node_mask = torch.zeros(pad_len, dtype=dtype)
    node_mask[:node_num] = 1.
    return node_mask.unsqueeze(0)


# collate function: padding with the max node

def collate_node(items):
    items = [(item.one_hot, item.pos, item.fc, item.num_atom) for item in items]
    one_hot, positions, formal_charges, num_atoms = zip(*items)
    max_node_num = max(num_atoms)

    # padding features
    one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in one_hot])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        one_hot=one_hot,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        positions=positions,
        formal_charges=formal_charges
    )


def collate_edge(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_edge_2D(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_cond(items):
    # collate_fn for the condition generation
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom, item.property) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, property = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # context property
    property = torch.stack(property, dim=0)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        context=property
    )

class CollateSpectra(object):
    def __init__(self, spectra_version='allspectra', aug_rotation=False, aug_translation=False, aug_translation_scale=0.01):
        self.spectra_version = spectra_version
        self.aug_rotation = aug_rotation
        self.aug_translation = aug_translation
        self.aug_translation_scale = aug_translation_scale

    def augment_positions(self, positions, atom_mask):
        batch_size = positions.shape[0]
        num_max_nodes = positions.shape[1]
        dtype = positions.dtype
        
        # Create mask to distinguish between actual atoms and padding positions
        mask = atom_mask.unsqueeze(-1)  # [batch_size, num_nodes, 1]
        
        if self.aug_rotation:
            rot_aug = Rotation.random(batch_size)
            # Only rotate actual atom positions
            positions_np = positions.numpy()
            # Rotate each molecule
            for i in range(batch_size):
                positions_np[i] = rot_aug[i].apply(positions_np[i])
            positions = torch.from_numpy(positions_np).to(dtype)
            # Keep padding positions as 0
            positions = positions * mask
            
        if self.aug_translation:
            # Only translate actual atom positions
            trans_aug = self.aug_translation_scale * torch.randn(batch_size, 1, 3, dtype=dtype).repeat(1, num_max_nodes, 1)
            positions = positions + trans_aug
            # Keep padding positions as 0
            positions = positions * mask
            
        return positions

    def pad_and_mask(self, atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms):
        max_node_num = max(num_atoms)

        # padding features
        atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
        formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
        positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
        edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

        # atom mask
        node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

        # edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.reshape(-1, 1)

        return atom_one_hot, edge_one_hot, formal_charges, positions, node_mask, edge_mask

    def __call__(self, items):
        # Uniformly extract all data
        items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom, 
                 item.uv, item.ir, item.raman) for item in items]
        atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, uv, ir, raman = zip(*items)

        # Use base class pad_and_mask method
        atom_one_hot, edge_one_hot, formal_charges, positions, node_mask, edge_mask = self.pad_and_mask(
            atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms
        )

        # Construct spectra based on spectra_version
        if self.spectra_version == 'allspectra':
            spectra = [torch.stack(uv, dim=0), torch.stack(ir, dim=0), torch.stack(raman, dim=0)]
        elif self.spectra_version == 'ir':
            spectra = torch.stack(ir, dim=0)
        elif self.spectra_version == 'uv':
            spectra = torch.stack(uv, dim=0)
        elif self.spectra_version == 'raman':
            spectra = torch.stack(raman, dim=0)
        else:
            raise ValueError(f'Invalid spectra version: {self.spectra_version}')

        # Data augmentation
        positions = self.augment_positions(positions, node_mask)

        return dict(
            atom_one_hot=atom_one_hot,
            edge_one_hot=edge_one_hot,
            positions=positions,
            formal_charges=formal_charges,
            atom_mask=node_mask,
            edge_mask=edge_mask,
            context=spectra
        )


class CollateTers(CollateSpectra):
    """TERS cube conditioning.  ADAPT of CollateSpectra by subclassing (docs/02-reuse-map.md 3).

    Only the context differs: one dense ``(C, H, W)`` cube per molecule replaces the
    uv/ir/raman triple, so the four-way ``spectra_version`` branch collapses to a single
    stack.  ``pad_and_mask`` (lines 408-426) and ``augment_positions`` (lines 380-406) are
    *inherited, not copied* -- they encode the mask conventions the whole pipeline depends
    on (the diagonal exclusion, the ``reshape(-1, 1)``), and a divergent second copy is a
    bug waiting to happen.
    """

    def __init__(self, aug_rotation=False, aug_translation=False, aug_translation_scale=0.01):
        # spectra_version is meaningless here; pin it so the inherited __init__ signature
        # stays satisfied without the caller having to pass a dead argument.
        super().__init__(spectra_version=None, aug_rotation=aug_rotation,
                         aug_translation=aug_translation,
                         aug_translation_scale=aug_translation_scale)

    # --- ADDED (stm): which attribute holds the cube -------------------------------------
    # A class constant rather than a constructor argument: it is a property of the modality,
    # not of a run, and get_dataloader picks the class already.
    cube_attr = 'ters'

    def __call__(self, items):
        items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom,
                  getattr(item, self.cube_attr)) for item in items]
        atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, ters = zip(*items)

        atom_one_hot, edge_one_hot, formal_charges, positions, node_mask, edge_mask = \
            self.pad_and_mask(atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms)

        positions = self.augment_positions(positions, node_mask)

        return dict(
            atom_one_hot=atom_one_hot,
            edge_one_hot=edge_one_hot,
            positions=positions,
            formal_charges=formal_charges,
            atom_mask=node_mask,
            edge_mask=edge_mask,
            context=torch.stack(ters, dim=0)
        )


class CollateStm(CollateTers):
    """--- ADDED (stm): STM z-stack conditioning.

    Identical to ``CollateTers`` but for which attribute carries the cube, because the
    contract downstream is identical: one dense ``(C, H, W)`` tensor per molecule stacked
    into ``context``. Subclassed rather than copied for the same reason ``CollateTers``
    subclasses ``CollateSpectra`` -- ``pad_and_mask`` and ``augment_positions`` encode the
    mask conventions the whole pipeline depends on.

    .. note::
       ``aug_rotation`` must stay False for STM. The cube is a *constant-height* stack over
       a fixed surface, so rotating the 3D coordinates without rotating the image stack
       would break the correspondence the conditioning relies on. ``get_dataloader`` sets it
       False for every DMT run, so this is a note rather than a guard -- but it is the first
       thing to check if augmentation is ever turned on.
    """

    cube_attr = 'stm'


class CollateSpectr2D(object):
    def __init__(self, spectra_version='allspectra'):
        self.spectra_version = spectra_version

    def pad_and_mask(self, atom_one_hot, edge_one_hot, formal_charges, num_atoms):
        max_node_num = max(num_atoms)

        # padding features
        atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
        formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
        edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

        # atom mask
        node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

        # edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.reshape(-1, 1)

        return atom_one_hot, edge_one_hot, formal_charges, node_mask, edge_mask

    def __call__(self, items):
        # Uniformly extract all data
        items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.num_atom, 
                    item.uv, item.ir, item.raman) for item in items]
        atom_one_hot, edge_one_hot, formal_charges, num_atoms, uv, ir, raman = zip(*items)

        # Use base class pad_and_mask method
        atom_one_hot, edge_one_hot, formal_charges, node_mask, edge_mask = self.pad_and_mask(
            atom_one_hot, edge_one_hot, formal_charges, num_atoms
        )

        # Construct spectra based on spectra_version
        if self.spectra_version == 'allspectra':
            spectra = [torch.stack(uv, dim=0), torch.stack(ir, dim=0), torch.stack(raman, dim=0)]
        elif self.spectra_version == 'ir':
            spectra = torch.stack(ir, dim=0)
        elif self.spectra_version == 'uv':
            spectra = torch.stack(uv, dim=0)
        elif self.spectra_version == 'raman':
            spectra = torch.stack(raman, dim=0)
        else:
            raise ValueError(f'Invalid spectra version: {self.spectra_version}')


        return dict(
            atom_one_hot=atom_one_hot,
            edge_one_hot=edge_one_hot,
            formal_charges=formal_charges,
            atom_mask=node_mask,
            edge_mask=edge_mask,
            context=spectra
        )