"""Sampling functions that draw conditions from a dataset and return processed molecules.

Moved from DiffSpectra sampling.py.
"""

import numpy as np
import torch

from models.utils import (sample_combined_position_feature_noise, assert_mean_zero_with_mask,
                          sample_symmetric_edge_feature_noise)
from datasets import stack_context
from utils import context_to, get_self_cond_fn

from .postprocess import mol_process, post_process
from .samplers import AncestralSampler


def get_sampling_fn(config, noise_scheduler, batch_size, n_samples, inverse_scaler, val_ds, eps=1e-3):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    image_attr = getattr(val_ds, 'image_attr', None)  # 'image' for ImageGraphDataset

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, self_cond,
                                   get_self_cond_fn(config))
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model):
        model.eval()
        processed_mols = []
        sampled_val_pos = []
        sampled_val_rdkit_mols = []
        with torch.no_grad():            
            # Modified sampling process, sample spectra from test set as condition, sample spectra first, and get corresponding real atom count
            num_mol_val = len(val_ds)
            permute_val_mol_id = torch.randperm(num_mol_val)

            for r in range(num_sampling_rounds):
                sampled_val_mol_id= permute_val_mol_id[r * batch_size:(r + 1) * batch_size]
                sampled_val_mol = []
                n_nodes = []
                for i in sampled_val_mol_id:
                    mol = val_ds[i]
                    sampled_val_mol.append(mol)
                    sampled_val_pos.append(mol.pos)
                    sampled_val_rdkit_mols.append(mol.rdmol)
                    n_nodes.append(int(mol.num_atom))
                
                if image_attr is not None:
                    context = context_to(stack_context(sampled_val_mol, image_attr), device)
                max_n_nodes = max(n_nodes)
                n_batch = len(n_nodes)
                

                # construct node and edge mask
                node_mask = torch.zeros(n_batch, max_n_nodes)
                for i in range(n_batch):
                    node_mask[i, 0:n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(n_batch * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                # sample initial noise
                z = sample_combined_position_feature_noise(n_batch, max_n_nodes, node_nf, node_mask)
                assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                # sample initial edge noise
                if pred_edge:
                    edge_z = sample_symmetric_edge_feature_noise(n_batch, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
                    # postprocessing
                    pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)
                else:
                    # sampling procedure
                    x_node = sampler.sampling(model, z, node_mask, edge_mask)
                    # postprocessing: split features and discretize and checking, and inverse
                    pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                assert_mean_zero_with_mask(pos, node_mask)

                # process tensors
                if pred_edge:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                else:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                print('Generate {}, Total {}.'.format(len(processed_mols), n_samples))

        # shuffle mols and pick n_samples
        # random.shuffle(processed_mols)
        # print('debug:', len(processed_mols), len(sampled_val_pos))
        # Save ground truth mol for coordinate error calculation and visualization
        return processed_mols[:n_samples], sampled_val_pos[:n_samples], sampled_val_rdkit_mols[:n_samples]

    return sampling_fn



def get_cond_sampling_eval_fn(config, noise_scheduler, batch_size, n_samples, inverse_scaler, test_ds, eps=1e-3):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    image_attr = getattr(test_ds, 'image_attr', None)  # 'image' for ImageGraphDataset

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, 
                                    self_cond, get_self_cond_fn(config), 
                                    sampling_temperature=config.eval.sampling_temperature)
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model):
        model.eval()
        processed_mols = []
        sampled_test_pos = []
        sampled_test_rdkit_mols = []

        with torch.no_grad():            
            num_mol_test = len(test_ds)
            # Fix random seed to ensure different models in each experiment are based on the same spectral sampling
            torch.manual_seed(42)
            permute_test_mol_id = torch.randperm(num_mol_test)

            for r in range(num_sampling_rounds):
                sampled_test_mol_id= permute_test_mol_id[r * batch_size:(r + 1) * batch_size]
                sampled_test_mol = []
                n_nodes = []
                for i in sampled_test_mol_id:
                    mol = test_ds[i]
                    sampled_test_mol.append(mol)
                    sampled_test_pos.append(mol.pos)
                    sampled_test_rdkit_mols.append(mol.rdmol)
                    n_nodes.append(int(mol.num_atom))
                
                if image_attr is not None:
                    context = context_to(stack_context(sampled_test_mol, image_attr), device)
                max_n_nodes = max(n_nodes)
                n_batch = len(n_nodes)

                # construct node and edge mask
                node_mask = torch.zeros(n_batch, max_n_nodes)
                for i in range(n_batch):
                    node_mask[i, 0:n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(n_batch * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                # sample initial noise
                z = sample_combined_position_feature_noise(n_batch, max_n_nodes, node_nf, node_mask)
                assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                # sample initial edge noise
                if pred_edge:
                    edge_z = sample_symmetric_edge_feature_noise(n_batch, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
                    # postprocessing
                    pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)
                else:
                    # sampling procedure
                    x_node = sampler.sampling(model, z, node_mask, edge_mask)
                    # postprocessing: split features and discretize and checking, and inverse
                    pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                assert_mean_zero_with_mask(pos, node_mask)

                if pred_edge:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                else:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                print('Generate {}, Total {}.'.format(len(processed_mols), n_samples))

        # Save ground-truth mol for coordinate error calculation and visualization
        return processed_mols[:n_samples], sampled_test_pos[:n_samples], sampled_test_rdkit_mols[:n_samples]

    return sampling_fn


def sample_batch(model, sampler, context, n_nodes, config, inverse_scaler):
    """Sample one molecule per entry of ``n_nodes``, conditioned on ``context``.

    Returns ``(pos, one_hot, fc, edge_types)`` from ``post_process``. Requires ``config.pred_edge``.
    """
    if not config.pred_edge:
        raise ValueError('sample_batch requires config.pred_edge=True')
    device = config.device
    n_batch, max_n = len(n_nodes), max(n_nodes)
    node_nf = config.data.atom_types + int(config.model.include_fc_charge)

    node_mask = torch.zeros(n_batch, max_n)
    for i, n in enumerate(n_nodes):
        node_mask[i, :n] = 1
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    edge_mask *= ~torch.eye(max_n, dtype=torch.bool).unsqueeze(0)
    edge_mask = edge_mask.view(n_batch * max_n * max_n, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    with torch.no_grad():
        z = sample_combined_position_feature_noise(n_batch, max_n, node_nf, node_mask)
        assert_mean_zero_with_mask(z[:, :, :3], node_mask)
        edge_z = sample_symmetric_edge_feature_noise(n_batch, max_n, config.model.edge_ch, edge_mask)
        x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
        return post_process(x_node, config.data.atom_types, config.model.include_fc_charge,
                            node_mask, inverse_scaler, x_edge, edge_mask, config.data.compress_edge)
