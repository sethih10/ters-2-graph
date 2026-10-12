"""Convert sampled tensors into atom types, coordinates, formal charges and bonds.

Moved unchanged from DiffSpectra sampling.py.
"""

import torch
from torch.nn import functional as F


def mol_process(one_hot, x, formal_charges, n_nodes, edge_types=None):
    """Convert tensor to mols"""
    mol_list = []
    bs = one_hot.shape[0]
    for i in range(bs):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:n_nodes[i]]
        pos = pos[0:n_nodes[i]]
        if edge_types is not None:
            edge_type = edge_types[i][:n_nodes[i], :n_nodes[i]].cpu().detach()
            if formal_charges.shape[-1] != 0:
                fc = formal_charges[i][:n_nodes[i], 0].long().cpu().detach()
            else:
                fc = formal_charges[i][:n_nodes[i]].cpu().detach()
            mol_list.append((pos, atom_type, edge_type, fc))
        else:
            mol_list.append((pos, atom_type))

    return mol_list



def mol_process_2D(one_hot, formal_charges, n_nodes, edge_types=None):
    """Convert tensor to mols, without 3D position."""
    mol_list = []
    bs = one_hot.shape[0]
    for i in range(bs):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        atom_type = atom_type[0:n_nodes[i]]

        edge_type = edge_types[i][:n_nodes[i], :n_nodes[i]].cpu().detach()
        if formal_charges.shape[-1] != 0:
            fc = formal_charges[i][:n_nodes[i], 0].long().cpu().detach()
        else:
            fc = formal_charges[i][:n_nodes[i]].cpu().detach()
        mol_list.append((None, atom_type, edge_type, fc))

    return mol_list



def post_process(xh, atom_types, include_charge, node_mask, inverse_scaler,
                 edge_x=None, edge_mask=None, compress_edge=False):
    """Split the xh [bs, n_nodes, pos_dim+atom_types+fc_charge], unormalize data"""
    pos = xh[:, :, :3]
    if include_charge:
        h_int = xh[:, :, -1:]
        h_cat = xh[:, :, 3:-1]
    else:
        h_int = torch.zeros(0).to(xh.device)
        h_cat = xh[:, :, 3:]

    assert h_cat.shape[-1] == atom_types

    if edge_x is not None:
        pos, h_cat, h_int, h_edge = inverse_scaler(pos, h_cat, h_int, node_mask, edge_x, edge_mask)
    else:
        pos, h_cat, h_int = inverse_scaler(pos, h_cat, h_int, node_mask)
    h_cat = F.one_hot(torch.argmax(h_cat, dim=2), atom_types) * node_mask
    h_int = torch.round(h_int).long() * node_mask
    if edge_x is not None:
        if compress_edge:
            edge_exist = h_edge[:, :, :, 0]
            edge_exist[edge_exist < 0.5] = 0.
            edge_exist[edge_exist >= 0.5] = 1.0
            edge_type = h_edge[:, :, :, 1] * 3.
            edge_type[edge_type >= 2.5] = 3.
            edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.
            edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.
            edge_type[edge_type < 0.5] = 0.
            edge_type = edge_exist * edge_type
            if h_edge.size(-1) == 3:
                edge_aromatic = h_edge[:, :, :, 2]
                edge_aromatic[edge_aromatic < 0.5] = 0.
                edge_aromatic[edge_aromatic >= 0.5] = 1.
                edge_aromatic = edge_exist * edge_aromatic
                edge_type[torch.bitwise_and(edge_aromatic > 0., edge_type == 0.)] = 4.
            h_edge = edge_type
        else:
            # all 0 set non-exist, others set argmax
            h_edge_exist = torch.sum(h_edge > 0.5, dim=-1) != 0
            h_edge = torch.argmax(h_edge, dim=-1) + 1.0
            h_edge = h_edge_exist * h_edge

        return pos, h_cat, h_int, h_edge
    return pos, h_cat, h_int



def post_process_2D(xh, atom_types, include_charge, node_mask, inverse_scaler,
                    edge_x=None, edge_mask=None, compress_edge=False):
    """Split the xh [bs, n_nodes, pos_dim+atom_types+fc_charge], unormalize data"""
    if include_charge:
        h_int = xh[:, :, -1:]
        h_cat = xh[:, :, :-1]
    else:
        h_int = torch.zeros(0).to(xh.device)
        h_cat = xh[:, :, :]

    assert h_cat.shape[-1] == atom_types
    assert edge_x is not None

    _, h_cat, h_int, h_edge = inverse_scaler(None, h_cat, h_int, node_mask, edge_x, edge_mask)

    h_cat = F.one_hot(torch.argmax(h_cat, dim=2), atom_types) * node_mask
    h_int = torch.round(h_int).long() * node_mask

    if compress_edge:
        edge_exist = h_edge[:, :, :, 0]
        edge_exist[edge_exist < 0.5] = 0.
        edge_exist[edge_exist >= 0.5] = 1.0
        edge_type = h_edge[:, :, :, 1] * 3.
        edge_type[edge_type >= 2.5] = 3.
        edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.
        edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.
        edge_type[edge_type < 0.5] = 0.
        edge_type = edge_exist * edge_type
        if h_edge.size(-1) == 3:
            edge_aromatic = h_edge[:, :, :, 2]
            edge_aromatic[edge_aromatic < 0.5] = 0.
            edge_aromatic[edge_aromatic >= 0.5] = 1.
            edge_aromatic = edge_exist * edge_aromatic
            edge_type[torch.bitwise_and(edge_aromatic > 0., edge_type == 0.)] = 4.
        h_edge = edge_type
    else:
        # all 0 set non-exist, others set argmax
        h_edge_exist = torch.sum(h_edge > 0.5, dim=-1) != 0
        h_edge = torch.argmax(h_edge, dim=-1) + 1.0
        h_edge = h_edge_exist * h_edge

    return h_cat, h_int, h_edge
