# Copied from https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/data/augmentations.py
# (Apache-2.0): ChnSelector, ListChnMask and ChnPad, unchanged.
from random import shuffle

import numpy as np
import torch
from torch import Tensor


class ChnSelector:
    """ abstract class to implement transforms that subselect channels of a data object """

    def _get_indices(self, data_obj):
        chn_indices = [(s,i) for s in range(len(data_obj['imgs'])) 
                    for i in range(len(data_obj['imgs'][s]))]
        return chn_indices

    def _apply_selection(self, data_obj, chn_indices):
        keys = data_obj.keys()
        data_obj_out = {k: [] for k in keys}
        for chn_group_in in range(len(data_obj['imgs'])):
            idx = [i for s,i in chn_indices if s == chn_group_in]
            if len(idx) > 0:
                for k in keys:
                    data_obj_out[k].append(data_obj[k][chn_group_in][idx])
        return data_obj_out

class ListChnMask(ChnSelector):
    """ randomly mask out channels """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, data_obj):
        chn_indices = self._get_indices(data_obj)

        high = min(self.high, len(chn_indices))
        low = self.low

        if len(chn_indices) <= low:
            return data_obj

        nchns = np.random.randint(low, high+1)
        if nchns == len(chn_indices):
            return data_obj

        shuffle(chn_indices)
        chn_indices = chn_indices[:nchns]
        return self._apply_selection(data_obj, chn_indices)

class ChnPad:
    """ pads channels to same size """
    def __init__(self, size):
        self.size = size

    def __call__(self, x_dict):
        if self.size < 0:
            return x_dict
        img: Tensor = x_dict['imgs']
        chn_ids = x_dict['chn_ids']
        add_kwargs = {k:v for k,v in x_dict.items() if k not in ['imgs', 'chn_ids']}
        device = img.device

        c, h, w = img.shape
        spec_masks = torch.zeros(c, device=device)
        if c < self.size:
            img = torch.cat([img, torch.zeros(self.size - c, h, w, device=device)])
            # Check the number of dimensions
            if chn_ids.ndimension() == 1:
                chn_ids = torch.cat([chn_ids, torch.full((self.size - c,), chn_ids[-1])])
            elif chn_ids.ndimension() == 2:
                # Repeat the last row to fill the remaining size
                last_row = chn_ids[-1].unsqueeze(0).repeat(self.size - c, 1)
                chn_ids = torch.cat([chn_ids, last_row], dim=0)
                
            spec_masks = torch.cat([spec_masks, torch.ones(self.size - c, device=device)])
        spec_masks = spec_masks.bool()
        return dict(imgs=img, chn_ids=chn_ids, spec_masks=spec_masks, **add_kwargs)
