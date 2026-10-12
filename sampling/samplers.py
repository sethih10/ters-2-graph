"""Reverse-diffusion samplers.

Moved unchanged from DiffSpectra sampling.py.
"""

import torch

from models.utils import (sample_combined_position_feature_noise, assert_mean_zero_with_mask,
                          sample_symmetric_edge_feature_noise, sample_gaussian_with_mask)


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]



class AncestralSampler:
    """Ancestral sampling for 2D & 3D joint generation."""
    def __init__(self, noise_scheduler, time_steps, model_pred_data, pred_edge=False, self_cond=False, cond_process_fn=None, sampling_temperature=1.0):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.pred_edge = pred_edge
        self.self_cond = self_cond
        self.cond_process_fn = cond_process_fn
        self.sampling_temperature = sampling_temperature
        
    def sampling(self, model, z_T, node_mask, edge_mask, edge_z_T=None, context=None):
        x = z_T
        edge_x = edge_z_T
        bs = z_T.shape[0]
        cond_x, cond_edge_x = None, None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            if self.pred_edge:
                if self.self_cond:
                    assert self.model_pred_data
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                                cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
                    cond_x, cond_edge_x = self.cond_process_fn(pred_t, edge_pred_t)
                else:
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                                context=context)

            else:
                if self.self_cond:
                    assert self.model_pred_data
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                   cond_x=cond_x, context=context)
                else:
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level, context=context)

            # node update
            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                sample_combined_position_feature_noise(bs, x_mean.shape[1], x_mean.shape[2] - 3, node_mask) * self.sampling_temperature

            # edge update
            if self.pred_edge:
                if self.model_pred_data:
                    edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                                  * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                         edge_pred_t.dim()) * edge_pred_t
                else:
                    edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t
                edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                         sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask) * self.sampling_temperature

        assert_mean_zero_with_mask(x_mean[:, :, :3], node_mask)

        if self.pred_edge:
            return x_mean, edge_x_mean
        else:
            return x_mean



class AncestralSampler_2D:
    """Ancestral Sampler without 3D positions."""
    def __init__(self, noise_scheduler, time_steps, model_pred_data, self_cond=False):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.self_cond = self_cond

    def sampling(self, model, z_T, node_mask, edge_mask, edge_z_T=None, context=None):
        x = z_T
        edge_x = edge_z_T
        bs = z_T.shape[0]
        cond_x, cond_edge_x = None, None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            if self.self_cond:
                assert self.model_pred_data
                pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                            cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
                cond_x, cond_edge_x = pred_t, edge_pred_t
            else:
                pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                            context=context)

            # node update
            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                sample_gaussian_with_mask(x.size(), x.device, node_mask)

            # edge update
            if self.model_pred_data:
                edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                              * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                     edge_pred_t.dim()) * edge_pred_t
            else:
                edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t

            edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                     sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask)

        return x_mean, edge_x_mean
