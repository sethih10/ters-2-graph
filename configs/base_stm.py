"""Auxiliary STM config for `--config_original_qm9`.

Only its graph splits are read, as reference molecules for the metrics; cubes are never loaded.
"""

import ml_collections
import torch

from env import data_root


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = data_root('stm-data')
    data.graph_root = data_root('stm-graphs')
    data.name = 'STM'
    data.processed_file = ''
    data.transform = 'EdgeComSpectra'
    data.collate = 'collate_cond'
    data.info_name = 'stm_10elem'
    data.num_workers = 2

    data.stm_key = 'X'
    data.load_cubes = False              # graph sidecar only
    data.use_normalize = True
    data.aug_translation_scale = 0.1

    data.compress_edge = True
    data.centered = True
    data.include_aromatic = False        # must match the main config
    data.atom_types = 10                 # must match the main config
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 54                   # max atoms per molecule; not read

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # Not read on this path; kept so the config is complete.

    config.model = model = ml_collections.ConfigDict()
    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True
    model.normalize_factors = '7, 4, 4, 1'
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256
    model.n_layers = 8
    model.n_heads = 16
    model.dropout = 0.1
    model.cond_time = True
    model.dist_gbf = True
    model.gbf_name = 'CondGaussianLayer'
    model.self_cond = True
    model.self_cond_type = 'ori'
    model.edge_quan_th = 0.
    model.n_extra_heads = 2
    model.CoM = True
    model.mlp_ratio = 2
    model.spatial_cut_off = 0.1276
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1
    model.stm_in_channels = 14
    model.stm_conv3d_channels = [4, 8, 16]
    model.stm_filters = [64, 128, 256]
    model.stm_pool_size = 2
    model.loss_weights = '1., 0.25, 0.1'
    model.noise_align = True

    config.training = training = ml_collections.ConfigDict()
    training.reduce_mean = False
    training.batch_size = 16
    training.eval_batch_size = 16
    training.eval_samples = 16
    training.log_freq = 50
    training.n_iters = 60000
    training.snapshot_freq = 5000
    training.snapshot_freq_for_preemption = 2000
    training.snapshot_sampling = True
    training.dataloader_drop_last = False

    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 2000
    optim.grad_clip = 10.
    optim.disable_grad_log = True

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 250
    sampling.vis_row = 4
    sampling.vis_col = 4

    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = 16
    evaluate.num_samples = 23123          # test split size
    evaluate.begin_ckpt = 12
    evaluate.end_ckpt = 12
    evaluate.ckpts = ''
    evaluate.save_graph = False
    evaluate.sub_geometry = False

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
