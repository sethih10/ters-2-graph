"""Auxiliary TERS config -- the one passed as `--config_original_qm9`.

ADAPT of configs/base_qm9.py.

**What this config is actually for.** `run_lib.py:181` and `:306` call `get_dataset` on it and
keep the resulting splits as `*_ds_original`. Those objects are used in exactly one place:

    run_lib.py:339   train_mols = [second_train_ds_original[i].rdmol for i in ...]
    run_lib.py:341   test_mols  = [test_ds_original[i].rdmol  for i in ...]

i.e. the MOSES reference sets. Nothing else reads them -- in `diffspectra_train` they are
loaded and never touched again.

So this config sets **`data.load_cubes = False`**. Without it, building `train_mols` would read
one image cube per molecule off disk and immediately discard it: tens of GB of I/O for data
nobody looks at. In that mode `TersGraphDataset` never opens the HDF5 at all and serves the
Phase-1 sidecar alone.

The model / optim / sampling sections below are inherited from base_qm9.py and are **not read**
on this path -- `create_model`, the optimiser and the sampler are all built from the *main*
config. They are kept so the file remains a valid, self-contained config rather than a fragment.
"""

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # `get_dataset` branches to _build_ters on data.name before it ever reads exp_type, so the
    # 5-tuple comes back regardless of what this says. Kept as 'diffspectra' for consistency
    # with the main config, and because that is what the QM9 pair does.
    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = '/scratch/project_2001912/planar_hdf5_0.5'
    data.graph_root = 'datasets/ters'
    data.name = 'TERS'
    data.processed_file = ''
    data.transform = 'EdgeComSpectra'
    data.collate = 'collate_cond'
    data.info_name = 'ters_hcno'         # added by Phase 5
    data.num_workers = 2

    data.ters_key = 'channels_100'
    data.load_cubes = False              # >>> the point of this file. See the module docstring.
    data.use_normalize = True
    data.aug_translation_scale = 0.1

    data.compress_edge = True
    data.centered = True
    data.include_aromatic = False        # must match the main config, or the edge feature width
                                         # of the reference mols disagrees with the model's
    data.atom_types = 4
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 36                   # dead config; documentation only

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # ---------------------------------------------------------------------------------------
    # Everything below is inherited from base_qm9.py and is NOT read on this path.
    # ---------------------------------------------------------------------------------------

    config.model = model = ml_collections.ConfigDict()
    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True
    model.normalize_factors = '6, 4, 4, 1'
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
    model.spatial_cut_off = 0.09
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1
    model.ters_in_channels = 100
    model.ters_filters = [64, 128, 256, 512]
    model.pretrained_ters_encoder_path = ''
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
    evaluate.num_samples = 782
    evaluate.begin_ckpt = 12
    evaluate.end_ckpt = 12
    evaluate.ckpts = ''
    evaluate.save_graph = False
    evaluate.sub_geometry = False

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config