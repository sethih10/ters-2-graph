"""DiffSpectra conditioned on QM9S TERS maps, binned to (100, 64, 64).

Pair with ``--config_original_qm9 configs/base_ters_qm9s.py``.
"""

import os

import ml_collections
import torch

from env import data_root

QM9S_ROOT = data_root('data')                           # graphs are cached in <root>/processed/
STM_MEAN, STM_STD = -3.6043, 1.5612                     # Å, over 4000 training molecules
QM9_SDF_GRAPHS = data_root('old', 'data', 'qm9s', 'diffspectra_processed', 'processed', 'data_qm9.pt')


def get_config():
    config = ml_collections.ConfigDict()
    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False
    config.seed = 42

    config.data = data = ml_collections.ConfigDict()
    data.name = 'QM9'
    data.root = QM9S_ROOT                 # {train,val,test}.h5
    data.modality = 'ters'                # 'ters' or 'stm'
    data.bin_modes = True                 # TERS mode maps summed into 100 bins of 40 cm^-1
    data.chn_mask = (24, 81)              # mode maps: modes kept in training (low, high); padded to high
    data.normalize = 'per_channel'        # 'per_channel', 'log_relative' (mode maps) or 'corpus' (STM)
    data.stm_mean = STM_MEAN              # used by normalize = 'corpus'
    data.stm_std = STM_STD
    data.cache_dir = QM9S_ROOT
    data.bonds = 'geometry'               # 'qm9' uses QM9's SDF graphs from data.qm9_path
    data.qm9_path = QM9_SDF_GRAPHS
    data.load_images = True
    data.info_name = 'qm9_second_half'    # H/C/N/O/F; also selects the QM9 stability checks
    data.atom_types = 5                   # must match data.info_name
    data.include_aromatic = False         # graphs are kekulised
    data.bond_types = 4
    data.compress_edge = True
    data.centered = True
    data.fc_scale = [-1., 1.]
    data.num_workers = 8
    # Options of datasets.transforms.get_transforms; every augmentation off.
    data.transforms = ml_collections.ConfigDict(dict(
        single_slice=False, hflip_rot90=False, rotate_deg=0.0, translate=0.0, blur_sigma=0.0,
        noise_std=0.0, random_erasing=0.0, p=0.5, eval_noise_std=0.0))

    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    config.model = model = ml_collections.ConfigDict()
    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True
    model.normalize_factors = '1, 4, 4, 1'     # positions stay in Å, as upstream
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256                             # must equal the encoder output_dim
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
    model.spatial_cut_off = 2.0                # squared distance in Å², as upstream
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1
    model.ters_encoder = 'cnn'                 # 'cnn' or 'mode_fusion' (TERS mode maps)
    model.ters_vit = 'tiny'                    # mode_fusion backbone: 'tiny' or 'small'
    model.ters_in_channels = 100               # must equal the image channels
    model.ters_filters = [64, 128, 256, 512]
    model.ters_pool_size = 4
    model.pretrained_ters_encoder_path = ''    # '' trains the encoder from scratch
    model.loss_weights = '1., 0.25, 0.1'       # positions, atom types, bonds
    model.noise_align = True

    config.training = training = ml_collections.ConfigDict()
    training.distributed = True
    training.ddp = False                       # True: DistributedDataParallel, batch_size per rank
    training.ddp_find_unused = True            # dist_layer is unused on steps without self-cond
    training.num_gpus = max(torch.cuda.device_count(), 1)
    training.world_size = int(os.environ.get('WORLD_SIZE', training.num_gpus))
    training.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    training.dataloader_drop_last = training.num_gpus > 1
    batch_size = 128                           # per GPU
    training.batch_size = batch_size * training.num_gpus
    training.eval_batch_size = batch_size * training.num_gpus
    training.eval_samples = batch_size * training.num_gpus
    training.reduce_mean = False
    training.log_freq = 50
    # Step counts are divided by num_gpus because batch_size scales with it. Keep n_iters a
    # multiple of snapshot_freq and eval.begin_ckpt = n_iters // snapshot_freq.
    training.n_iters = 150000 // training.num_gpus
    training.snapshot_freq = 25000 // training.num_gpus
    training.snapshot_freq_for_preemption = 2000 // training.num_gpus
    training.snapshot_sampling = False         # generation after training uses sample.py
    config.device = torch.device(f'cuda:{training.local_rank}')

    training.wandb = wandb = ml_collections.ConfigDict()
    wandb.enabled = True
    wandb.project = 'ters-diffspectra'
    wandb.entity = ''
    wandb.run_name = ''
    wandb.group = 'qm9s-ters'
    wandb.run_id = ''                          # set to rejoin a run after preemption
    wandb.mode = 'online'
    wandb.dotenv_path = '.env'                 # holds WANDB_API_KEY
    wandb.init_timeout = 60

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
    evaluate.batch_size = batch_size * training.num_gpus
    evaluate.num_samples = 6279                # must not exceed the test set
    evaluate.begin_ckpt = 6
    evaluate.end_ckpt = 6
    evaluate.ckpts = ''
    evaluate.sub_geometry = False
    evaluate.save_mols = 'false'
    evaluate.sampling_temperature = 1.0
    return config
