"""DiffSpectra conditioned on STM constant-height z-stacks.

Pair with `--config_original_qm9 configs/base_stm.py`; graphs come from `datasets.preprocess_stm`.
"""

import os

import ml_collections
import torch

from env import data_root, output_root


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False        # must stay False; get_dataloader rejects only_2D for STM

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = data_root('stm-data')
    # {train,val,test}.h5; the cube <split>/X has shape (N, 128, 128, 14), one channel per z-slice.
    data.graph_root = data_root('stm-graphs')   # from datasets.preprocess_stm
    data.name = 'STM'                    # selects the dataset/collate/encoder branches
    data.processed_file = ''
    data.info_name = 'stm_10elem'        # 10-element entry in datasets_config
    data.num_workers = 8

    data.compress_edge = True
    data.centered = True                 # center with 0
    data.include_aromatic = False        # sidecars are kekulized; True also needs model.edge_ch=3
    data.atom_types = 10                 # H,C,N,O,F,Si,P,S,Cl,Br; must match data.info_name
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 54                   # max atoms per molecule; not read

    data.stm_key = 'X'                   # cube dataset name inside the split group
    data.load_cubes = True

    # Cube-noise probe: std relative to the normalised cube; 0.0 disables. Keep it a float.
    data.cube_noise_std = 0.0
    data.cube_noise_seed = 0
    data.aug_translation_scale = 0.1
    data.transform = 'EdgeComSpectra'
    data.use_normalize = True            # ir/uv/raman only; a no-op for cubes

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # model
    config.model = model = ml_collections.ConfigDict()

    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True       # fc holds RDKit formal charges from preprocess_stm
    # Same as upstream DiffSpectra: positions stay in Angstrom.
    model.normalize_factors = '1, 4, 4, 1'
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256                       # must equal the encoder output_dim
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
    # Squared distance in A^2, as in upstream DiffSpectra.
    model.spatial_cut_off = 2.0
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1

    model.stm_in_channels = 14           # z-slices per cube; must match the file
    # 'v1', 'v2' or 'clip'; must match the checkpoint being resumed or evaluated.
    model.stm_encoder = 'clip'
    # v1 encoder (models/stm_encoder.py); read only when stm_encoder == 'v1'.
    model.stm_conv3d_channels = [4, 8, 16]
    model.stm_filters = [64, 128, 256]
    model.stm_pool_size = 2              # head's spatial grid; 1 is a global average pool

    # v2 encoder (models/stm_encoder_v2.py); read only when stm_encoder == 'v2'.
    model.stm_v2_conv3d_channels = [48, 96, 192]   # EDAFMNet widths
    model.stm_v2_block_depth = 2                   # convolutions per residual 3D block
    model.stm_v2_filters = [128, 256]              # 2D trunk after the z-collapse
    model.stm_v2_pool_size = 2
    model.stm_v2_stem_channels = 16
    model.stm_v2_stem_stride = 2                   # spatial stride of the stem; 1 is memory-heavy

    # clip encoder (models/stm_encoder_clip.py); shapes are fixed by the pretrained checkpoint.
    model.stm_clip_conv3d_channels = [4, 8, 16]
    model.stm_clip_z_strides = [2, 2, 1]
    model.stm_clip_filters = [64, 128, 256]
    model.stm_clip_pool_size = 1         # must be 1: the checkpoint head is Linear(256, 256)

    # '' trains the clip encoder from scratch; requires stm_encoder == 'clip'.
    model.pretrained_stm_encoder_path = (
        output_root('runs', 'clip-full-1264302', 'cube_encoder_best.pt'))

    # # loss function
    model.loss_weights = '1., 0.25, 0.1'
    model.noise_align = True

    # training
    config.training = training = ml_collections.ConfigDict()
    training.dataloader_drop_last = False
    training.distributed = True
    # True: DistributedDataParallel with per-rank batch_size; False: torch.nn.DataParallel.
    training.ddp = False
    training.ddp_find_unused = True   # required: dist_layer is unused on steps without self-cond
    # At least 1, so the divisions below work on CPU-only nodes.
    training.num_gpus = max(torch.cuda.device_count(), 1)
    print('number of gpus:', training.num_gpus)
    if training.num_gpus > 1:
        training.dataloader_drop_last = True
    training.world_size = int(os.environ.get('WORLD_SIZE', training.num_gpus))
    training.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    base_batch_size = 128                # per GPU
    training.batch_size = base_batch_size * training.num_gpus
    # Used only by in-run sampling, which snapshot_sampling=False disables.
    training.eval_batch_size = base_batch_size * training.num_gpus
    training.eval_samples = base_batch_size * training.num_gpus
    training.accum_steps = 8              # not read

    if training.distributed:
        config.device = torch.device(f'cuda:{training.local_rank}')
    else:
        config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    training.reduce_mean = False
    training.log_freq = 50
    # Step counts are divided by num_gpus because batch_size scales with it.
    training.n_iters = 150000 // training.num_gpus
    # n_iters must be a multiple of this; eval.begin_ckpt = n_iters // snapshot_freq.
    training.snapshot_freq = 25000 // training.num_gpus
    # Resume checkpoint, overwritten in place.
    training.snapshot_freq_for_preemption = 2000 // training.num_gpus
    # Sampling during training; generation after training uses sample.py.
    training.snapshot_sampling = False

    # Weights & Biases logging; every key is read through getattr.
    training.wandb = wandb = ml_collections.ConfigDict()
    wandb.enabled = True
    wandb.project = 'stm-diffspectra'
    wandb.entity = ''
    wandb.run_name = ''
    wandb.group = ''
    wandb.run_id = ''
    wandb.mode = 'online'
    wandb.dotenv_path = '.env'
    wandb.init_timeout = 60

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 2000                  # steps of linear LR ramp
    optim.grad_clip = 10.
    optim.disable_grad_log = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 250
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = base_batch_size * training.num_gpus
    evaluate.num_samples = 2048          # must not exceed the test molecules in graphs_test.pt
    evaluate.begin_ckpt = 6              # = n_iters // snapshot_freq (tests/test_stm.py)
    evaluate.end_ckpt = 6
    evaluate.ckpts = ''                  # eg. '30'; '25, 30'
    evaluate.sub_geometry = False
    evaluate.save_mols = 'false'         # compared as a lowercase string in diffspectra_evaluate
    evaluate.sampling_temperature = 1.0

    config.seed = 42

    return config
