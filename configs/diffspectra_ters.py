"""DiffSpectra conditioned on TERS image cubes, adapted from configs/diffspectra_qm9s.py upstream.

Pair with `--config_original_qm9 configs/base_ters.py`; graphs come from `datasets.preprocess_ters`.
"""

import os

import ml_collections
import torch

from env import data_root


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False        # must stay False; get_dataloader rejects only_2D for TERS

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = data_root('planar_hdf5_0.5')   # {train,val,test}.h5
    data.graph_root = 'datasets/ters'    # graphs_{split}.pt from datasets.preprocess_ters
    data.name = 'TERS'                   # selects the dataset/collate/encoder branches
    data.processed_file = ''
    data.info_name = 'ters_hcno'         # H/C/N/O entry in datasets_config
    data.num_workers = 8

    data.compress_edge = True
    data.centered = True                 # center with 0
    data.include_aromatic = False        # sidecars are kekulized; True also needs model.edge_ch=3
    data.atom_types = 4                  # H/C/N/O; must match data.info_name
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 40                   # max atoms per molecule; not read

    data.ters_key = 'channels_100'       # the only cube stored in the HDF5 files
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
    model.include_fc_charge = True       # fc holds RDKit formal charges from preprocess_ters
    # First entry: position divisor ~ p95 centred radius (A); spatial_cut_off scales with it.
    model.normalize_factors = '7, 4, 4, 1'
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
    # Squared distance in (A / pos_norm) units: (1.9 / 7)**2, above the longest bond.
    model.spatial_cut_off = 0.0737
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1

    model.ters_in_channels = 100         # must equal TersGraphDataset.in_channels
    model.ters_filters = [64, 128, 256, 512]
    model.ters_pool_size = 4             # head's spatial grid; 1 is a global average pool
    model.pretrained_ters_encoder_path = ''      # empty = train the encoder from scratch

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
    training.accum_steps = 8             # not read

    if training.distributed:
        config.device = torch.device(f'cuda:{training.local_rank}')
    else:
        config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    training.reduce_mean = False
    training.log_freq = 50
    # Step counts are divided by num_gpus because batch_size scales with it.
    training.n_iters = 15000 // training.num_gpus
    # n_iters must be a multiple of this; eval.begin_ckpt = n_iters // snapshot_freq.
    training.snapshot_freq = 5000 // training.num_gpus
    training.snapshot_freq_for_preemption = 2000 // training.num_gpus
    # Sampling during training; generation after training uses sample.py.
    training.snapshot_sampling = False

    # Weights & Biases logging; every key is read through getattr.
    training.wandb = wandb = ml_collections.ConfigDict()
    wandb.enabled = True
    wandb.project = 'ters-diffspectra'
    wandb.entity = ''                    # '' = your default wandb entity
    wandb.run_name = ''                  # '' = let wandb name it
    wandb.group = ''
    wandb.run_id = ''                    # set to rejoin a specific run after preemption
    wandb.mode = 'online'                # 'offline' writes to <workdir>/wandb/ for `wandb sync`
    wandb.dotenv_path = '.env'           # WANDB_API_KEY lives here; not committed
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
    evaluate.num_samples = 782           # test split size; must not exceed len(test_ds)
    evaluate.begin_ckpt = 3              # = n_iters // snapshot_freq (tests/test_phases.py)
    evaluate.end_ckpt = 3
    evaluate.ckpts = ''                  # eg. '30'; '25, 30'
    evaluate.sub_geometry = False        # needs top_*_sym lists recomputed for this corpus
    evaluate.save_mols = 'false'         # compared as a lowercase string in diffspectra_evaluate
    evaluate.sampling_temperature = 1.0

    config.seed = 42

    return config
