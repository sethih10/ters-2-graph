"""DiffSpectra conditioned on STM constant-height z-stacks.

ADAPT of configs/diffspectra_ters.py. Every value that differs from that file is marked
`# STM:` with the reason; everything unmarked is inherited unchanged and deliberately so --
the denoiser architecture is not what this adds.

Run:
    module load python-pytorch/2.10
    export PYTHONPATH=$HOME/.cache/ters-overlay:$PWD      # see docs/05-progress-log.md D-2

    # once, to build the graph sidecars (~40 min for all three splits at ~100 mol/s):
    python -m datasets.preprocess_stm \\
        --data-dir /scratch/project_2001912/stm-data \\
        --out-dir  /scratch/project_2001912/stm-graphs --stats

    python main.py --config configs/diffspectra_stm.py \\
        --config_original_qm9 configs/base_stm.py \\
        --mode train --workdir /scratch/project_2001912/stm-runs/run1

Every value here is now MEASURED -- corpus statistics on the real HDF5, training cost on a
real GH200 (``slurm/smoke_stm_gputest.sbatch``, job 1134273). Nothing is inherited from TERS
on faith; where a TERS number was carried over unchecked it turned out to be wrong by 7.7x
(see ``base_batch_size``).
"""

import os

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False        # STM: must stay False -- the z-stack conditions 3D
                                  # positions, and get_dataloader refuses only_2D for STM.

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = '/scratch/project_2001912/stm-data'
    # STM: {train,val,test}.h5 -- 184180 / 23067 / 23123 molecules, split 8:1:1 on disk
    # (`ratios` attr). Each holds ONE group named after the split; the cube is <split>/X,
    # shape (N, 128, 128, 14) uint16, and the 14 channels are constant-height slices at
    # z = 6..19 (`z_slices` attr).
    data.graph_root = '/scratch/project_2001912/stm-graphs'
    # STM: NOT $HOME. The three sidecars are ~900 MB together (~4 KB/molecule x 230k) and
    # $HOME here is a 15 GB quota sitting at 13.9 GB -- a run that writes there dies on quota
    # partway through, which is how ters-ae lost job 989420 (see slurm/train_gpumedium.sbatch).
    data.name = 'STM'                    # STM: selects the dataset/collate/encoder branches
    data.processed_file = ''
    data.info_name = 'stm_10elem'        # STM: 10 elements, not the 4 of ters_hcno
    data.num_workers = 8                 # STM: kept at the TERS value. Justified rather than
                                         # copied: an STM cube is 128*128*14*4 B = 917 KB
                                         # against TERS's 26.2 MB, so the per-sample decode is
                                         # ~28x cheaper and 8 workers is, if anything,
                                         # generous. get_dataloader does NOT clamp
                                         # prefetch_factor for STM -- see the note there.

    data.compress_edge = True
    data.centered = True                 # center with 0
    data.include_aromatic = False        # STM: kept False, so preprocess_stm kekulizes.
                                         # STM-MEASURED: 116445 of 184157 train molecules
                                         # (63.2%) contain an aromatic bond before
                                         # kekulization, so the D-4 trap is very live here --
                                         # with this False and un-kekulized input, two thirds
                                         # of the corpus would silently lose its ring bonds.
    data.atom_types = 10                 # STM-MEASURED: Z = {1,6,7,8,9,14,15,16,17,35} =
                                         # H,C,N,O,F,Si,P,S,Cl,Br over 20000 train molecules.
                                         # NOT the 4 of TERS. Resizes the node embedding and
                                         # output head automatically (dmt.py:185) -- but
                                         # data.info_name's atom_encoder must agree, or the
                                         # one-hot width and this disagree silently.
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 54                   # STM-MEASURED on the full train split: the xyz block
                                         # is padded to 54 rows and 4 molecules of 184157
                                         # genuinely fill it. The size histogram tapers
                                         # smoothly into 54 (…51:21, 52:12, 53:4, 54:4)
                                         # rather than piling up there, so these are real
                                         # molecules and not truncations. Median 23, p95 35.
                                         # Dead config -- never read anywhere in this repo --
                                         # but kept accurate.

    # STM: no `spectra_version` -- selects among uv/ir/raman, meaningless here. DMT and the
    #      collate both read it through getattr, so its absence is handled.
    data.stm_key = 'X'                   # STM: cube dataset name inside the split group.
    data.load_cubes = True               # STM: base_stm.py sets this False. See that file.
    data.aug_translation_scale = 0.1
    data.transform = 'EdgeComSpectra'
    data.use_normalize = True            # log10 normalisation; ir/uv/raman only, so a no-op
                                         # here. The z-stack is normalised in the dataset by
                                         # NormalizeStmCube -- NOT by NormalizeVectorized,
                                         # which would delete the decay envelope along z.

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # model
    config.model = model = ml_collections.ConfigDict()

    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True       # Kept True. preprocess_stm fills `fc` with RDKit's
                                         # assigned FORMAL charges. Note it does NOT use the
                                         # partial charge in xyz column 4 -- that is a
                                         # fractional quantity and would be a different
                                         # feature entirely. See preprocess_stm's warning.
    # STM-MEASURED on the full 184180-molecule train split: centred radius p95 = 6.82 A,
    # max = 11.61 A, so the divisor that lands p95 near 1.0 is 7. (Coincidentally the same
    # value TERS uses, for a different corpus.) Leaving this at the QM9 value of 1 hands the
    # diffusion SNR schedule non-unit-scale data -- it does not crash, it just trains badly.
    # Coupled to model.spatial_cut_off below, which is in units of (A / pos_norm), so these
    # two must always be changed together.
    model.normalize_factors = '7, 4, 4, 1'
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256                       # = StmEncoder output_dim; cond_lin is Linear(nf, nf*4)
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
    # SQUARED distance in units of (A / pos_norm) -- utils.py:56 divides pos by pos_norm
    # BEFORE the model sees it, so this is not angstrom**2. The naive angstrom**2 value (6.25)
    # would mark nearly every atom pair adjacent, making `adj_spatial` an all-ones constant --
    # silent, not a crash.
    # STM-MEASURED on the full 184180-molecule train split (8437270 RDKit bonds): the LONGEST
    # bond is 2.437 A, so 0.1276 = (2.5 / 7)**2 keeps 100% of real bonds. Derived from the
    # longest bond, not from a nearest-neighbour proxy -- an atom's nearest neighbour is
    # usually an H at ~1.0 A while the longest bond in the same molecule is a C-Br at ~1.9 A.
    # NOTE this is much wider than TERS's 1.9 A cutoff, and it has to be: TERS is H/C/N/O
    # only, while this corpus contains Br, Si and P, whose bonds are simply longer.
    model.spatial_cut_off = 0.1276
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1

    # STM: `patch_len` / `stride` deleted -- SpecFormer-only, and SpecFormer is not built here.
    # STM: `pretrained_specformer_path` deliberately absent. Setting it on an STM run raises,
    #      because SpecFormer weights cannot match a 3D conv stem (see the change manifest).
    model.stm_in_channels = 14           # STM-MEASURED: = len(attrs['z_slices']), z = 6..19.
                                         # StmGraphDataset cross-checks this against the file
                                         # and raises if they disagree.
    model.stm_conv3d_channels = [4, 8, 16]   # STM: the ASDAFMNet 3D stem ladder.
    model.stm_filters = [64, 128, 256]   # STM: three 2D blocks, not TERS's four. The 3D stem
                                         # already downsamples 8x spatially (128 -> 16), so a
                                         # fourth block would be pooling a 2x2 map.
    # STM: 2, not TERS's 4. After the stem (128->16) and three pooling blocks (16->8->4->2)
    # the feature map is exactly 2x2, so pool_size=2 keeps it whole; 4 would ask
    # AdaptiveAvgPool2d to invent a 4x4 grid from 2x2 by duplication -- more parameters in the
    # head for no more information. 1 is the original global average pool, which discards
    # spatial layout entirely; E-1 (docs/experimental.md) found that harmful on TERS.
    # Override without editing: --config.model.stm_pool_size=1
    model.stm_pool_size = 2

    # # loss function
    model.loss_weights = '1., 0.25, 0.1'
    model.noise_align = True

    # training
    config.training = training = ml_collections.ConfigDict()
    training.dataloader_drop_last = False
    training.distributed = True
    # `max(..., 1)` so importing this config on a CPU-only node is not a ZeroDivisionError.
    training.num_gpus = max(torch.cuda.device_count(), 1)
    print('number of gpus:', training.num_gpus)
    if training.num_gpus > 1:
        training.dataloader_drop_last = True
    training.world_size = int(os.environ.get('WORLD_SIZE', training.num_gpus))
    training.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # STM-MEASURED on one GH200 (gputest job 1134273, slurm/smoke_stm_gputest.sbatch):
    #     batch  peak GiB  resv GiB  s/step  mol/s
    #        64      12.6      13.8    0.28    231
    #       128      23.4      42.9    0.28    454     <- chosen
    #       256      46.0      62.3    0.52    493
    # 128 uses 23.4 GiB of 95, so there is a lot of headroom; 256 also fits and is 8.6%
    # faster per molecule, but halves the number of optimiser updates for the same views and
    # the throughput gain does not pay for that. Raise to 256 only if wall-clock becomes the
    # binding constraint.
    #
    # NOTE what this replaced: the TERS-inherited guess assumed ~2.16 s/step. The real figure
    # at this batch size is 0.279 s/step -- 7.7x faster -- because the STM cube is 28x smaller
    # than a TERS cube (917 KB vs 26.2 MB) and the conditioning encoder is 1.49M parameters
    # against TersEncoder's larger trunk. The larger max_node (54 vs 40) did not offset it.
    base_batch_size = 128
    training.batch_size = base_batch_size * training.num_gpus
    # Consumed only by get_sampling_fn, which snapshot_sampling=False disables. Left tied to
    # base_batch_size so they stay correct if sampling is re-enabled.
    training.eval_batch_size = base_batch_size * training.num_gpus
    training.eval_samples = base_batch_size * training.num_gpus
    training.accum_steps = 8              # STILL DEAD -- nothing outside this file reads it.

    if training.distributed:
        config.device = torch.device(f'cuda:{training.local_rank}')
    else:
        config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    training.reduce_mean = False
    training.log_freq = 50
    # STM-MEASURED: 0.279 s/step at base_batch_size=128 (job 1134273, 150 real training steps
    # through main.py, 3.58 steps/s sustained over the last 100). So:
    #
    #     150000 steps x 0.279 s = 41850 s = 11.6 h, inside the 16 h request in the sbatch.
    #
    # Molecule-views seen is base_batch_size * n_iters * num_gpus = base_batch_size * 150000
    # regardless of GPU count (batch_size multiplies by num_gpus, n_iters divides by it). At
    # base_batch_size=128 that is 19.2M views over the 184157-molecule train split = ~104
    # epochs.
    #
    # Why 150000 and not the 28000 TERS uses: at the measured speed 28000 steps is 2.2 h and
    # only ~19.5 epochs of this corpus, against ~573 epochs for the same step count on TERS's
    # 6254 molecules. This corpus is 29x larger, so it needs proportionally more steps to be
    # seen a comparable number of times. If the loss is still falling at 150000, raise this --
    # the constraint is walltime and checkpoint disk, not anything about the model.
    #
    # You CANNOT watch a validation loss while this runs: the training loop has no eval
    # branch -- run_lib.py:188 builds `val_loader` and never reads it again.
    training.n_iters = 150000 // training.num_gpus
    # MEASURED: an STM checkpoint is 587 MB (job 1134273 wrote one). /scratch had 13 GiB free
    # when this was set, so 5000 -- which would write 30 checkpoints = 17.6 GB -- WOULD HAVE
    # FILLED THE DISK and killed the run partway through. At 25000 it is 6 checkpoints =
    # 3.5 GB. Keep this in step with eval.begin_ckpt below, and re-check `df /scratch` before
    # lowering it.
    training.snapshot_freq = 25000 // training.num_gpus
    # Overwrites a single checkpoints-meta file rather than accumulating, so its cost is one
    # 587 MB file no matter how often it fires; frequent is good, it is the resume point.
    training.snapshot_freq_for_preemption = 2000 // training.num_gpus
    # STM: True -> False, and for the same reason as TERS -- sampling.py has no cube branch.
    # sampling.py:158 reads `config.data.spectra_version` by direct attribute access (not
    # getattr), which this config deliberately does not define, and `context` at
    # sampling.py:206-219 is only assigned inside the uv/ir/raman/allspectra branches, so
    # even after adding the key `context` would be unbound. With this True the run dies at
    # the FIRST SNAPSHOT (~1 h in), not at startup -- the kind of failure that wastes an
    # allocation. Use tools/sample_ters.py after training instead.
    training.snapshot_sampling = False

    # --- ADDED (wandb) ------------------------------------------------------------------
    # Every key is read through getattr, so deleting the block disables logging.
    training.wandb = wandb = ml_collections.ConfigDict()
    wandb.enabled = True
    wandb.project = 'stm-diffspectra'    # STM: separate project from the TERS runs
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
    # CRITICAL. losses.py:84 scales the LR by min(step/warmup, 1); leaving this at the
    # upstream 100000 in a 28000-step run means the LR peaks at 28% of target and never
    # arrives.
    optim.warmup = 2000
    optim.grad_clip = 10.
    optim.disable_grad_log = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 250                 # STM: 1000 -> 250. Each step is a full DMT pass;
                                         # 1000 of them per batch is the wall-clock floor.
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = base_batch_size * training.num_gpus
    # STM-MEASURED: the test split holds 23123 cubes. This MUST stay <= len(test_ds) or
    # sampling crashes -- and len(test_ds) is the number of molecules that SURVIVED
    # preprocessing, not the number of cubes, so check graphs_test.pt before raising it.
    # 23123 x 250 ancestral steps is also far more sampling than any eval needs; the value
    # here is the ceiling, not a recommendation.
    evaluate.num_samples = 2048
    evaluate.begin_ckpt = 6              # STM: 150000 // 25000 = 6. Pointing at a checkpoint
                                         # that is never written makes --mode eval fail on a
                                         # missing file. Keep in step with training.n_iters
                                         # and training.snapshot_freq; tests/test_stm.py
                                         # asserts the three agree.
    evaluate.end_ckpt = 6
    evaluate.ckpts = ''                  # eg. '30'; '25, 30'
    evaluate.sub_geometry = False        # STM: the top_*_sym lists in datasets_config.py ARE
                                         # measured for this corpus, so this can be turned on
                                         # -- left False to match the TERS default until a
                                         # first eval run confirms the populations are hit.
    evaluate.save_mols = 'false'         # compared as a lowercase STRING at run_lib.py:420
    evaluate.sampling_temperature = 1.0

    config.seed = 42

    return config
