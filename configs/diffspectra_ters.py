"""DiffSpectra conditioned on TERS image cubes.

ADAPT of configs/diffspectra_qm9s.py. Every value that differs from that file is marked
`# TERS:` with the reason. Everything unmarked is inherited unchanged and deliberately so --
the denoiser architecture is not what this retrofit is changing.

Run:
    module load python-pytorch/2.10
    export PYTHONPATH=$HOME/.cache/ters-overlay:$PWD      # see docs/05-progress-log.md D-2

    python main.py --config configs/diffspectra_ters.py \
        --config_original_qm9 configs/base_ters.py \
        --mode train --workdir exp/ters_run1

Before the first run:
  1. `python -m datasets.preprocess_ters --data-dir <hdf5 dir> --out-dir datasets/ters --stats`
     and paste the four suggested values it prints into the marked slots below.
  2. Set `data.root` and `data.graph_root`.
  3. Phase 5 must have added `'ters_hcno'` to datasets/datasets_config.py, or
     `get_dataset_info` raises on `data.info_name` below.
"""

import os

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False        # TERS: must stay False -- the cube conditions 3D positions,
                                  # and get_dataloader refuses only_2D for TERS on purpose.

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = '/scratch/project_2001912/planar_hdf5_0.5'
    # TERS: holds a real disjoint {train,val,test}.h5 -- 6254 / 782 / 782 molecules, verified
    # zero filename overlap between all three pairs. Only `channels_100` is stored now;
    # `channels_400` was dropped upstream, which is why data.ters_key below cannot change.
    data.graph_root = 'datasets/ters'    # TERS: graphs_{split}.pt from Phase 1. Separate from
                                         # data.root because the HDF5s are read-only scratch.
    data.name = 'TERS'                   # TERS: selects the dataset/collate/encoder branches
    data.processed_file = ''
    data.info_name = 'ters_hcno'         # TERS: added by Phase 5. Until then this raises.
    data.num_workers = 8                 # TERS: MEASURED on gputest (job 923931). At the old
                                         # value of 2 the loader dominated: 2.70 s to fetch a
                                         # batch. 8 workers takes that to 0.03 s. 16 is the
                                         # same, 32 is worse. build_dataset.py no longer
                                         # clamps this.
                                         #   The win is ~2.2x, not the 11x first claimed here:
                                         #   fetch and step are serialised, so this cuts
                                         #   2.70+2.16 to 0.03+2.16 s. The 11x came from the
                                         #   probe's bad step time -- see D-10.

    data.compress_edge = True
    data.centered = True                 # center with 0
    data.include_aromatic = False        # TERS: kept False, so Phase 1 kekulizes. Revisit only
                                         # with edge_ch=3 -- see D-4 and the D-5 open question.
    data.atom_types = 4                  # TERS: H/C/N/O, no F -- RE-MEASURED, Z = {1,6,7,8}
                                         # exactly across all 7818 molecules of the full dataset
                                         # (6254 train + 782 val + 782 test), so this survives the
                                         # move off the 782-molecule file unchanged. Resizes node
                                         # embedding and output head automatically (dmt.py:185).
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 40                   # TERS: dead config -- never read anywhere in this repo.
                                         # RE-MEASURED: largest molecule in the 6254-molecule
                                         # train split has 40 atoms (was 36 on the old file).

    # TERS: no `spectra_version` -- that key selects among uv/ir/raman and has no meaning here.
    #       DMT and the collate both read it through getattr, so its absence is handled.
    data.ters_key = 'channels_100'       # TERS: NOT channels_400. 4x fewer bytes per sample;
                                         # see the decisions log entry of 2026-08-25.
    data.load_cubes = True               # TERS: base_ters.py sets this False. See that file.
    data.aug_translation_scale = 0.1
    data.transform = 'EdgeComSpectra'
    data.use_normalize = True            # log10 normalisation; applies to ir/uv/raman only, so it
                                         # is a no-op here. The cube is normalised in the dataset.

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # model
    config.model = model = ml_collections.ConfigDict()

    model.name = 'DMT'
    model.pred_data = True
    model.include_fc_charge = True       # Kept True. Phase 1 fills `fc` with RDKit's assigned
                                         # charges, so this channel carries real signal.
    # TERS: position divisor 1 -> 7. RE-MEASURED on the 6254-molecule train split:
    # centred radius p95 = 6.63 A, max = 10.44 A. (The old value of 6 came from the
    # 782-molecule file: p95 6.47, max 8.11.) Leaving it at the QM9 value of 1 hands the
    # diffusion SNR schedule non-unit-scale data -- it does not crash, it just trains badly.
    # Coupled to model.spatial_cut_off below: that constant is in units of (A / pos_norm),
    # so changing this without changing that silently rescales the adjacency threshold.
    model.normalize_factors = '7, 4, 4, 1'
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256                       # = TersEncoder output_dim; cond_lin is Linear(nf, nf*4)
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
    # SQUARED distance, in units of (A / pos_norm) -- utils.py:56 divides pos by pos_norm
    # BEFORE the model sees it, so this is not angstrom**2. Upstream QM9 hides the distinction
    # because its pos_norm is 1; at pos_norm=7 the two differ by 49x.
    # 0.0737 = (1.9 / 7)**2 = 1.9 A. RE-MEASURED on the 6254-molecule train split of the full
    # planar_hdf5_0.5 dataset (270848 RDKit bonds): the longest bond is 1.867 A. The previous
    # 1.8 A -- fitted to the old 782-molecule file, whose longest bond was 1.601 A -- would now
    # sever real bonds from `adj_spatial`. 1.9 A restores 100% coverage.
    # The naive angstrom**2 answer (3.61) would mean 13.3 A -> nearly all pairs adjacent,
    # i.e. `adj_spatial` becomes an all-ones constant. Silent, not a crash.
    model.spatial_cut_off = 0.0737
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1

    # TERS: `patch_len` / `stride` deleted -- SpecFormer-only, and SpecFormer is not built here.
    # TERS: `pretrained_specformer_path` deliberately absent. Setting it on a TERS run raises,
    #       because SpecFormer weights cannot match a conv trunk (see the change manifest).
    model.ters_in_channels = 100         # TERS: must equal TersGraphDataset.in_channels
    model.ters_filters = [64, 128, 256, 512]
    # E-1 (docs/experimental.md). 1 = the original global average pool, which averages the
    # 256x256 map to one number per channel and so cannot carry spatial layout. 4 keeps a
    # 4x4 grid. Output width is unchanged, so nothing downstream of cond_lin moves.
    # Override without editing: --config.model.ters_pool_size=1
    model.ters_pool_size = 4
    model.pretrained_ters_encoder_path = ''      # empty = train the encoder from scratch

    # # loss function
    model.loss_weights = '1., 0.25, 0.1'
    model.noise_align = True

    # training
    config.training = training = ml_collections.ConfigDict()
    training.dataloader_drop_last = False
    training.distributed = True
    # TERS: `max(..., 1)`. The QM9S config divides by num_gpus unguarded, so importing it on a
    # CPU-only node is a ZeroDivisionError. Config files get imported for inspection.
    training.num_gpus = max(torch.cuda.device_count(), 1)
    print('number of gpus:', training.num_gpus)
    if training.num_gpus > 1:
        training.dataloader_drop_last = True
    training.world_size = int(os.environ.get('WORLD_SIZE', training.num_gpus))
    training.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # TERS: 128 -> 16 -> 128, per-GPU. Chosen on MEMORY, which the probe measures reliably:
    #     batch  peak GiB  reserved GiB   (gputest job 923974)
    #        64      21.4        27.5
    #       128      41.7        54.0     <- chosen
    #       256      82.4        92.5     of 95 -- 2.5 GiB of headroom over 4 steps
    # 256 was rejected because one batch of unusually large molecules would OOM a 17 h run.
    # 128 keeps half the card free, and production confirms it (42.6 GiB peak, wandb
    # perf/gpu_mem_gb). This is still the first number to cut on OOM.
    #
    # The same probe's THROUGHPUT column is not trustworthy -- it claimed 239 mol/s here
    # against a real 59 mol/s. Its batch-to-batch *ordering* held up; its absolute numbers
    # did not. See D-10 before quoting any figure from it.
    base_batch_size = 128
    training.batch_size = base_batch_size * training.num_gpus
    # TERS: both are consumed only by get_sampling_fn, which snapshot_sampling=False disables.
    # They are left tied to base_batch_size so they stay correct if sampling is re-enabled,
    # but note that raising base_batch_size raises the per-snapshot sampling cost with it.
    training.eval_batch_size = base_batch_size * training.num_gpus
    training.eval_samples = base_batch_size * training.num_gpus
    training.accum_steps = 8             # TERS: NEW and STILL DEAD -- grep confirms nothing
                                         # outside this file reads it. Kept as the hook for
                                         # Phase 6. It does NOT multiply the batch: the
                                         # effective batch is training.batch_size, full stop.

    if training.distributed:
        config.device = torch.device(f'cuda:{training.local_rank}')
    else:
        config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    training.reduce_mean = False
    training.log_freq = 50                              # TERS: 500 -> 50, a much shorter run
    # TERS: 2e6 -> 6e4 -> 2.8e4. Molecule-views seen is base_batch_size * n_iters * num_gpus
    # = base_batch_size * 28000 regardless of GPU count (batch_size multiplies by num_gpus,
    # n_iters divides by it). At base_batch_size=128 that is 3.58M views = ~573 epochs over
    # the 6254-molecule train split.
    #
    # 28000 is set by the walltime, from a REAL measurement, not a probe: production runs at
    # 2.16 s/step (job 925143, averaged over 500 steps, sd ~0.03). So 28000 steps = 16.8 h,
    # inside an 18 h request. 60000 would need 36.0 h -- exactly the gpumedium ceiling, with
    # no margin.
    #   Do not trust slurm/probe_batch_size.py for this number. It reported 0.50 s/step, 4x
    #   optimistic, because it measures 5 steps and losses.py:357 adds a second forward pass
    #   on a `random() < 0.5` draw -- unmeasurable in a 5-step sample. See D-10.
    #
    # You CANNOT watch a validation loss while this runs: the training loop has no eval branch
    # at all -- run_lib.py:188 builds `val_loader` and never reads it again, and `eval_freq`
    # exists nowhere. In-run signals are the training loss and the wandb metrics below. Judge
    # overfitting after the fact from the checkpoints, via --mode eval (which needs the
    # sampler work described under snapshot_sampling).
    training.n_iters = 28000 // training.num_gpus
    # TERS: with n_iters = 28000 this gives checkpoints at steps 5000/10000/15000/20000/25000,
    # plus one forced at step == n_iters (run_lib.py:250). That last one computes
    # save_step = 28000 // 5000 = 5 and therefore OVERWRITES checkpoint_5. The net effect is
    # five files where checkpoint_5 holds the FINAL weights, not step 25000's -- which is the
    # more useful of the two, so this is left as is rather than padding n_iters to a multiple
    # of snapshot_freq. eval.begin_ckpt below must stay in step with it.
    training.snapshot_freq = 5000 // training.num_gpus
    training.snapshot_freq_for_preemption = 2000 // training.num_gpus
    # TERS: True -> False. MUST stay False until sampling.py grows a TERS branch. That file
    # has no cube support whatsoever: sampling.py:158 reads `config.data.spectra_version` by
    # direct attribute access (not getattr), which this config deliberately does not define,
    # and `context` at sampling.py:206-219 is only ever assigned inside the uv/ir/raman/
    # allspectra branches -- so even after adding the key, context would be unbound.
    # With this True the run dies at the first snapshot (step 5000//num_gpus, ~1 h in), NOT
    # at startup, which is exactly the kind of failure that wastes a 36 h allocation.
    # Checkpoints are still written on the same schedule; only in-run generation is skipped.
    training.snapshot_sampling = False

    # --- ADDED (wandb) ------------------------------------------------------------------
    # Experiment tracking. Consumed by ters_wandb.init (called from run_lib.py); every key
    # is read through getattr, so deleting the block disables logging rather than raising.
    training.wandb = wandb = ml_collections.ConfigDict()
    wandb.enabled = True
    wandb.project = 'ters-diffspectra'
    wandb.entity = ''                    # '' = your default wandb entity
    wandb.run_name = ''                  # '' = let wandb name it
    wandb.group = ''
    wandb.run_id = ''                    # set to rejoin a specific run after preemption
    # 'online' because the gh200 compute nodes were MEASURED to have outbound internet
    # (job 925402: api.wandb.ai reached over HTTPS in 0.2 s, no proxy needed). The usual
    # HPC assumption is the opposite, so re-run slurm/verify_wandb_gputest.sbatch before
    # changing this. 'offline' writes to <workdir>/wandb/ for a later
    #     wandb sync exp/<run>/wandb/offline-run-*
    # and is the fallback if the network ever closes. See D-9 in docs/05-progress-log.md.
    wandb.mode = 'online'
    wandb.dotenv_path = '.env'           # WANDB_API_KEY lives here; not committed
    wandb.init_timeout = 60

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    # TERS: 100000 -> 2000. CRITICAL. losses.py:84 scales the LR by min(step/warmup, 1); leaving
    # this at 100k in a 60k-step run means the LR peaks at 60% of target and never arrives.
    optim.warmup = 2000
    optim.grad_clip = 10.
    optim.disable_grad_log = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 250                 # TERS: 1000 -> 250. Each step is a full DMT pass; 1000
                                         # of them per batch is the wall-clock floor, not the model.
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = base_batch_size * training.num_gpus
    # TERS: 10000 -> 782. MEASURED: the real test split holds exactly 782 molecules, all of
    # which survive Phase 1. This MUST stay <= len(test_ds) or sampling crashes.
    evaluate.num_samples = 782
    evaluate.begin_ckpt = 5              # TERS: 28000 // 5000 = 5. Was 12, sized for the old
                                         # n_iters = 60000; pointing at a checkpoint that is
                                         # never written makes --mode eval fail on a missing
                                         # file. tests/test_phases.py asserts these agree.
    evaluate.end_ckpt = 5
    evaluate.ckpts = ''                  # eg. '30'; '25, 30'
    evaluate.sub_geometry = False        # TERS: geometry MMD needs QM9-specific symbol lists.
                                         # Enable once the top_*_sym lists are recomputed.
    evaluate.save_mols = 'false'         # compared as a lowercase STRING at run_lib.py:420
    evaluate.sampling_temperature = 1.0

    config.seed = 42

    return config