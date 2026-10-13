# TERS-2-Graph

TERS-2-Graph generates a molecule (graph and 3D coordinates) from a tip-enhanced Raman
spectroscopy (TERS) image cube or a scanning tunnelling microscopy (STM) image stack. It
adapts [DiffSpectra](https://github.com/AzureLeon1/DiffSpectra), replacing its 1D spectrum
encoder with an image encoder.

```text
image cube ──► cond_encoder ──► cond_lin ──► DMT denoiser ──► atoms, bonds, charges, 3D coordinates
 (C×H×W)         (CNN)           (linear)     (DiffSpectra)
```

| Dataset | Input | Elements | Config |
| --- | --- | --- | --- |
| Planar TERS | 256×256 px × 100 channels | H C N O | `diffspectra_ters.py` |
| QM9S-TERS | 64×64 px × 100 channels | H C N O F | `diffspectra_ters_qm9s.py` |
| STM | 14 z-slices | H C N O F Si P S Cl Br | `diffspectra_stm.py` |

There are two ways to train:

- **From scratch** with `main.py`: the encoder and the denoiser are trained together.
- **Fine-tuning** with `main_finetune.py`: a pretrained DiffSpectra denoiser is frozen,
  and only the encoder is trained.

## Repository layout

```text
ters-2-graph/
├── main.py                 # train or evaluate from scratch
├── main_finetune.py        # fine-tune with a frozen pretrained denoiser
├── run_lib.py              # training and evaluation loops
├── losses.py               # diffusion losses and optimizer
├── utils.py                # checkpoint save/restore, data scaling, self-conditioning
├── distributed.py          # DistributedDataParallel setup under Slurm
├── ters_wandb.py           # Weights & Biases logging
├── sample.py               # generate molecules from a checkpoint
├── merge_shards.py         # merge sharded sample.py outputs
├── compute_metrics.py      # target-similarity scoring used by main.py --mode eval
├── visualize.py            # molecule images saved during training
├── env.sh, env.py          # environment setup, machine paths from .env
├── configs/
│   ├── diffspectra_*.py    # main configs: TERS, QM9S-TERS, STM
│   ├── base_*.py           # auxiliary configs: graph-only reference sets
│   └── *_frozen*.py        # fine-tuning configs
├── datasets/
│   ├── preprocess_*.py     # HDF5 -> graph files (graphs_{split}.pt)
│   ├── ters_dataset.py     # TERS cube + graph dataset
│   ├── stm_dataset.py      # STM cube + graph dataset
│   ├── qm9s_dataset.py     # DiffSpectra's QM9S spectrum dataset (data.name = 'QM9S')
│   ├── build_dataset.py    # selects the dataset and batching from data.name
│   ├── datasets_config.py  # element sets (data.info_name)
│   └── ters/               # graph files for planar TERS
├── models/
│   ├── dmt.py              # DMT denoiser (model.name = 'DMT'); builds the conditioning encoder
│   ├── layers.py           # DMT attention and message-passing layers
│   ├── ters_encoder.py     # TERS image encoder
│   ├── stm_encoder*.py     # STM image encoders (model.stm_encoder = v1, v2, clip)
│   ├── specformer*.py      # DiffSpectra's 1D spectrum encoder, used when data.name = 'QM9S'
│   ├── dmt_wo_eq.py        # DiffSpectra's non-equivariant DMT ablation (model.name = 'DMT_WO_EQ')
│   ├── cdgs.py             # DiffSpectra's CDGS 2D graph model (model.name = 'CDGS')
│   ├── ema.py              # exponential moving average of weights
│   └── utils.py            # model registry and create_model
├── diffusion/, sampling/   # noise schedule and samplers
├── evaluation/             # metric functions
├── eval_sampled_mols/      # evaluation commands
└── tools/                  # diagnostics, analysis, visualisation
```

## Setup

```bash
source env.sh              # creates and activates the "ters-2-graph" Conda environment
cp .env.example .env       # then set DATA_ROOT and OUTPUT_ROOT
```

- **Paths.** `DATA_ROOT` holds the datasets and `OUTPUT_ROOT` holds runs and pretrained
  encoders. Exported environment variables override `.env`.
- **Logging.** Training logs to Weights & Biases. Set `WANDB_API_KEY` in `.env`, or pass
  `--config.training.wandb.enabled=False`.

## Data

A dataset is a directory with `train.h5`, `val.h5` and `test.h5`. These hold the image
cubes, atom positions and atomic numbers. A preprocessing step writes one graph file
`graphs_{split}.pt` per split. Each graph file stores the target molecule for every HDF5
row, with bonds inferred from geometry with RDKit.

```bash
# TERS / QM9S-TERS (--elements hcno or hcnof)
python -m datasets.preprocess_ters --data-dir $DATA_ROOT/planar_hdf5_0.5 --out-dir datasets/ters --stats

# STM
python -m datasets.preprocess_stm --data-dir $DATA_ROOT/stm-data --out-dir $DATA_ROOT/stm-graphs --stats
```

- **Paths in the config.** Set `data.root` to the HDF5 directory and `data.graph_root` to
  the graph directory. The graph files for planar TERS are included in `datasets/ters`.
- **Unfamiliar file.** `--inspect` prints the layout of an HDF5 file.
- **Self-test.** `--self-test` checks a preprocessor on synthetic data.

## Train from scratch

Each run takes a main config and an auxiliary config. The auxiliary config is the
graph-only reference set used by the evaluation metrics. Its flag keeps DiffSpectra's
name, `--config_original_qm9`.

| Dataset | Main config | Auxiliary config |
| --- | --- | --- |
| Planar TERS | `configs/diffspectra_ters.py` | `configs/base_ters.py` |
| QM9S-TERS | `configs/diffspectra_ters_qm9s.py` | `configs/base_ters_qm9s.py` |
| STM | `configs/diffspectra_stm.py` | `configs/base_stm.py` |

```bash
python main.py --mode train --workdir RUN \
    --config configs/diffspectra_ters.py --config_original_qm9 configs/base_ters.py
```

- **Outputs.** Checkpoints go to `RUN/checkpoints/checkpoint_N.pth`.
- **Resuming.** An existing `RUN` resumes from `RUN/checkpoints-meta/`, so use a new
  directory for each new run.
- **Overrides.** Any field can be overridden with `--config.FIELD=VALUE`.
- **Multiple GPUs.** The default is DataParallel. `--config.training.ddp=True` switches to
  DistributedDataParallel. It needs one `srun` task per GPU, plus `MASTER_ADDR` and
  `MASTER_PORT`.
- **STM encoder.** The STM config loads a pretrained encoder from `OUTPUT_ROOT`. Pass
  `--config.model.pretrained_stm_encoder_path=''` to train it from scratch.

## Fine-tune a pretrained denoiser

`main_finetune.py` loads a checkpoint's EMA weights, freezes the denoiser, and trains
only `cond_encoder` and `cond_lin`. Use it for both training and evaluation:

```bash
python main_finetune.py --mode train --workdir RUN \
    --config configs/qm9s_ters_frozen.py --config_original_qm9 configs/base_ters_qm9s.py
python main_finetune.py --mode eval  --workdir RUN \
    --config configs/qm9s_ters_frozen.py --config_original_qm9 configs/base_ters_qm9s.py
```

| Config | Source checkpoint | Encoder |
| --- | --- | --- |
| `diffspectra_ters_frozen.py` | DiffSpectra (H/C/N/O/F, cut to H/C/N/O) | new |
| `qm9s_ters_frozen.py` | DiffSpectra (H/C/N/O/F) | new |
| `ters_frozen_p4.py` | a from-scratch planar TERS run | continued |

The pretrained DiffSpectra checkpoint (`checkpoint_40.pth`, 793 MB, the
all-spectra model with pretrained SpecFormer) is published on
[Hugging Face](https://huggingface.co/AzureLeon1/DiffSpectra/tree/main/allspectra_pretrained_specformer/checkpoints).
Download it to the path the configs expect:

```bash
mkdir -p $DATA_ROOT/DiffSpectra/exp/allspectra_pretrained_specformer/checkpoints
wget -O $DATA_ROOT/DiffSpectra/exp/allspectra_pretrained_specformer/checkpoints/checkpoint_40.pth \
    https://huggingface.co/AzureLeon1/DiffSpectra/resolve/main/allspectra_pretrained_specformer/checkpoints/checkpoint_40.pth
```

To use a file elsewhere, pass `--config.model.pretrained_denoiser_path=PATH`.

- **Evaluate with `main_finetune.py` only.** The saved EMA covers only the trainable
  weights. `main.py`, `sample.py` and `run_metrics` would apply it to the wrong weights
  without an error.
- **Match the source checkpoint.**
  - Architecture: a mismatch fails when the weights are loaded.
  - Position scale (`normalize_factors`, `spatial_cut_off`): this is not checked.
  - Elements: the config's elements must be the first elements of the checkpoint's,
    in the same order.

## Generate and evaluate

```bash
# Generate 5 candidates for each of 20 test targets
python sample.py --modality ters --ckpt RUN/checkpoints/checkpoint_6.pth \
    --split test --n-mols 20 --n-samples 5 --batch-mols 4 --out SAMPLES

# Recovery: exact match and top-k similarity to the target
python -m eval_sampled_mols.paper_metrics --cells test=SAMPLES/mols_test.pkl --ks 1 3 5

# Quality: validity, stability, geometry, FCD, SNN, diversity
python -m eval_sampled_mols.eval_cond --samples SAMPLES/processed_test.pkl
```

**`sample.py` options:**

- `--modality ters|stm` picks the default config pair. For QM9S-TERS or any other
  dataset, pass `--config` and `--config_original_qm9`.
- `--shuffle-cubes` pairs each target with another molecule's image, as a null control.
- `--n-shards N --shard I` splits generation across processes. Combine the outputs with
  `merge_shards.py`.

**Other evaluation commands:**

- `python -m eval_sampled_mols.run_metrics --modality ters|stm --ckpt ...` generates
  and scores in one command, for the standard planar TERS and STM configs.
- `main.py --mode eval` scores checkpoints `config.eval.begin_ckpt` to `end_ckpt` with
  DiffSpectra's metrics.

[tools/README.md](tools/README.md) covers the scoring scripts, diagnostics and
visualisation tools.

## Extending

### Add an encoder

An encoder is an `nn.Module` that maps a batch of image cubes to one vector per molecule.
Input shapes:

- TERS: `(B, C, H, W)`, with C the number of spectral channels.
- STM: `(B, Z, H, W)`, with Z the number of z-slices.

The output must be `(B, config.model.nf)`.

1. **Write the module** in `models/`. The existing encoders are references:
   `ters_encoder.py` and `stm_encoder*.py`.
2. **Build it** in the conditioning-encoder block of `models/dmt.py`
   (`DMT.__init__`, after `_cond_kinds`). Select it with a config key, the way
   `model.stm_encoder` selects `v1`, `v2` or `clip`. Pass `output_dim=hidden_dim`.
3. **Declare every new config key** in the base config, for example `model.ters_encoder`
   and `model.<name>_*`. Command-line flags can override a key but cannot create one.

The encoder is stored under `cond_encoder.`, so `main_finetune.py` trains a new encoder
with no changes.

### Add a config

Configs are Python files that import an existing config and override fields. Two
existing configs show the pattern:

- [diffspectra_ters_qm9s.py](configs/diffspectra_ters_qm9s.py) is a new dataset for
  training from scratch.
- [qm9s_ters_frozen.py](configs/qm9s_ters_frozen.py) is the same dataset for
  fine-tuning.

**For a new dataset, set:**

| Field | Requirement |
| --- | --- |
| `data.root`, `data.graph_root` | HDF5 and graph directories. Set them in the auxiliary config too. |
| `data.info_name`, `data.atom_types` | An entry in `datasets/datasets_config.py`, and its element count. |
| `model.ters_in_channels` | The number of image channels. |
| `training.n_iters`, `training.snapshot_freq` | `n_iters` must be a multiple of `snapshot_freq`. |
| `eval.begin_ckpt`, `eval.end_ckpt` | Checkpoint numbers. Checkpoint N is saved at step N × `snapshot_freq`. |
| `eval.num_samples` | At most the size of the test split. |

**For fine-tuning, also set:**

- `model.pretrained_denoiser_path`.
- `model.load_pretrained_cond_encoder`: `False` for a new encoder, `True` to continue
  the checkpoint's encoder.
- `model.dropout = 0.0`.
- The source checkpoint's architecture and position scale. The scale defaults to
  DiffSpectra's, so change it only for checkpoints trained with another value.

## Conventions and limitations

- **Sampling is given each target's atom count.** The metrics measure structure recovery
  with a known atom count.
- **Position scale.** Positions are in Å (`normalize_factors = '1, 4, 4, 1'`), and
  `spatial_cut_off = 2.0` Å², both as in DiffSpectra. The cutoff marks atom pairs within
  1.41 Å as neighbours for one attention head. That covers C–H, double and aromatic bonds,
  but not C–C single bonds. `3.61` would cover every bond. Fine-tuning from DiffSpectra
  requires `2.0`.
- **Elements.** A DiffSpectra checkpoint cannot be fine-tuned to STM's ten elements.
