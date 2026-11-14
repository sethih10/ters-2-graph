# TERS-2-Graph

> **Known issues with the new HDF5 files (2026-11-12). Read before running a model.**
>
> 1. **Split leak.** About 76% of the new val and test molecules were in DiffSpectra's training
>    split. Evaluating the frozen pretrained denoiser (`main_finetune.py`) on them overstates the
>    scores. Training from scratch is unaffected.
> 2. **`bonds='qm9'` is disabled**, so `configs/qm9s_ters_frozen.py` does not run. The files carry
>    QM9S ids and `data_qm9.pt` is indexed by QM9 ids; a lookup has to be built from the old files.

TERS-2-Graph generates a molecule (graph and 3D coordinates) from an image: a tip-enhanced Raman
spectroscopy (TERS) image cube or a scanning tunnelling microscopy (STM) image stack. It adapts
[DiffSpectra](https://github.com/AzureLeon1/DiffSpectra), replacing its 1D spectrum encoder with an
image encoder.

```text
image (C×H×W) ──► cond_encoder ──► cond_lin ──► DMT denoiser ──► atoms, bonds, charges, 3D coordinates
                   (CNN)            (linear)     (DiffSpectra)
```

| Dataset | Input | Elements | Config |
| --- | --- | --- | --- |
| QM9S-TERS | 64×64 px × 100 frequency channels | H C N O F | `diffspectra_ters_qm9s.py` |
| QM9-STM | 64×64 px × 11 current setpoints (one used per sample) | H C N O F | `diffspectra_stm_qm9.py` |

There are two ways to train:

- **From scratch** with `main.py`: the encoder and the denoiser are trained together.
- **Fine-tuning** with `main_finetune.py`: a pretrained DiffSpectra denoiser is frozen, and only the
  encoder is trained.

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
│   ├── diffspectra_*.py    # main configs: QM9S-TERS, QM9-STM
│   ├── base_*.py           # auxiliary configs: reference molecules for the metrics
│   └── qm9s_ters_frozen.py # fine-tuning config
├── datasets/
│   ├── image_dataset.py    # ImageGraphDataset: cached graphs + images read from HDF5
│   ├── build_dataset.py    # get_dataset, get_dataloader, graph features, batching
│   ├── transforms.py       # normalisation, slice selection, augmentation
│   └── datasets_config.py  # element sets (data.info_name)
├── models/
│   ├── dmt.py              # DMT denoiser; builds the conditioning encoder
│   ├── layers.py           # DMT attention and message-passing layers
│   ├── ters_encoder.py     # image encoder
│   ├── stm_encoder*.py     # encoders for the old 128×128×14 STM corpus (unused)
│   ├── ema.py              # exponential moving average of weights
│   └── utils.py            # model registry and create_model
├── diffusion/, sampling/   # noise schedule and samplers
├── evaluation/             # metric functions
├── eval_sampled_mols/      # evaluation commands
├── tests/                  # checks on real data (run on a compute node)
└── tools/                  # diagnostics, analysis, visualisation
```

## Setup

```bash
source env.sh              # creates and activates the "ters-2-graph" Conda environment
cp .env.example .env       # then set DATA_ROOT and OUTPUT_ROOT
```

- **Paths.** `DATA_ROOT` holds the datasets and `OUTPUT_ROOT` holds runs. Exported environment
  variables override `.env`.
- **Logging.** Training logs to Weights & Biases. Set `WANDB_API_KEY` in `.env`, or pass
  `--config.training.wandb.enabled=False`.

## Data

A dataset is a directory with `train.h5`, `val.h5` and `test.h5`. Each file holds:

- the images, `(N, H, W, C)`, under `data.image_key`;
- every molecule's atoms and coordinates in Å, either as one group per row
  (`atomic_numbers/{i}`, `atom_positions/{i}`) or as an `xyz` array `(N, atoms, 4+)`;
- optionally `qm9_idx` (the 1-based QM9 molecule ID) and `smiles`.

There is no separate preprocessing step. On first use, `ImageGraphDataset` builds every molecule's
graph and caches it in `<data.cache_dir>/processed/<split>_<bonds>.pt`; later runs load the cache.
Delete that folder to rebuild. `data.bonds` chooses where the bonds come from:

| `data.bonds` | Bonds from | Use for |
| --- | --- | --- |
| `geometry` (default) | RDKit bond assignment on the coordinates; the SMILES when that fails (strained cages) | training from scratch: every kept graph is a valid neutral molecule |
| `qm9` | QM9's SDF graphs (DiffSpectra's `data_qm9.pt`), matched by `qm9_idx` | fine-tuning the DiffSpectra denoiser, which was trained on them, and comparison with published QM9 results |

About 21% of QM9's SDF graphs are not valid closed-shell molecules (for example a C=N double bond
stored as single); see `docs/research/08-faq.md` Q29. Images are read from the HDF5 file for each
sample and normalised with `data.normalize`: `per_channel` (TERS) or `corpus` (one mean and std
stored in the file, STM). QM9-STM files have no SMILES; `data.smiles_root` points at the QM9S-TERS
files, reached through each row's `ters_row`.

**Augmentation.** `data.transforms` holds the options of `datasets.transforms.get_transforms`
(flips and quarter turns, rotation, translation, blur, noise, erasing, single-slice selection). All
are off by default and apply to training only, for example
`--config.data.transforms.hflip_rot90=True`.

## Train from scratch

Each run takes a main config and an auxiliary config. The auxiliary config is the main config
without images; it supplies the reference molecules for the metrics. Its flag keeps DiffSpectra's
name, `--config_original_qm9`.

| Dataset | Main config | Auxiliary config |
| --- | --- | --- |
| QM9S-TERS | `configs/diffspectra_ters_qm9s.py` | `configs/base_ters_qm9s.py` |
| QM9-STM | `configs/diffspectra_stm_qm9.py` | `configs/base_stm_qm9.py` |

```bash
python main.py --mode train --workdir RUN \
    --config configs/diffspectra_ters_qm9s.py --config_original_qm9 configs/base_ters_qm9s.py
```

- **Outputs.** Checkpoints go to `RUN/checkpoints/checkpoint_N.pth`.
- **Resuming.** An existing `RUN` resumes from `RUN/checkpoints-meta/`, so use a new directory for
  each new run.
- **Overrides.** Any field can be overridden with `--config.FIELD=VALUE`.
- **Multiple GPUs.** The default is DataParallel. `--config.training.ddp=True` switches to
  DistributedDataParallel. It needs one `srun` task per GPU, plus `MASTER_ADDR` and `MASTER_PORT`.

## Fine-tune a pretrained denoiser

`main_finetune.py` loads a checkpoint's EMA weights, freezes the denoiser, and trains only
`cond_encoder` and `cond_lin`. Use it for both training and evaluation:

```bash
python main_finetune.py --mode train --workdir RUN \
    --config configs/qm9s_ters_frozen.py --config_original_qm9 configs/base_ters_qm9s.py
python main_finetune.py --mode eval  --workdir RUN \
    --config configs/qm9s_ters_frozen.py --config_original_qm9 configs/base_ters_qm9s.py
```

`qm9s_ters_frozen.py` uses `data.bonds = 'qm9'`. The pretrained DiffSpectra checkpoint
(`checkpoint_40.pth`, 793 MB, the all-spectra model with pretrained SpecFormer) is published on
[Hugging Face](https://huggingface.co/AzureLeon1/DiffSpectra/tree/main/allspectra_pretrained_specformer/checkpoints).
Download it to the path the config expects:

```bash
mkdir -p $DATA_ROOT/code/DiffSpectra/exp/allspectra_pretrained_specformer/checkpoints
wget -O $DATA_ROOT/code/DiffSpectra/exp/allspectra_pretrained_specformer/checkpoints/checkpoint_40.pth \
    https://huggingface.co/AzureLeon1/DiffSpectra/resolve/main/allspectra_pretrained_specformer/checkpoints/checkpoint_40.pth
```

To use a file elsewhere, pass `--config.model.pretrained_denoiser_path=PATH`.

- **Evaluate with `main_finetune.py` only.** The saved EMA covers only the trainable weights.
  `main.py`, `sample.py` and `run_metrics` would apply it to the wrong weights without an error.
- **Match the source checkpoint.** A mismatched architecture fails when the weights are loaded; a
  mismatched position scale (`normalize_factors`, `spatial_cut_off`) is not checked. The config's
  elements must be the first elements of the checkpoint's, in the same order.

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

- `--modality ters|stm` picks the QM9S-TERS or QM9-STM config pair; `--config` and
  `--config_original_qm9` override it.
- `--shuffle-cubes` pairs each target with another molecule's image, as a null control.
- `--n-shards N --shard I` splits generation across processes. Combine the outputs with
  `merge_shards.py`.

**Other evaluation commands:**

- `python -m eval_sampled_mols.run_metrics --modality ters|stm --ckpt ...` generates and scores in
  one command.
- `main.py --mode eval` scores checkpoints `config.eval.begin_ckpt` to `end_ckpt` with DiffSpectra's
  metrics.

[tools/README.md](tools/README.md) covers the scoring scripts, diagnostics and visualisation tools.

## Tests

The tests read real data, so run them on a compute node from the repository root:

```bash
python tests/test_image_dataset.py <empty cache dir>          # datasets, both bond sources
python -m unittest discover -s tests -p test_frozen.py        # fine-tuning weight handling
```

## Extending

### Add an encoder

An encoder is an `nn.Module` that maps a batch of images `(B, C, H, W)` to `(B, config.model.nf)`.

1. **Write the module** in `models/`, using `ters_encoder.py` as the reference.
2. **Build it** in the conditioning-encoder block of `models/dmt.py` (`DMT.__init__`, after
   `_cond_kinds`), selected by a config key. Pass `output_dim=hidden_dim`.
3. **Declare every new config key** in the main config. Command-line flags can override a key but
   cannot create one.

The encoder is stored under `cond_encoder.`, so `main_finetune.py` trains a new encoder with no
changes.

### Add a dataset

Copy the pattern of [diffspectra_stm_qm9.py](configs/diffspectra_stm_qm9.py): import the QM9S-TERS
config and override the data fields.

| Field | Requirement |
| --- | --- |
| `data.root`, `data.image_key` | HDF5 directory and image dataset name |
| `data.cache_dir` | where the graph cache is written; one per dataset |
| `data.normalize` | `per_channel` or `corpus` |
| `data.info_name`, `data.atom_types` | an entry in `datasets/datasets_config.py`, and its element count |
| `model.ters_in_channels` | image channels after slice selection |
| `training.n_iters`, `training.snapshot_freq` | `n_iters` must be a multiple of `snapshot_freq` |
| `eval.begin_ckpt`, `eval.end_ckpt` | checkpoint N is saved at step N × `snapshot_freq` |
| `eval.num_samples` | at most the size of the test split |

The auxiliary config only needs the main config with `data.load_images = False`.

## Conventions and limitations

- **Sampling is given each target's atom count.** The metrics measure structure recovery with a
  known atom count.
- **Position scale.** Positions are in Å (`normalize_factors = '1, 4, 4, 1'`), and
  `spatial_cut_off = 2.0` Å², both as in DiffSpectra. The cutoff marks atom pairs within 1.41 Å as
  neighbours for one attention head. That covers C–H, double and aromatic bonds, but not C–C single
  bonds. Fine-tuning from DiffSpectra requires `2.0`.
- **Stereochemistry.** Graphs carry no stereochemistry.
