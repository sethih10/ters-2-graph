# TERS-2-Graph

Generate molecular graphs and 3D coordinates from TERS hyperspectral image cubes
and STM z-stacks. This repository adapts
[DiffSpectra](https://github.com/AzureLeon1/DiffSpectra) with cube encoders,
dataset preprocessing, conditional sampling, and molecular evaluation.

## Repository structure

| Path | Purpose |
| --- | --- |
| `configs/` | Model, training, sampling, and dataset configurations for TERS, STM, and QM9S-TERS. |
| `datasets/` | HDF5 readers, graph preprocessing, transforms, and batch collation. |
| `models/` | Molecular diffusion models, TERS/STM encoders, transformer layers, and EMA. |
| `diffusion/` | Diffusion noise schedules. |
| `sampling/` | Diffusion samplers, dataset-conditioned pipelines, and molecule postprocessing. |
| `evaluation/` | Reusable validity, stability, geometry, distribution, and RMSD metrics. |
| `eval_sampled_mols/` | Commands for generating evaluation samples and scoring saved predictions. |
| `tools/` | Conditioning diagnostics, STM corpus analysis, and visualization; see the [tools guide](tools/README.md). |
| `main.py`, `run_lib.py`, `losses.py` | Training/evaluation entry point, execution loops, and loss functions. |
| `sample.py`, `merge_shards.py` | Checkpoint-based generation and merging independently generated shards. |
| `env.py`, `distributed.py`, `ters_wandb.py` | Machine settings, distributed training support, and W&B logging. |
| `compute_metrics.py`, `utils.py`, `visualize.py` | Shared similarity scoring, model utilities, and molecule grid rendering. |

## Setup

Use the [upstream dependency setup](https://github.com/AzureLeon1/DiffSpectra/blob/main/env.sh)
as a baseline, with PyTorch and PyTorch Geometric builds matching your CUDA
runtime. Core dependencies include RDKit, ASE, HDF5 support, and MOSES/FCD.
Install the additional configuration, logging, and scoring dependencies:

```bash
pip install --upgrade ml-collections python-dotenv wandb pulp myopic_mces
cp .env.example .env
```

Set your machine paths in `.env`:

```dotenv
DATA_ROOT=/path/to/data
OUTPUT_ROOT=/path/to/outputs
WANDB_API_KEY=
```

`env.py` loads this file automatically when paths are requested. Exported
variables take precedence, so `.env` is optional if those variables are already
set. A W&B key is needed only for online logging. Keep experiment settings in
`configs/`; explicit arguments such as `--workdir` and `--out` choose run outputs.

Raw image datasets and trained checkpoints must be supplied separately. Readers
pair split HDF5 files with `graphs_<split>.pt` graph sidecars. Inspect or prepare
these using `datasets/preprocess_ters.py` and `datasets/preprocess_stm.py`; both
provide `--help` and `--self-test`.

## Configuration

Use matching main and auxiliary configurations:

| Dataset | Main configuration | Auxiliary configuration |
| --- | --- | --- |
| TERS | `configs/diffspectra_ters.py` | `configs/base_ters.py` |
| STM | `configs/diffspectra_stm.py` | `configs/base_stm.py` |
| QM9S-TERS | `configs/diffspectra_ters_qm9s.py` | `configs/base_ters_qm9s.py` |

Review dataset locations and pretrained encoder paths before running. The STM
configuration defaults to a pretrained encoder; set
`model.pretrained_stm_encoder_path` to an empty string to train it from scratch.
When sampling, the model configuration must match the checkpoint architecture.
Optional field overrides use `ml_collections.config_flags` syntax:
`--config.FIELD=VALUE`.

## Training and generation

Run commands from the repository root. For example, train an STM model:

```bash
python main.py \
    --config configs/diffspectra_stm.py \
    --config_original_qm9 configs/base_stm.py \
    --mode train --workdir /path/to/run
```

Generate five candidates for each of twenty test targets:

```bash
python sample.py --modality stm \
    --ckpt /path/to/run/checkpoints/checkpoint_3.pth \
    --split test --n-mols 20 --n-samples 5 --out /path/to/samples
```

For TERS, use the TERS configuration pair when training and `--modality ters`
when sampling. QM9S-TERS sampling requires its configuration pair explicitly.
Use `--batch-mols` to limit generation batch size. For independent generation
processes, pass `--n-shards` and `--shard`, then combine their outputs before scoring:

```bash
python merge_shards.py --shards /path/to/shard0 /path/to/shard1 \
    --out /path/to/merged
```

Sampling saves target/prediction molecules in `mols_<split>.pkl`, processed
predictions in `processed_<split>.pkl`, and a JSON summary. Optional SDF export
uses `--save-sdf`; `--no-geometry` skips the separate visualization payload.

## Evaluation

Generate and score a checkpoint in one command:

```bash
python -m eval_sampled_mols.run_metrics --modality stm \
    --ckpt /path/to/run/checkpoints/checkpoint_3.pth --out /path/to/evaluation
```

Matched test samples are the default. `--dry-run` previews commands;
`--skip-sampling` scores existing outputs. Use
`--cells test test_null val val_null` to include validation and shuffled-cube controls.

| Command in `eval_sampled_mols/` | Purpose |
| --- | --- |
| `paper_metrics.py` | Target recovery and top-k molecular similarity from `mols_*.pkl`. |
| `eval_cond.py` | Chemical validity, valence/stability, predicted-graph geometry, and distribution metrics from `processed_*.pkl`. |
| `eval_saved_mols.py` | Similarity scoring for the separate molecule lists saved by `main.py` evaluation. |
| `eval_rmsd.py` | Coordinate alignment and RMSD for the separate saved molecule lists. |

Use `--help` for each command's input format. Geometry validity checks predicted
bonds against coordinates; geometry distribution metrics in `cal_geometry.py`
compare generated and reference populations. Distance-inferred Metric-3D has
known limitations on STM and should be interpreted alongside predicted-bond
valence and geometry checks.

The current sampling protocol uses each target molecule's atom count. Results
therefore measure structure recovery with a known atom count, rather than
inference from an unknown cube alone.

## Attribution and license

Based on [DiffSpectra](https://github.com/AzureLeon1/DiffSpectra)
([paper](https://arxiv.org/abs/2507.06853)). Third-party component provenance is
recorded in the corresponding source modules. See [LICENSE](LICENSE) for the
Apache-2.0 license.
