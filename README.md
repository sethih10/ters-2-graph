# TERS-2-Graph

TERS-2-Graph generates molecular graphs and 3D atomic coordinates conditioned on
tip-enhanced Raman spectroscopy (TERS) hyperspectral image cubes or scanning
tunneling microscopy (STM) image stacks. It adapts
[DiffSpectra](https://github.com/AzureLeon1/DiffSpectra) with image encoders,
dataset preprocessing, conditional generation, and molecular evaluation.

The workflow is to prepare graph targets from HDF5 datasets, train a conditional
diffusion model, generate candidate molecules, and evaluate their recovery and
chemical quality. Configurations are provided for TERS, STM, and QM9S-TERS.

**Sampling currently uses the target molecule's atom count.** Evaluation therefore
measures structure recovery with a known atom count; it does not establish
recovery from an image alone.

## Setup

Run the commands below from the repository root. You will need:

- A Python environment with PyTorch, PyTorch Geometric, RDKit, ASE, and HDF5 support.
- A CUDA-capable GPU for the supplied training configurations and `sample.py`.
- Split HDF5 datasets, supplied separately. Generation also requires a trained checkpoint.

Install the dependencies with [env.sh](env.sh) from a Conda-enabled shell:

```bash
source env.sh
```

The script creates and activates the `ters-2-graph` Conda environment and installs
the packages needed for training, generation, and evaluation.

The script targets Python 3.10, PyTorch 2.3.1 with CUDA 12.1, and PyTorch Geometric
2.4.0. Adjust the PyTorch and PyG builds if your platform requires different ones.
Its MOSES clone command uses GitHub SSH access; change it to HTTPS if needed.
For later sessions, activate the environment with `conda activate ters-2-graph`.

### Machine paths and logging

Copy the environment template, then replace its example paths:

```bash
cp .env.example .env
```

```dotenv
DATA_ROOT=/absolute/path/to/data
OUTPUT_ROOT=/absolute/path/to/outputs
WANDB_API_KEY=
```

[env.py](env.py) loads `.env` when resolving paths. Exported environment variables
take precedence. `DATA_ROOT` locates datasets; `OUTPUT_ROOT` supplies default
output and pretrained-encoder locations. Explicit `--workdir` and `--out`
arguments select the output directory for a particular command.

Training configurations enable online Weights & Biases logging by default.
Supply a `WANDB_API_KEY`, use `--config.training.wandb.mode=offline`, or disable
logging with `--config.training.wandb.enabled=False`.

## Prepare the data

Each dataset consists of `train.h5`, `val.h5`, and `test.h5`, paired with
`graphs_train.pt`, `graphs_val.pt`, and `graphs_test.pt`. These graph files
("sidecars") store molecular targets and their corresponding HDF5 row indices.
The preprocessing scripts infer bonds from atomic geometry, encode the graphs,
and validate the records.

The configurations use these locations by default:

| Dataset | HDF5 directory | Graph directory | Image key |
| --- | --- | --- | --- |
| TERS | `DATA_ROOT/planar_hdf5_0.5` | `datasets/ters` in this repository | `channels_100` |
| STM | `DATA_ROOT/stm-data` | `DATA_ROOT/stm-graphs` | `<split>/X` |
| QM9S-TERS | `DATA_ROOT/QM9S-dataset/qm9s_hdf5_64px_5A` | `DATA_ROOT/qm9s-ters/graphs` | `channels_100` |

Prepare the dataset you intend to use. Replace `/absolute/path/to/data` in these
commands with your `DATA_ROOT`; Python's `.env` loading does not set shell variables.

```bash
# TERS: H/C/N/O targets
python -m datasets.preprocess_ters \
    --data-dir /absolute/path/to/data/planar_hdf5_0.5 \
    --out-dir datasets/ters --stats

# STM: H/C/N/O/F/Si/P/S/Cl/Br targets
python -m datasets.preprocess_stm \
    --data-dir /absolute/path/to/data/stm-data \
    --out-dir /absolute/path/to/data/stm-graphs --stats

# QM9S-TERS: H/C/N/O/F targets
python -m datasets.preprocess_ters \
    --data-dir /absolute/path/to/data/QM9S-dataset/qm9s_hdf5_64px_5A \
    --out-dir /absolute/path/to/data/qm9s-ters/graphs \
    --elements hcnof --stats
```

To inspect an unfamiliar HDF5 file before preprocessing:

```bash
python -m datasets.preprocess_ters --h5 /path/to/train.h5 --inspect
python -m datasets.preprocess_stm --inspect /path/to/train.h5
```

Both preprocessors expose `--help` and a `--self-test` that requires no external data.

## Configure and train

Use a matching pair of configurations. The main configuration controls the model,
data, training, and sampling. The auxiliary configuration supplies reference graph
datasets for evaluation; the flag retains its upstream name, `--config_original_qm9`.

| Dataset | Main configuration | Auxiliary configuration |
| --- | --- | --- |
| TERS | [diffspectra_ters.py](configs/diffspectra_ters.py) | [base_ters.py](configs/base_ters.py) |
| STM | [diffspectra_stm.py](configs/diffspectra_stm.py) | [base_stm.py](configs/base_stm.py) |
| QM9S-TERS | [diffspectra_ters_qm9s.py](configs/diffspectra_ters_qm9s.py) | [base_ters_qm9s.py](configs/base_ters_qm9s.py) |

Check `data.root` and `data.graph_root` in both configurations. If you relocate a
dataset, update both paths consistently. Match the element set, image channel count,
and encoder architecture to your data and any checkpoint you load.

For example, train an STM model with an encoder initialized from scratch and
W&B logging disabled:

```bash
python main.py \
    --config configs/diffspectra_stm.py \
    --config_original_qm9 configs/base_stm.py \
    --config.model.pretrained_stm_encoder_path='' \
    --config.training.wandb.enabled=False \
    --mode train \
    --workdir /path/to/run
```

The default STM configuration uses the `clip` encoder and expects pretrained
weights at `OUTPUT_ROOT/runs/clip-full-1264302/cube_encoder_best.pt`. To use
pretrained weights, replace the empty override with your encoder checkpoint path.
For TERS or QM9S-TERS, select the corresponding configuration pair and omit the
STM encoder override.

Configuration fields can be overridden as `--config.FIELD=VALUE`, for example
`--config.training.batch_size=16`. Review batch size and worker count for your
hardware. Multi-GPU training defaults to DataParallel.

Training writes logs to `stdout.txt`, numbered checkpoints to `checkpoints/`,
and a periodic resume checkpoint to `checkpoints-meta/checkpoint.pth` under the
work directory. Reusing that work directory resumes from the periodic checkpoint
when it exists.

## Generate candidates

Generate five candidates for each of twenty test targets using a trained STM
checkpoint:

```bash
python sample.py \
    --modality stm \
    --ckpt /path/to/run/checkpoints/checkpoint_6.pth \
    --config.model.pretrained_stm_encoder_path='' \
    --split test --n-mols 20 --n-samples 5 \
    --batch-mols 4 \
    --out /path/to/samples
```

Use an existing checkpoint filename and the architecture settings from its
training run. The empty encoder-path override avoids requiring the separate
pretraining file; the full model weights are restored from `--ckpt`. Sampling
uses the checkpoint's exponential moving average (EMA) weights.

For TERS, use `--modality ters` and omit the STM encoder override. For QM9S-TERS,
also pass its configuration pair explicitly:

```bash
python sample.py \
    --modality ters \
    --config configs/diffspectra_ters_qm9s.py \
    --config_original_qm9 configs/base_ters_qm9s.py \
    --ckpt /path/to/qm9s-run/checkpoints/checkpoint_6.pth \
    --split test --n-mols 20 --n-samples 5 \
    --batch-mols 4 --out /path/to/qm9s-samples
```

`--batch-mols` limits the number of targets per generation batch; each target
produces `--n-samples` candidates. Use `--split val` for validation targets and
`--shuffle-cubes` to pair targets with other molecules' images as a null control.

The output directory contains:

| File | Contents |
| --- | --- |
| `samples_<split>.json` | Per-target recovery results and summary metrics. |
| `mols_<split>.pkl` | Target and generated RDKit molecules for similarity scoring. |
| `processed_<split>.pkl` | Generated atom, bond, and coordinate data for quality metrics. |
| `geometry_<split>.pkl` | Geometry and image projections for visualization; omitted with `--no-geometry`. |

Shuffled runs add `_shuf` before the file extension. Use `--save-sdf` for portable
SDF exports. If `--out` is omitted, samples go to `generated/` beside the run's
`checkpoints/` directory.

For independent generation jobs, use the same sampling settings and seed with
`--n-shards N --shard I`, where `I` runs from `0` to `N-1`, and give each shard its
own output directory. Merge the outputs before scoring:

```bash
python merge_shards.py \
    --shards /path/to/shard0 /path/to/shard1 \
    --split test --out /path/to/merged
```

Add `--tag _shuf` when merging shuffled controls.

## Evaluate results

For standard STM or TERS configurations, generate and score a checkpoint in one
command:

```bash
python -m eval_sampled_mols.run_metrics \
    --modality stm \
    --ckpt /path/to/run/checkpoints/checkpoint_6.pth \
    --config.model.pretrained_stm_encoder_path='' \
    --n-mols 20 --n-samples 10 --batch-mols 4 \
    --out /path/to/evaluation \
    --csv /path/to/evaluation/recovery.csv
```

This runs target-similarity scoring and molecular-quality scoring on matched test
samples. Use `--cells test test_null val val_null` to include validation and
shuffled-image controls. Each cell gets its own subdirectory under `--out`.
`--dry-run` prints the commands without executing them. `--skip-sampling` reuses
existing molecule files in those subdirectories; missing cells are still generated.
Keep `--n-samples` at least as large as the largest requested `--ks` value.

To score the five-candidate outputs from `sample.py` directly:

```bash
python -m eval_sampled_mols.paper_metrics \
    --cells stm_test=/path/to/samples/mols_test.pkl \
    --ks 1 3 5 --out /path/to/samples/recovery.csv

python -m eval_sampled_mols.eval_cond \
    --samples /path/to/samples/processed_test.pkl
```

Use these separate commands for custom configurations, including QM9S-TERS;
`run_metrics` selects the standard STM or TERS configuration pair. The processed
sample file records the auxiliary configuration used for reference metrics.

- `paper_metrics` measures exact target recovery and top-k molecular similarity.
- `eval_cond` measures chemical validity, valence/stability, predicted-bond geometry,
  and distribution metrics such as FCD, SNN, and diversity.
- `eval_saved_mols` and `eval_rmsd` score the separate molecule-list format produced
  by `main.py --mode eval`. See the [tools guide](tools/README.md).

Distance-inferred Metric-3D has known limitations for STM. Interpret it alongside
predicted-bond valence and geometry checks.

## Frozen-denoiser fine-tuning

See [the frozen-denoiser guide](docs/frozen-denoiser.md) for source-compatible
configs, CPU load/step/data validation, a fresh-workdir training example, and
sampling with the matching EMA parameter set. DiffSpectra-to-STM vocabulary
expansion is not supported.

## Validation and development

Run the preprocessing self-tests in the configured Python environment. They use
synthetic records and require no external datasets:

```bash
python -m datasets.preprocess_ters --self-test
python -m datasets.preprocess_stm --self-test
```

| Path | Purpose |
| --- | --- |
| `configs/`, `datasets/` | Experiment settings, HDF5 readers, graph preprocessing, and batching. |
| `models/`, `diffusion/`, `sampling/` | Image encoders, molecular diffusion models, noise schedules, and generation. |
| `main.py`, `run_lib.py`, `losses.py` | Training and evaluation entry point, loops, and losses. |
| `sample.py`, `merge_shards.py` | Checkpoint-based generation and shard merging. |
| `evaluation/`, `eval_sampled_mols/` | Reusable metric functions and evaluation commands. |
| `tools/` | Diagnostics, analysis, and visualization. |
