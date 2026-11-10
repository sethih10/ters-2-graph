# TERS token encoders

Two optional conditioning encoders for TERS cubes, selected with `config.model.ters_encoder`:

| Value | Class | What it does |
|---|---|---|
| `v1` (default) | `TersEncoder` | The existing encoder. Nothing in this folder is imported. |
| `token` | `TersTokenEncoder` | CNN to an 8 x 8 grid of tokens, a small transformer, CLS readout. One vector per molecule. |
| `token_xattn` | `TersTokenXAttn` | The same encoder, plus cross-attention from every atom to the 64 tokens in every DMT block. |

## Motivation

**Token encoder.** A pooled CNN averages the image into one vector and loses where each signal was. The token encoder changes four things:

- A per-pixel spectral MLP (1x1 convolutions) mixes the wavenumber channels before any spatial convolution, so spectra of neighbouring pixels are not blurred together first.
- Position is explicit: x, y and radius channels are added before the stem, and the tokens get a learned position embedding.
- The first stage runs at full resolution and downsampling stops at 8 x 8.
- The readout is a CLS token that attends over the grid, not an average.

**Cross-attention adapters.** With one conditioning vector, every atom receives the same information. The adapters let each atom query the 64 spatial tokens itself. In TERSFormer this roughly doubled top-1 accuracy over the vector-only encoder (about 17% to 30% on its planar QM9S subset, with different cube preprocessing; not yet measured in this repository).

## Architecture

```
TERS cube (B, 100, 64, 64)
   |
   |  TersTokenEncoder                                          2.96 M parameters
   |    spectral MLP    1x1 convs, per pixel     (B, 128, 64, 64)
   |    + x, y, r channels                       (B, 131, 64, 64)
   |    stem            3x3 conv + ResBlock      (B,  32, 64, 64)
   |    3 ResBlocks, stride 2                    (B, 256,  8,  8)
   |    flatten + position embedding             (B, 64, 256)
   |    + CLS, 2 transformer layers              (B, 65, 256)
   |
   +--> CLS -> LayerNorm -> Linear -> vector (B, 256) -> cond_lin -> added to the time embedding
   |
   +--> 64 tokens (B, 64, 256)                                  token_xattn only
           -> one CrossAttnAdapter in front of each DMT block   2.11 M parameters for 8 blocks
              h <- h + tanh(gate) * attention(q = LayerNorm(h), k = v = tokens) * node_mask
```

- **Vector route.** The vector goes through the existing `cond_encoder -> cond_lin -> time embedding` path. `DMT.forward` is unchanged.
- **Token route.** A forward pre-hook on each `e_block_i` replaces the incoming node features `h` with the adapter output. The node, edge and coordinate updates of that block all read the adapted `h`.
- **Identity at initialisation.** Every gate starts at 0, so a `token_xattn` model starts with exactly the output of the `token` model with the same weights.
- **Equivariance.** Queries come from the invariant node features only, never from coordinates.
- **Multi-GPU.** The encoder hands its tokens to the hooks through a thread-local, so each `DataParallel` replica uses its own tokens and adapters.

## What was added

Everything is additive. Existing configs and run commands behave as before.

| Path | Content |
|---|---|
| `models/ters_tokens/encoder.py` | `TersTokenEncoder` |
| `models/ters_tokens/adapters.py` | `CrossAttnAdapter`, `TersTokenXAttn`, the block pre-hook |
| `models/ters_tokens/__init__.py` | `build_encoder(config, dmt)`, `load_trunk` |
| `models/ters_tokens/export.py` | converts a frozen-run checkpoint into a standard one |
| `models/ters_tokens/check.py` | self-check on CPU |
| `configs/ters_tokens.py` | the `model.ters_*` keys, shared by the four configs below |
| `configs/qm9s_ters_frozen_token[_xattn].py` | QM9S TERS, pretrained denoiser, for `main_finetune.py` |
| `configs/diffspectra_ters_qm9s_token[_xattn].py` | QM9S TERS from scratch, for `main.py` |
| `models/dmt.py` | one branch in `DMT.__init__` that calls `build_encoder` when `ters_encoder` is not `v1` |
| `main_finetune.py` | reads `model.freeze_denoiser` (default `True` = the previous behaviour) |

All new parameters are stored under `cond_encoder.`, so `main_finetune.py` trains them without further changes.

## Training

Replace `token_xattn` by `token` in a config name for the vector-only encoder. All commands also take `--config_original_qm9 configs/base_ters_qm9s.py`.

| Mode | Command | Trains | Starts from |
|---|---|---|---|
| Frozen | `main_finetune.py --config configs/qm9s_ters_frozen_token_xattn.py` | encoder, adapters, `cond_lin` (5.3 M) | pretrained DiffSpectra denoiser |
| Joint fine-tuning | the frozen command plus `--config.model.freeze_denoiser=False` | everything (32.9 M) | pretrained DiffSpectra denoiser |
| From scratch | `main.py --config configs/diffspectra_ters_qm9s_token_xattn.py` | everything (32.9 M) | random weights |

```
python main_finetune.py --mode train --workdir RUN \
    --config configs/qm9s_ters_frozen_token_xattn.py --config_original_qm9 configs/base_ters_qm9s.py
```

Notes:

- **Frozen** is the cheapest and the mode used in TERSFormer. The pretrained checkpoint is the one described under fine-tuning in the main README.
- **Joint fine-tuning** changes the pretrained denoiser, so use a lower rate and restore its dropout: `--config.optim.lr=2e-5 --config.model.dropout=0.1`. These values come from TERSFormer and are not yet tested here.
- **Two stages.** To train `token_xattn` on top of a trained `token` encoder, add `--config.model.ters_encoder_init=TOKEN_RUN/checkpoints/checkpoint_12.pth`.
- **Frozen, then joint.** Export the frozen run's checkpoint (below) and start a new run with `--config.model.pretrained_denoiser_path=OUT.pth --config.model.load_pretrained_cond_encoder=True --config.model.freeze_denoiser=False`.

## Checkpoints

Training writes `RUN/checkpoints/checkpoint_<k>.pth` every `training.snapshot_freq` steps. How to use one depends on the mode that wrote it:

| Written by | EMA covers | Evaluate | Generate with `sample.py` |
|---|---|---|---|
| Frozen run | trained weights only | `main_finetune.py --mode eval --workdir RUN` | after export |
| Joint fine-tuning | all weights | `main_finetune.py --mode eval --workdir RUN --config.model.freeze_denoiser=False` | yes |
| From scratch | all weights | `main.py --mode eval --workdir RUN` | yes |

Use the config and flags of the training run. `--config.eval.ckpts=<k>` selects the checkpoint.

A frozen-run checkpoint stores the EMA only for the trained weights, so `sample.py` and `main.py` misread it without an error. Convert it first:

```
python -m models.ters_tokens.export RUN/checkpoints/checkpoint_12.pth OUT.pth
```

`OUT.pth` has the EMA weights merged in and no optimizer state. It works with `sample.py --ckpt OUT.pth --config <the training config> --config_original_qm9 configs/base_ters_qm9s.py` and as `model.pretrained_denoiser_path`. It cannot resume training or be read by `--mode eval`, which need the optimizer state.

**Shared weights.** No trained weights are published yet. Keep exported files as `$DATA_ROOT/ters_tokens/<name>.pth` and name the file after its config, for example `qm9s_frozen_token_xattn.pth`. Do not commit them to the repository.

## Limits

- Cubes must be 64 x 64. Other sizes raise an error.
- `ters_encoder_init` copies the raw encoder weights, not the EMA weights, and does not copy `cond_lin`.
- Adapter gates are not logged during training. Read them from a checkpoint under `cond_encoder.adapters.<block>.gate`.

## Check

```
python -m models.ters_tokens.check
```

Runs on CPU with random inputs and no dataset. It checks shapes and parameter counts, identity at initialisation, masking of padded atoms, gate gradients, per-thread tokens, warm start, strict reloading and the checkpoint export.
