# Tools

Tools are grouped by task. TERS and STM share the evaluation, diagnostic, and visualization commands unless a command says otherwise.

## Main workflow

1. Generate cube-conditioned candidates with `python sample.py --modality ters|stm ...` (repository root).
2. Score recovery against target molecules with `python -m eval_sampled_mols.paper_metrics --cells NAME=/path/to/mols_test.pkl`.
3. Score validity, stability, and distributional quality with `python -m eval_sampled_mols.eval_cond --samples /path/to/processed_test.pkl`.
4. Merge independently generated shards with `python merge_shards.py ...` (repository root) before scoring.

Run generation and both scoring scripts with `python -m eval_sampled_mols.run_metrics --ckpt CHECKPOINT --modality ters|stm`. It scores matched test samples by default; use `--cells` to select validation or shuffled controls. The Slurm examples live in `slurm/`. See the [metric guide](../docs/metrics.md).

## Files saved by main.py evaluation

The separate molecule lists from `main.py --mode eval --save_mols` are scored from the
repository-level `eval_sampled_mols/` folder alongside the current scoring scripts:

```bash
python -m eval_sampled_mols.eval_saved_mols --samples-dir /path/to/molecules_ckpt_20
python -m eval_sampled_mols.eval_rmsd --samples-dir /path/to/molecules_ckpt_20
```

`eval_saved_mols` calls the upstream similarity scorer and writes `metrics_results/similarity_metrics.csv`.
`eval_rmsd` compares generated coordinates with their targets and writes `rmsd.json`.
Reusable alignment functions live in `evaluation/rmsd.py`. These commands use explicit
checkpoint folders; they do not assume a particular experiment or checkpoint number.

Runnable evaluators live in `eval_sampled_mols/`; reusable metric functions live in `evaluation/`. The shared upstream target-similarity scorer remains in `compute_metrics.py`.

## Folders

- `diagnostics/`: conditioning, checkpoint progress, split, stability calibration, and representation probes.
- `analysis/`: STM corpus analysis and shared planarity calculations.
- `visualization/`: molecule renderings, generated geometry plots, and paper figures.
