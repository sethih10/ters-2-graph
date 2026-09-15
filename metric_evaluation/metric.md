# Metric evaluation — the DiffSpectra suite, applied to the STM run

**What this folder is.** The evaluation DiffSpectra reports, run against our own checkpoints, with
a null control the paper does not have. `run_metrics.py` orchestrates it; this file says what each
number means and — more usefully — what each number *cannot* tell you.

**Nothing here reimplements a metric.** Every per-pair definition is upstream's, imported:

| Layer | File | What it provides |
| --- | --- | --- |
| per-pair chemistry | [`compute_metrics.py`](../compute_metrics.py) | the seven metrics below, at the paper's parameters |
| top-k + null aggregation | [`tools/paper_metrics.py`](../tools/paper_metrics.py) | what upstream cannot express (see §3) |
| generation | [`tools/sample_cond.py`](../tools/sample_cond.py) | candidates, matched or shuffled cubes |
| distributional | [`tools/eval_cond.py`](../tools/eval_cond.py) | FCD, SNN, Frag, Scaf, IntDiv, stability, novelty. **Metric-2D only** — Metric-3D scores 0.250 on ground-truth structures (2D: 0.974), so it is not quotable on this corpus |
| orchestration | [`run_metrics.py`](run_metrics.py) | this folder — invokes the above, computes nothing |

---

## 1. The seven metrics

All are computed between the **true** molecule and one **generated** candidate. Higher is better
except MCES.

| Metric | Definition | Parameters | What it is sensitive to |
| --- | --- | --- | --- |
| **Exact / top-k** | InChIKey string equality | — | the whole molecule, including stereochemistry-free connectivity. Binary and unforgiving |
| **Tanimoto (Morgan)** | Tanimoto over Morgan bit vectors | radius 2, 2048 bits | local circular environments; the standard "are these the same kind of molecule" score |
| **Cosine (Morgan)** | Cosine over the same fingerprint | radius 2, 2048 bits | as above, less penalising of size mismatch |
| **Tanimoto (MACCS)** | Tanimoto over 166 MACCS keys | — | presence/absence of curated substructures. **Runs high**: two unrelated aromatics share many keys |
| **Fraggle** | RDKit `FraggleSim.GetFraggleSimilarity` | defaults | similarity after fragmenting on rotatable bonds; credits a shared core with different decoration |
| **Functional Group** | Jaccard over the detected functional-group set | `FUNCTIONAL_GROUPS` in `compute_metrics.py` | chemistry class only. **Runs very high** — it ignores the skeleton entirely |
| **MCES** | Maximum Common Edge Subgraph distance | `threshold=100`, `always_stronger_bound=False` | graph edit distance in spirit. **Lower is better.** An ILP — the slow column |

Each is reported twice per cell:

- **(best)** — the best of the k candidates for each target, then averaged over targets. This is
  the "we get 10 guesses" number.
- **(mean)** — the mean over all candidates, then averaged. This is what **one** random sample is
  worth, and it is the honest number if you only ever get one shot.

### Reading them together

The four similarity metrics are **not** four independent pieces of evidence — they are ordered by
how much of the molecule they are willing to ignore:

    Functional Group  >  MACCS  >  Fraggle  >  Tanimoto (Morgan)  >  exact match
    (most forgiving)                                                (least)

A run that scores 0.97 Functional Group and 0.00 exact has produced *the right kind of chemistry
and the wrong molecule*. That is a real and common result here, and reporting only the left-hand
end of that chain would be misleading.

---

## 2. The protocol

**20 spectra × 10 samples per cell, 4 cells.** Identical to the TERS protocol in
[`docs/experimental.md`](../docs/experimental.md) §2d, deliberately — the point of running this is
to compare, and a different target count changes the granularity of top-k (on 20 targets it moves
in steps of 0.05) and quietly makes the tables incomparable.

| cell | split | cubes |
| --- | --- | --- |
| `stm_test` | test | matched |
| `stm_test_null` | test | shuffled |
| `stm_val` | val | matched |
| `stm_val_null` | val | shuffled |

**10 samples is a floor, not a choice.** The paper's headline is top-10; scoring fewer candidates
than k makes it undefined. `run_metrics.py` refuses `--n-samples` below `max(--ks)` rather than
reporting a top-10 computed from five candidates.

### The null cells are the load-bearing half

`sample_cond.py --shuffle-cubes` pairs every target with **another molecule's** cube (a
derangement, so no target keeps its own). Target, atom count, noise draw and scoring are unchanged
— the conditioning is the only thing that differs.

This matters because **the sampler is handed the true atom count**
([`tools/sample_cond.py:98`](../tools/sample_cond.py#L98)) and the chemical space is narrow, so the
prior alone scores well above zero on every similarity metric. On TERS the shuffled null reached
Tanimoto MACCS 0.361 and Functional Group 0.512 while recovering *zero* targets. **A raw Tanimoto
is close to meaningless here; the matched-minus-shuffled gap is the measurement's contribution.**

> If a null cell scores well above zero on top-k, the protocol has leaked (pairing preserved
> somewhere) and the run is a bug report, not a result.

---

## 3. Why `compute_metrics.py` is not enough on its own

The shipped scorer, `evaluate_jsonl_predictions`, takes **one candidate per target** —
`pred_mol = pred_mol_list[0]` ([`compute_metrics.py:207`](../compute_metrics.py#L207)). The paper's
headline "top-10 = 0.9949" therefore cannot come from it as written. `tools/paper_metrics.py`
adds the top-k aggregation and the null column, and imports the per-pair chemistry unchanged so
the definitions cannot drift.

---

## 4. Running it

```bash
module load python-pytorch/2.10
export PYTHONPATH=$HOME/.cache/ters-overlay:$PWD

python3 metric_evaluation/run_metrics.py --dry-run        # print commands, touch nothing
sbatch metric_evaluation/run_metrics_gpumedium.sbatch     # full protocol, ~65 min of 2 h
```

Smoke test (proves the wiring, not the science):

```bash
python3 metric_evaluation/run_metrics.py --n-mols 3 --n-samples 3 --ks 1 3 --no-mces \
    --out /scratch/project_2001912/stm-eval/paper_metrics_smoke
```

`--skip-sampling` rescores existing pickles; `--no-mces` drops the ILP column, which dominates
runtime. Outputs land under `/scratch/project_2001912/stm-eval/paper_metrics/`, never in the repo
(`$HOME` is a 15 GB quota — see [`tools/paths.py`](../tools/paths.py)).

---

## 5. Results

Full protocol, job **1175516** (gpumedium, 9 min 45 s), checkpoint 3 (step 75,001) of STM run
1134521. 20 targets x 10 samples x 4 cells.

| | stm_test | **test null** | stm_val | **val null** |
| --- | --- | --- | --- | --- |
| **top-1 / 3 / 5 / 10** | **0.000** | 0.000 | **0.000** | 0.000 |
| Tanimoto Morgan (best) | 0.222 | 0.194 | 0.217 | 0.196 |
| Tanimoto Morgan (mean) | 0.128 | 0.109 | 0.123 | 0.103 |
| Cosine Morgan (best) | 0.358 | 0.319 | 0.355 | 0.327 |
| Tanimoto MACCS (best) | 0.527 | 0.435 | 0.496 | 0.432 |
| Fraggle (best) | 0.628 | 0.565 | 0.660 | 0.574 |
| Functional Group (best) | 0.744 | 0.630 | 0.653 | 0.611 |
| **MCES (best, lower better)** | **9.63** | 10.53 | **9.40** | 10.75 |
| MCES (mean, lower better) | 15.53 | 16.54 | 15.00 | 16.27 |

### The headline: zero

**top-k is 0.000 in every cell at every k, including the matched ones.** Out of 20 targets with
10 guesses each, on both splits, the model recovered **no molecule exactly**. TERS on the same
protocol reached top-10 = 0.350 (test) and 0.200 (val); DiffSpectra reports 0.9949 on QM9S.

### The gap to the null is consistent, small, and barely survives pairing

Matched beats its null on **every metric in both splits** — including MCES, where lower is better
and matched is lower. The direction is never wrong, which is what makes it worth reporting at all.
But the size is the story. Paired per target (same targets, same seed; only the pairing differs):

| split | metric | matched | null | diff | **targets improved** | paired t |
| --- | --- | --- | --- | --- | --- | --- |
| test | Tanimoto Morgan | 0.222 | 0.194 | +0.028 | **10 / 20** | +2.18 |
| test | Tanimoto MACCS | 0.527 | 0.435 | +0.093 | **11 / 20** | +1.92 |
| val | Tanimoto Morgan | 0.217 | 0.196 | +0.022 | **9 / 20** | +1.82 |
| val | Tanimoto MACCS | 0.496 | 0.432 | +0.065 | **13 / 20** | +2.19 |

**9-13 of 20 is a coin flip.** TERS improved **18/20** on the same test. The means differ because
a few targets improve a lot, not because most improve a little — and on 20 targets, t between
1.8 and 2.2 sits exactly at the threshold this repo's own diagnostics call indistinguishable
(`diagnose_conditioning`: "|t| < 2 => indistinguishable").

### This does not contradict the loss-level result — it localises it

The loss-level shuffled control on the same checkpoint is unambiguous: **t = +11.0**, conditioning
is active. Both are true, and together they say something sharper than either alone:

> The z-stack measurably changes the **denoising loss**, and does **not** measurably change the
> **molecules that come out**.

A +0.98% loss effect is real and far too small to move a 250-step ancestral sampler onto the right
molecule. The null cells score 0.194 Morgan / 0.435 MACCS on their own — that is the prior, given
the true atom count — and matched generation is not meaningfully above it.

### What this rules out

- **"It just needs more steps."** The fixed-t probe shows learning continuing at 75k, but what is
  improving is denoising at low noise, not molecular identification. Nothing here suggests the
  remaining 75k steps would move top-k off zero.
- **"The metrics are too strict."** They are not the binding constraint: the *forgiving* metrics
  (Functional Group 0.744, Fraggle 0.628) are also barely above their nulls (0.630, 0.565).

The open problem is the one the conditioning diagnostics already named: encoder embeddings at mean
pairwise cosine **0.984** cannot discriminate 23,123 test molecules. See
[`docs/09-research-summary.md`](../docs/09-research-summary.md) §5b.

### Caveats to carry into any comparison

None of these are like-for-like with the paper, and saying so is not hedging — it changes what the
numbers mean:

- **Training length.** Step **75,001 of a planned 150,000**; the run died on a Lustre project quota
  at step 88,000. Not a converged model.
- **No pre-training.** DiffSpectra's QM9S numbers come with a pre-trained encoder; this is from
  scratch.
- **Different inverse problem.** QM9S conditions on 1-D spectra of ~130k small molecules; this on a
  14-slice STM z-stack over 184k molecules with 10 elements.
- **20 targets** gives top-k a granularity of 0.05 and leaves paired tests underpowered. A 0.000
  top-10 is not "less than 0.05" — with 20 targets it is exactly zero recoveries, but the upper
  confidence bound is around 0.14, so this does not establish that top-10 is *far* below TERS's
  0.350, only that it is below it.
