# SOPHIA: state-conditioned steering of latent reasoning transitions

SOPHIA segments a reasoning model's chain of thought into steps, clusters the
steps into K=5 latent states, and builds one steering direction per state: the
mean residual activation of steps that leave the state (crossers) minus that
of steps that stay (stayers). At inference, when two consecutive steps fall
in the same state (a self-loop, z_{i-1} = z_i), that state's direction is
added to the residual stream for the next step.

This repository reproduces the held-out evaluation: does the next step leave
the current state? It compares SOPHIA with greedy decoding, a norm-matched
random direction, the reversed direction, and a single global (state-agnostic)
direction, on Qwen3-1.7B, Qwen3-4B-Thinking-2507 and Qwen3-8B, over gsm8k,
AQuA, LogiQA and MATH, under 5-fold trace-level cross-fitting.

No traces, caches, vectors or results are included; everything is regenerated
under `runs/` (git-ignored).

## Setup

Two Python environments (tested with Python 3.12):

```bash
pip install -r requirements.txt          # main: every stage except generation
pip install -r requirements-vllm.txt     # separate env for trace generation (vLLM)
```

Models are pulled from the Hugging Face hub: the three steering targets, plus
`Qwen/Qwen3-4B-Base`, which embeds steps for clustering. For LogiQA, download
`Train.txt` to `data/logiqa_train.txt` (see `data/README.md`). The other three
datasets download automatically.

Run every command from the repository root.

## Pipeline

| Stage | Module | Output in `runs/<model>/` |
|---|---|---|
| 1. traces | `sophia.gen_traces` | `raw/<ds>.jsonl`: 500 traces per dataset |
| 2. segment | `sophia.build_segmented` | `segmented/<ds>_segmented.jsonl`: steps, discourse-marker labels, `is_correct` |
| 3. embed | `sophia.embed_steps` | `work/<ds>/embeddings/<sid>.pt`: mean-pool step features (Qwen3-4B-Base) |
| 4. full-data K-means | `sophia.cluster_full` | `clustered/<ds>_clustered.jsonl`: reference partition (`state_labels`) |
| 5. residual cache | `sophia.extract_states` | `step_states/<ds>.pt`: target's residual at each step's marker token |
| 6. full-data vectors | `sophia.full_vectors` | `full_vectors/steering_vectors_<ds>.pt`: reference for stability checks |
| 7. cross-fit build | `sophia.crossfit_build` | `crossfit/<ds>/fold{0..4}.pt`, `heldout_labels.jsonl`, `build_summary.json` |
| 8. held-out eval | `sophia.crossfit_eval` | `crossfit/<ds>/hit_rate_results_<rule>_a0.05_seed42.jsonl` |

The scripts wrap these stages: `scripts/generate.sh` (stage 1),
`scripts/prep.sh` (stages 2–7) and `scripts/eval.sh` (stage 8).

## Reproducing the experiments

| Run dir | Steering target | Patch layers (upper half of the decoder) |
|---|---|---|
| `runs/qwen3_1p7b` | `Qwen/Qwen3-1.7B` | `14-27` (28 layers) |
| `runs/qwen3_4b` | `Qwen/Qwen3-4B-Thinking-2507` | `18-35` (36 layers) |
| `runs/qwen3_8b` | `Qwen/Qwen3-8B` | `18-35` (36 layers) |

For one model (Qwen3-8B shown):

```bash
# 1. traces, in the vLLM environment
VLLM_PYTHON=/path/to/vllm-env/bin/python scripts/generate.sh 0 Qwen/Qwen3-8B runs/qwen3_8b

# 2-7. offline stages (one GPU for embedding and the residual cache)
scripts/prep.sh runs/qwen3_8b Qwen/Qwen3-8B 18-35 cuda:0

# 8. held-out eval, main protocol (prev_same). The steering target and the
#    Qwen3-4B-Base re-clusterer may share one GPU or be split across two.
scripts/eval.sh runs/qwen3_8b Qwen/Qwen3-8B 18-35 cuda:0 cuda:1 gsm8k aqua logiqa math

# robustness variant: prefixes selected by z_i == z_{i+1}
STAYER_RULE=next_same scripts/eval.sh runs/qwen3_4b Qwen/Qwen3-4B-Thinking-2507 18-35 cuda:0 cuda:0 gsm8k aqua logiqa math
```

To spread the work over GPUs, start one `generate.sh` per GPU with different
dataset lists (datasets already written are skipped), and one `eval.sh` per
GPU with different datasets. `DS=aqua scripts/prep.sh ...` runs prep for a
subset of datasets.

Reports (written to `reports/`):

```bash
python analysis/report_scale.py --model Qwen3-1.7B=runs/qwen3_1p7b \
  --model Qwen3-4B=runs/qwen3_4b --model Qwen3-8B=runs/qwen3_8b --reference Qwen3-4B
python analysis/make_latex.py --model Qwen3-1.7B=runs/qwen3_1p7b \
  --model Qwen3-4B-Thinking-2507=runs/qwen3_4b --model Qwen3-8B=runs/qwen3_8b
python analysis/compare_rules.py --work-dir runs/qwen3_4b/crossfit   # needs both rules
python analysis/diagnostics.py --run-dir runs/qwen3_4b                 # position, self-loop, marker checks
```

- **`report_scale.py`** reports, per model:
  - prefixes evaluated
  - exit rates for every condition
  - SOPHIA vs Greedy and vs Global (paired exact McNemar)
  - pooled results
  - fold clustering stability and fold vs full-data vector cosine
  - no-marker rates

  It also flags cells where SOPHIA doesn't beat Greedy, where Global ≥
  SOPHIA, where clustering is unstable, and where no-marker rates shift by
  ≥ 10 points from the reference model.
- **`make_latex.py`** writes the per-cluster table and the per-dataset/pooled
  table.

### Reference numbers

Pooled held-out exit rate (%) from our runs, `prev_same` rule:

| Model | Greedy | Random | Negative | Global | SOPHIA |
|---|---|---|---|---|---|
| Qwen3-1.7B | 43.8 | 43.3 | 31.0 | 41.7 | 56.7 |
| Qwen3-4B-Thinking-2507 | 41.9 | 41.5 | 24.8 | 49.3 | 58.1 |
| Qwen3-8B | 42.5 | 45.0 | 11.7 | 42.5 | 57.3 |

### Compute (RTX A6000, 48 GB)

| Stage | Time | Memory |
|---|---|---|
| Generation | about 2–4 h per model on one GPU; MATH is the longest | — |
| Prep | about 1–1.5 h per model | — |
| Eval | 20–80 min per dataset; MATH is the longest | about 20 GB for the 4B target plus re-clusterer; the 8B target needs ≈ 22 GB, plus ≈ 11 GB for the re-clusterer |

## Protocol (identical for every model)

- **Traces:** the first 500 training questions per dataset; temperature 0.6,
  top_p 0.95, seed 0, at most 16,384 new tokens. Each model is evaluated on
  its own traces.
- **Steps and states:**
  - steps are split at sentence-initial discourse markers (`sophia.segment_cot`);
  - step features are last-layer mean-pool from Qwen3-4B-Base, which also
    re-clusters regenerated steps;
  - features are z-scored, then K-means with K=5 and seed 42;
  - n_init is 10 for the full-data partition and 50 inside folds.
- **Vectors:** crosser-minus-stayer per state, from residual states at the
  first token of each step's discourse marker. Intro steps and final steps are
  excluded. The global vector pools all states.
- **Cross-fitting:** 5 folds over traces, stratified by correctness (fold
  seed 0). z-scoring, K-means and all vectors are fit on the other folds.
  Each held-out trace is labelled and evaluated with its own fold's
  artifacts.
- **Prefixes:** `prev_same` (z_{i-1} = z_i, i ≥ 2) in held-out incorrect
  traces, up to 30 per state (selection seed 42).
- **Intervention:** α = 0.05 times the unit direction, scaled per layer by the
  mean stayer-activation norm. It's added at the last position of every
  forward pass while the next step is decoded greedily, up to 200 tokens.
  **α is the same for every model and was not tuned.**
- **Scoring:** an exit is a regenerated step assigned to a different state. A
  generation with no step boundary within 200 tokens ("no marker") counts as
  not exiting. Conditions are compared with exact McNemar tests on the paired
  prefixes.

## Model-specific settings

1. **Steering target:** hybrid `Qwen3-1.7B` / `Qwen3-8B` in thinking mode.
   There's no Thinking-2507 release at these sizes.
2. **Opening `<think>`:** hybrid chat templates end the generation prompt at
   `assistant\n`. `"<think>\n"` is appended (for generation and eval prefixes)
   so every model sees the same prompt shape as Qwen3-4B-Thinking-2507.
3. **Patch layers:** the upper half of the decoder, as in the table above.

## Notes

- **Cluster ids are arbitrary and model-local.** One of the five clusters is
  usually the 500 `<think>` intro steps. It has no vector (intro steps are not
  cached), so most model × dataset cells have four evaluable states. On
  Qwen3-4B MATH the intro steps merge into another cluster, leaving five.
- **K-means is not bit-reproducible across BLAS thread counts.** Rebuilding the
  Qwen3-4B gsm8k cross-fit with a different thread count changed 6 of 11,799
  held-out step labels; fold vectors kept cosine ≥ 0.998. `scripts/prep.sh`
  caps BLAS threads at 16 by default, which also avoids an OpenBLAS abort
  (`double free or corruption`) on machines with more than 64 cores.
- **MATH clustering is unstable under resampling.** It has two nearly
  equal-inertia K=5 partitions, and some folds pick the other one. The
  `build_summary.json` agreement numbers show this.
- **Shared GPUs:** `scripts/eval.sh` retries a crashed dataset (usually OOM)
  and resumes from the rows already written.
- **vLLM compile errors:** if vLLM's Triton compilation fails for lack of
  Python headers, point `C_INCLUDE_PATH` at a Python 3.12 include directory.

## Validation against the original runs

This code was checked against the outputs that produced the results above:

- **Segmentation and the task loader:** identical output.
- **Full-data clustering:** 100% agreement up to relabelling.
- **Full-data vectors:** bit-identical.
- **Cross-fit build:** matches up to the K-means rounding above.
- **Held-out eval:** a smoke run reproduced 20/20 original rows exactly
  (regenerated text, state and exit outcome).
- **Report and LaTeX scripts:** reproduce the reported tables.
