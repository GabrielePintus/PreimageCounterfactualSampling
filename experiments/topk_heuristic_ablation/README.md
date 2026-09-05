# CertCF top-k heuristic ablation

This experiment measures how often the strict nearest-anchor top-$k$ search
recovers the exact optimum over the same certified atlas, and how large the
optimality gap is when it does not.

## Fixed protocol

- deterministic balanced synthetic dataset with 32 features;
- trained `depth_03_width_128` classifier from the network-complexity grid;
- 10,000 balanced training points as the anchor-sampling pool;
- 500 randomly sampled anchors per predicted class;
- the existing 1,000 balanced test queries;
- $L_1$ atlas construction and $L_1$ projection;
- backward LiRPA with adaptive epsilon reduction;
- strict prefixes $k=1,\ldots,7$;
- sorted lower-bound search as the exact optimum over the fixed atlas;
- no sparsity refinement, so that the optimized score is exactly $D_k$.

The atlas is built once and serialized. Candidate projections are shared across
all prefix sizes: the region at rank $r$ is solved only once, and its result is
carried forward to every $k \geq r$.

## Commands

Run the stages separately:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/topk_heuristic_ablation.py prepare \
    --config configs/experiments/topk_heuristic_ablation.yaml

MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/topk_heuristic_ablation.py build \
    --config configs/experiments/topk_heuristic_ablation.yaml

MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/topk_heuristic_ablation.py benchmark \
    --config configs/experiments/topk_heuristic_ablation.yaml

MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/topk_heuristic_ablation.py aggregate \
    --config configs/experiments/topk_heuristic_ablation.yaml

MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/topk_heuristic_ablation.py analyze \
    --config configs/experiments/topk_heuristic_ablation.yaml
```

Or execute the complete pipeline:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python scripts/topk_heuristic_ablation.py all \
    --config configs/experiments/topk_heuristic_ablation.yaml
```

The benchmark saves a resumable partial artifact every 25 complete queries.
Rerunning the command without `--force` skips complete query blocks. Use
`status` to inspect progress.

## Interpretation

`strict_success_rate` measures whether the top-$k$ prefix returns a certified
target-class point. `exact_recovery_rate` instead measures whether its distance
matches the exact atlas optimum. A successful result may therefore still miss
the best certified region.

The absolute gap is $D_k-D^*$ and the relative excess is
$(D_k-D^*)/D^*$. The notebook also reports recovery within 1% and 5%, the
minimum observed recovery rank, query cost, and an exact one-sided binomial
upper confidence bound on the miss probability. These statements concern the
fixed certified atlas, not the full target-class preimage of the classifier.
