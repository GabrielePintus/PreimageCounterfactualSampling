# Project History

This file is a lightweight operational history of the repository.
It is meant for humans and coding agents who need quick context on what
changed, what was intentionally abandoned, and which directions are current.

It is not a replacement for git history. The goal is to preserve decision
context, not line-by-line diffs.

## Current State

- The active research surface is centered on CertCF and the modular
  benchmarking framework in `src/counterfactuals/`.
- The active notebooks are:
  - `notebooks/6.12 - CertCF boundary-biased latent anchor diagnostics.ipynb`
  - `notebooks/6.13 - CertCF proximity distribution histograms.ipynb`
  - `notebooks/6.14 - CertCF random anchor k-sweep analysis.ipynb`
  - `notebooks/6.15 - Method comparison KDE analysis.ipynb`
- Historical notebooks were moved to `archive/notebooks/`.

## Key Repository Decisions

### Prediction-aligned benchmark support

- Benchmark-time method fitting is now prediction-aligned.
- `scripts/benchmark.py` computes model-predicted training labels once per run.
- Methods receive predicted support labels during `fit(...)`.
- Dataset ground-truth labels remain available for diagnostics and reporting.

### CertCF support semantics

- CertCF no longer performs benchmark-specific relabeling internally during
  `fit(...)`.
- The wrapper now treats the labels passed into `fit(...)` as authoritative.

### Benchmark result semantics

- Benchmark rows now distinguish:
  - `method_success`
  - `target_reached`
  - final benchmark `success`
- `success` is the strict conjunction:
  - method reported success
  - returned counterfactual reaches the requested target class

### Method provenance

- Result files now distinguish:
  - `method`: implementation key
  - `run_name`: configured experiment variant label
- Notebook utilities now preserve variant-level labels instead of collapsing
  multiple runs under the same implementation label.

### Boundary-random subsampling

- `boundary_random` exists only for CertCF.
- Generic methods reject it explicitly instead of silently degrading to random.

## Cleanup History

### Notebook cleanup

- Temporary and duplicate notebooks were removed.
- Older exploratory notebooks were moved out of the active surface into
  `archive/notebooks/`.

### Operational docs reorganization

- The old `agent_docs/` folder was replaced with `ops/`.
- `README.md` remains the public welcome page.
- `ops/` is now the single home for operational Markdown documents useful to
  both humans and coding agents.

### Removed side paths

- The old CertCF query-benchmark side suite was removed:
  - `scripts/certcf_query_benchmark.py`
  - `scripts/certcf_query_analyze.py`
  - related configs/docs
- This reduced semantic drift relative to the main benchmark path.

## Maintenance Rule

When a change affects repository structure, benchmark semantics, or major
research direction, add a short note here in the same change set.
