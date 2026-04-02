# AGENTS.md

Agent instructions for the `PreimageCounterfactualSampling` workspace.

## Project Overview

This repository contains:
- Main research code in `src/` for Certified Polyhedral Projection (CPP) and preimage-based counterfactual generation.
- A modular counterfactual benchmarking framework under `src/counterfactuals/`.

Primary human-facing docs are in `README.md` and notebooks under `notebooks/`.
Primary executable training entrypoint is `train.py`.
Counterfactual benchmark entrypoint is `scripts/benchmark.py`.

## Fast Navigation

Read these in order when working on the main project:
1. `agent_docs/00-repository-map.md`
2. `agent_docs/01-training-playbook.md`
3. `agent_docs/02-certified-atlas-playbook.md`
4. `agent_docs/04-agent-operating-guidelines.md`

Read this for dependencies and subproject notes:
- `agent_docs/03-subprojects-and-dependencies.md`

## Setup Commands

Typical setup for local development:

```bash
pip install -e .
```

If you need optional dev tooling for the main package:

```bash
pip install -e .[dev]
```

Training is driven by Lightning CLI configs:

```bash
python train.py fit --config configs/mnist_classifier.yaml
python train.py fit --config configs/mnist_ae.yaml
python train.py fit --config configs/spiral_classifier.yaml
python train.py fit --config configs/adult_classifier.yaml
```

## Testing And Validation

Validation is primarily workflow-based for core CPP modules:
- Run focused scripts/notebooks for changed modules.
- For training changes, run at least one short `train.py fit` command with reduced epochs.
- For atlas/sampling changes, run a minimal `CertifiedAtlas.build(...)` plus one `find_counterfactual(...)` call.

Automated tests currently available:
- `tests/counterfactuals/` provides `pytest` coverage for the modular benchmarking framework.
- Run with: `pytest tests/counterfactuals -q`

## Scope And Safety Rules

- Prefer editing `src/` and `configs/` for core CPP work.
- Prefer `src/counterfactuals/` + `scripts/benchmark.py` for benchmarking changes.
- Do not commit large generated artifacts from `wandb/`, `checkpoints/`, or notebook outputs.
- Keep code LiRPA-compatible when changing classifier/model pieces used for certification.

## Code Style Notes

- The codebase uses PyTorch + Lightning with explicit docstrings and clear module boundaries.
- Keep public APIs stable where possible:
  - `preimage_sampling.CertifiedAtlas`
  - `CertifiedAtlas.build(...)`
  - `CertifiedAtlas.find_counterfactual(...)`
- Keep the modular benchmarking contracts stable:
  - `counterfactuals.core.BaseCounterfactualMethod`
  - `fit(...)`, `generate(...)`, `generate_batch(...)`
- Preserve compatibility with YAML `class_path` references used by LightningCLI.

## Documentation Pattern

This repo now includes agent-focused docs in `agent_docs/`.
The structure is intentionally concise and indexable, following cross-agent conventions similar to AGENTS.md and LLM-oriented context-map practices.
