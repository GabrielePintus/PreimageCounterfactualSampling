# AGENTS.md

Agent instructions for the `PreimageCounterfactualSampling` workspace.

## Project Overview

This repository contains:
- Main research code in `src/` for CertCF and preimage-based counterfactual generation.
- A modular counterfactual benchmarking framework under `src/counterfactuals/`.

Primary human-facing docs are in `README.md` and notebooks under `notebooks/`.
Primary executable training entrypoint is `scripts/train_classifier.py`.
Counterfactual benchmark entrypoint is `scripts/benchmark.py`.

## Fast Navigation

Start with `README.md`, then inspect the relevant modules under `src/`, configs
under `configs/`, and notebooks under `notebooks/` for the task at hand.

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
python scripts/train_classifier.py fit --config configs/training/mnist_classifier.yaml
python scripts/train_classifier.py fit --config configs/training/mnist_ae.yaml
python scripts/train_classifier.py fit --config configs/training/spiral_classifier.yaml
python scripts/train_classifier.py fit --config configs/training/adult_classifier.yaml
```

## Testing And Validation

Validation is primarily workflow-based for core CertCF modules:
- Run focused scripts/notebooks for changed modules.
- For training changes, run at least one short `scripts/train_classifier.py fit`
  command with reduced epochs.
- For atlas/sampling changes, run a minimal `CertCFAtlas.build(...)` plus one `find_counterfactual(...)` call.

Automated tests currently available:
- `tests/counterfactuals/` provides `pytest` coverage for the modular benchmarking framework.
- Run with: `pytest tests/counterfactuals -q`

## Scope And Safety Rules

- Prefer editing `src/` and `configs/` for core CertCF work.
- Prefer `src/counterfactuals/` + `scripts/benchmark.py` for benchmarking changes.
- Do not commit large generated artifacts from `wandb/`, `checkpoints/`, or notebook outputs.
- Keep code LiRPA-compatible when changing classifier/model pieces used for certification.

## Code Style Notes

- The codebase uses PyTorch + Lightning with explicit docstrings and clear module boundaries.
- Keep public APIs stable where possible:
  - `certcf.CertCFAtlas`
  - `CertCFAtlas.build(...)`
  - `CertCFAtlas.find_counterfactual(...)`
- Keep the modular benchmarking contracts stable:
  - `counterfactuals.core.BaseCounterfactualMethod`
  - `fit(...)`, `generate(...)`, `generate_batch(...)`
- Preserve compatibility with YAML `class_path` references used by LightningCLI.

## Documentation Pattern

Keep repository-facing instructions concise and indexable for both humans and
LLM agents. Prefer updating `README.md`, config READMEs, and focused notebook
READMEs when repository layout changes.
