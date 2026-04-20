# Agent Operating Guidelines

## Goal

Help coding agents produce safe, minimal, and reproducible changes in this repository.

## Read Strategy

Use targeted reads in this order:
1. `AGENTS.md`
2. Relevant file in `ops/`
3. Specific source module(s)
4. Matching config and notebook only if needed

Avoid full-repo linear reads unless explicitly asked.

## Task Routing

- Training bugs or hyperparameter workflows:
  - start with `ops/01-training-playbook.md`
- Counterfactual algorithm or certification behavior:
  - start with `ops/02-certified-atlas-playbook.md`
- Dependency boundary or package ownership questions:
  - start with `ops/03-subprojects-and-dependencies.md`

## Safe Edit Zones

Preferred edit zones for core work:
- `src/certcf/**`
- `src/models/**`
- `src/training/**`
- `configs/*.yaml`

Use caution in:
- `notebooks/` (stateful and output-heavy)
- `legacy/` (older code path)

Default no-touch zones unless requested:
- `wandb/**`
- `checkpoints/**`
- `vicreg-loss/**`

## Validation Expectations

After non-trivial edits, do at least one of:
- run a reduced-epoch `train.py fit` command for impacted model path
- run a minimal `CertCFAtlas.build(...)` + `find_counterfactual(...)` check
- run subproject tests only when those subprojects are edited

## Common Failure Modes

- Breaking LightningCLI by changing class/module paths used in YAML `class_path`.
- Inconsistent feature ordering/cardinality for Adult tabular model.
- Changing solver defaults or tolerances without checking feasibility/success rate impact.
- Editing generated artifacts instead of source code.

## Recommended Documentation Hygiene

When adding new functionality:
- update the closest `ops/*.md` file with new entrypoints or commands
- keep instructions concise and executable
- include exact file paths and command examples

This keeps context quality high and reduces repeated exploration cost for future agents.
