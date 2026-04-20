# Subprojects And Dependencies

## Repository Composition

This workspace includes two Python packages with different maintenance intent.

## Main Package: `certcf`

Location:
- `src/certcf/`
- `src/models/`
- `src/training/`

Packaging:
- root `setup.py` with `package_dir={"": "src"}`
- install via `pip install -e .`

Used for:
- training classifiers/autoencoders
- LiRPA-based certification
- counterfactual generation

## Embedded Library: `vicreg-loss/`

`vicreg-loss/` is an external package implementing VICReg and VICRegL losses.

Signals:
- own `pyproject.toml`, package source, and README
- intended as reusable standalone loss implementation

Guidance:
- use as dependency/reference for representation-learning experiments.
- treat as separate ownership boundary unless explicitly asked to modify it.

## Practical Dependency Notes

Main package dependencies (from root `setup.py`) include:
- `torch`, `torchvision`, `numpy`, `scipy`
- `auto-LiRPA`, `cvxpy`, `shapely`
- `matplotlib`, `scikit-learn`, `lightning`, `tqdm`

Optional dev extras include `pytest`, `black`, `flake8`, `mypy`, notebook tools.

## Artifact-Heavy Directories

Avoid scanning or editing unless required:
- `wandb/`
- `checkpoints/`
- generated notebook outputs
- `*.egg-info/`, `build/`, and `__pycache__/`

These can add noise and token overhead for agents.

## Online Best-Practice Alignment

Doc structure in this repository follows widely used agent-context patterns:
- `AGENTS.md`: a predictable root context file for coding agents
- segmented markdown docs in `ops/`: small, task-focused context chunks
- concise navigation instead of forcing agents to parse the entire repository

This aligns with public guidance from AGENTS.md conventions and LLM-focused context map approaches (for example, `llms.txt` style curated linking).
