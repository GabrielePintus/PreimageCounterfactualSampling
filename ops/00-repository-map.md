# Repository Map

## Purpose

`PreimageCounterfactualSampling` implements CertCF:
- Build certified class-region approximations (polytopes) via LiRPA.
- Search those certified regions to generate valid counterfactuals.
- Use BVH indexing to reduce query-time projection cost.

## Top-Level Ownership

- `train.py`: LightningCLI entrypoint for training/testing.
- `configs/`: training recipes for MNIST/Spiral/Adult models.
- `src/models/`: classifier and autoencoder architectures.
- `src/training/`: Lightning modules and datamodules.
- `src/certcf/`: certification, atlas construction, geometry, indexing, sampling.
- `notebooks/`: experiment workflows and analysis.
- `checkpoints/`, `wandb/`: generated artifacts.
- `vicreg-loss/`: separate VICReg loss package.

## Core Runtime Paths

### Training path

1. `train.py` imports model/datamodule modules so Lightning can resolve YAML `class_path`.
2. Config in `configs/*.yaml` selects:
   - `training.lit_classifier.LitClassifier` or `training.lit_autoencoder.LitAutoencoder`
   - model class in `models.*`
   - datamodule in `training.datamodules.*`
3. Trainer callbacks save checkpoints under `checkpoints/*`.

### Counterfactual path

1. Instantiate `certcf.CertCFAtlas(model, dataset, device, ...)`.
2. Call `build(...)` to compute LiRPA bounds and BVH indices.
3. Call `find_counterfactual(...)` for online query-time projection.

## Main Public APIs

- `src/certcf/atlas.py`:
  - `CertCFAtlas.build(...)`
  - `CertCFAtlas.find_counterfactual(...)`
  - `CertCFAtlas.find_counterfactual_batch(...)`
  - `CertCFAtlas.verify_counterfactual(...)`
- `src/certcf/eps_strategies.py`:
  - `ConstantEpsStrategy`
  - `NearestOppositeClassClearanceStrategy`

## Data + Modeling Modes

- Spiral (2D tabular-like synthetic): fast visual sanity checks and geometry diagnostics.
- MNIST pixel-space classifier: CNN certification path.
- MNIST latent-space with VAE decoder+classifier composite: lower-dimensional certification workflow.
- Adult dataset tabular classifier: mixed numerical/categorical features.

## Agent Priorities

When asked to work quickly and safely, prioritize:
1. `src/certcf/*` for CertCF behavior.
2. `src/models/*` and `src/training/*` for train/inference regressions.
3. `configs/*.yaml` for reproducible experiments.
4. `notebooks/` only when workflow parity matters.
