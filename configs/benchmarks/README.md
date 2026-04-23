# Benchmark Pipeline

This directory contains YAML configuration files for the counterfactual benchmark pipeline.
The official entrypoint is `scripts/benchmark.py`.
It accepts both single-dataset and multi-dataset config schemas.

---

## Quick Start

```bash
# Smoke benchmark
python scripts/benchmark.py --config configs/benchmarks/benchmark_smoke_all.yaml

# Main multi-dataset benchmark
python scripts/benchmark.py --config configs/benchmarks/benchmark_meeting_all_200q.yaml

# Subset of datasets
python scripts/benchmark.py \
    --config configs/benchmarks/benchmark_meeting_all_200q.yaml \
    --datasets adult compas

# Override output path
python scripts/benchmark.py \
    --config configs/benchmarks/benchmark_meeting_all_200q.yaml \
    --output results/my_run.parquet
```

---

## Output

The benchmark pipeline produces:
- **`.parquet`** — flat DataFrame loaded directly by the analysis notebooks (`notebooks/6.x`).

For a multi-dataset config, `benchmark.py` additionally writes one per-dataset parquet
(`<stem>_<dataset>.parquet`) alongside the combined output file.

### Incremental benchmark databank

For iterative method additions, you can store one parquet per method inside a benchmark family and rebuild a derived catalog:

```text
results/benchmarks/<family_name>/
  manifest.json
  methods/
    <method_or_run_name>.parquet
  combined.parquet
  summary.json
  index.html
```

Recommended workflow:

```bash
# Run one method only
python scripts/benchmark.py \
    --config configs/benchmarks/benchmark_meeting_all_200q.yaml \
    --methods certcf \
    --output results/benchmarks/adult_meeting_all_seed42_v1/methods/certcf.parquet

# Refresh the databank family
python scripts/benchmark_databank_refresh.py \
    --family results/benchmarks/adult_meeting_all_seed42_v1
```

If `manifest.json` does not exist yet, the refresh script can bootstrap it from an existing parquet:

```bash
python scripts/benchmark_databank_refresh.py \
    --family results/benchmarks/adult_meeting_all_seed42_v1 \
    --init-from-parquet results/benchmark_adult_meeting.parquet \
    --dataset adult \
    --config-path configs/benchmarks/benchmark_meeting_all_200q.yaml \
    --seed 42 \
    --task-definition meeting_all_200q
```

The manifest locks the task set (`query_indices`, `y_orig`, `target_class`) and the refresh step validates each method parquet against it before rebuilding `combined.parquet`.

---

## Single-Dataset Config Schema

Used by `benchmark.py`. All fields except `dataset`, `model`, and `methods` are optional.

```yaml
seed: 42
timeout_per_sample: 60        # seconds per (method, query); 0 = no limit

dataset:
  name: adult                 # registry key: adult | compas | german_credit |
                              #   heloc | give_me_some_credit | lending_club |
                              #   wisconsin_breast_cancer | mnist
  params:
    data_dir: data/

model:
  name: tabular_classifier_ckpt   # or: mnist_classifier_ckpt | sklearn_mlp | torch_model
  params:
    checkpoint: checkpoints/adult_classifier/best.ckpt
    device: cuda              # cpu | cuda | auto
    dataset_module: adult     # which datamodule to load constants from
    hidden_dims: [64, 32]
    dropout: 0.2

sampling:
  n_queries: 50               # how many test queries (shared by all methods)
  # MNIST only:
  # balanced_per_class: 1
  # target_policy: all_other_classes

output:
  path: results/benchmark_adult.parquet

methods:
  - name: nearest_neighbor    # registry key — selects the implementation
    run_name: nn              # label used in results; defaults to name
    params:
      norm: 1
      subsample_method: kmedoids
      k_per_class: 200

  - name: certcf
    run_name: certcf
    params:
      checkpoint: checkpoints/adult_classifier/best.ckpt
      device: cuda
      norm: 1
      eps_alpha: 0.25
      batch_size: 256
      atlas_subsample_method: kmedoids
      atlas_subsample_space: input
      k_per_class: 200
```

Method-specific runner knobs:
- DiCE accepts `query_batch_size` in `params`; the default is `1`, and values greater than `1` let the benchmark call `DiceMethod.generate_batch(...)` on query chunks.
- CertCF accepts `query_parallelism` in `params`; the default is `1`, and values greater than `1` enable threaded `generate_batch(...)` calls with soft per-query timeout handling.

Result semantics:
- `method` in the saved parquet is the implementation key, for example `certcf` or `dice`
- `run_name` is the configured variant label used in benchmark tables and plots
- benchmark `success` is strict: the method must report success and the returned point must reach the requested target class
- `method_success` and `target_reached` are also stored explicitly for failure analysis

### Grid expansion

Any parameter whose value is a **list** is treated as a sweep axis.
All list parameters are expanded into a Cartesian product; the `run_name` gets a
`_param=value` suffix for each varied parameter.

Exception:
- `certcf.params.cvxpy_solvers` is treated as an atomic solver chain, not a sweep axis.

```yaml
# This single entry expands to 8 runs:
- name: certcf
  run_name: certcf
  params:
    norm: [1, 2]
    eps_alpha: [0.15, 0.25, 0.35, 0.45]   # 2 × 4 = 8
    k_per_class: 200                        # fixed
```

---

## Multi-Dataset Config Schema

Used by `benchmark.py` when the config contains a top-level `datasets:` list.
Methods are defined once and applied to every dataset.

```yaml
seed: 42
output: results/benchmark_meeting_all.parquet
timeout_per_sample: 60

methods:                        # shared across all datasets
  - name: nearest_neighbor
    run_name: nn
    params: { norm: 1, subsample_method: kmedoids, k_per_class: 200 }

  - name: certcf
    run_name: certcf
    params:
      # 'checkpoint' is auto-inherited from each dataset's model.params.checkpoint
      device: cuda
      norm: 1
      eps_alpha: 0.25
      atlas_subsample_method: kmedoids
      atlas_subsample_space: input

datasets:
  - name: adult
    sampling:
      n_queries: 50
    model:
      name: tabular_classifier_ckpt
      params:
        checkpoint: checkpoints/adult_classifier/best.ckpt
        device: cuda
        dataset_module: adult
        hidden_dims: [64, 32]
        dropout: 0.2

  - name: compas
    sampling:
      n_queries: 50
    model:
      name: tabular_classifier_ckpt
      params:
        checkpoint: checkpoints/compas_classifier/best.ckpt
        device: cuda
        dataset_module: compas
        hidden_dims: [64, 32]
        dropout: 0.2
    method_overrides:           # optional: override individual params for this dataset
      certcf:
        eps_alpha: 0.35
```

### Auto-inherit for `certcf`

If `certcf` appears in the shared methods list **without** a `checkpoint` key,
`benchmark.py` automatically copies `model.params.checkpoint` (and `device`) into it.
This avoids repeating the checkpoint path in both `model` and `method_overrides`.

### CertCF atlas subsampling

CertCF supports selecting atlas anchors in either raw input space or penultimate latent space:

```yaml
- name: certcf
  run_name: certcf_latent_kmedoids
  params:
    checkpoint: checkpoints/adult_classifier/best.ckpt
    device: cuda
    norm: 1
    eps_alpha: 0.45
    k_per_class: 200
    atlas_subsample_method: kmedoids
    atlas_subsample_space: latent
```

`atlas_subsample_space: latent` is currently supported for tabular torch classifiers used by the CertCF benchmark path. The atlas itself is still built and queried in input space; only anchor selection changes.

`atlas_subsample_method: boundary_random` is supported only by `certcf`. Generic benchmark methods do not implement boundary-aware subsampling and now fail explicitly if configured with that value.

During benchmark runs, method fitting is now prediction-aligned for **all** methods:

- the benchmark computes model-predicted training labels once per run
- generic methods receive those predicted labels in `fit(x_train, y_train_pred)`
- `CertCF.fit(...)` also receives predicted support labels and treats them as authoritative
- dataset ground-truth training labels are kept only for diagnostics and analysis
- `atlas_subsample_space` still controls whether CertCF chooses anchors in raw input space or penultimate latent space

### `method_overrides`

A dict keyed by `run_name` (or `name`). Values are **shallow-merged** on top of the shared
method params — individual keys are overwritten, everything else is preserved.

### `dataset_params`

If a dataset needs non-default `dataset.params` (e.g. `heloc` has no `data_dir`), set it
directly on the dataset block:

```yaml
- name: heloc
  dataset_params: {}            # overrides the default {data_dir: data/}
  ...
```

---

## Available Config Files

Active benchmark configs live in `configs/benchmarks/`.
Historical or superseded configs have been moved to `archive/configs/benchmarks/`.

### Multi-dataset (run with `benchmark.py`)

| File | Datasets | Methods | Queries |
|------|----------|---------|---------|
| `benchmark_meeting_all_200q.yaml` | adult, compas, german_credit, heloc, give_me_some_credit, lending_club | nn, dice, gs, face, certcf | 200 |
| `benchmark_full_all_200q.yaml` | adult, compas, german_credit, heloc, give_me_some_credit, lending_club, wisconsin_breast_cancer | nn, wachter, dice, gs, face, certcf | 200 |
| `benchmark_smoke_all.yaml` | compas, german_credit, heloc, give_me_some_credit, lending_club | nn, gs, certcf | 50 |
| `benchmark_meeting_certcf_input_vs_latent_200q.yaml` | adult, compas, german_credit | certcf input kmedoids, certcf latent kmedoids | 200 |
| `benchmark_meeting_certcf_random_k_sweep_adult.yaml` | adult, compas, german_credit | certcf random anchors, k sweep | 200 |
| `benchmark_meeting_certcf_boundary_random_k_sweep_adult.yaml` | adult, compas, german_credit | certcf boundary-random anchors, k sweep | 200 |
| `benchmark_meeting_certcf_boundary_random_nearest_anchor_top3_200q.yaml` | adult, compas, german_credit | certcf boundary-random + nearest-anchor (`k=[3,5]`) | 200 |
| `benchmark_meeting_certcf_boundary_random_top3_plus_nn_gs_200q.yaml` | adult, compas, german_credit | certcf boundary-random top-3, nn, dice, gs | 200 |

### Single-dataset (run with `benchmark.py`)

| File | Dataset | Purpose |
|------|---------|---------|
| `benchmark_mnist.yaml` | mnist | Full benchmark on MNIST (balanced per-class sampling) |

---

## Available Methods

| Registry key | Description |
|---|---|
| `certcf` | CertCF (our method) |
| `nearest_neighbor` | Closest opposite-class training point |
| `face` | FACE: density-weighted shortest path |
| `dice` | DiCE: gradient-based diverse counterfactuals |
| `growing_spheres` | Growing Spheres: shell sampling |
| `wachter` | Wachter: L-BFGS-B with validity loss |
