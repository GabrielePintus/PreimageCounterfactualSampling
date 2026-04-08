# Benchmark Pipeline

This directory contains YAML configuration files for the counterfactual benchmark pipeline.
The official entrypoint is `scripts/benchmark.py`.
It accepts both single-dataset and multi-dataset config schemas.

---

## Quick Start

```bash
# Single dataset
python scripts/benchmark.py --config configs/benchmarks/benchmark_adult.yaml

# All tabular datasets (meeting benchmark)
python scripts/benchmark.py --config configs/benchmarks/benchmark_meeting_all.yaml

# Subset of datasets
python scripts/benchmark.py \
    --config configs/benchmarks/benchmark_meeting_all.yaml \
    --datasets adult compas

# Override output path
python scripts/benchmark.py \
    --config configs/benchmarks/benchmark_meeting_all.yaml \
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
                              #   heloc | give_me_some_credit | lending_club | mnist
  params:
    data_dir: data/

model:
  name: tabular_classifier_ckpt   # or: mnist_classifier_ckpt | sklearn_mlp | torch_model
  params:
    checkpoint: checkpoints/adult_classifier/best.ckpt
    device: cuda              # cpu | cuda | auto
    dataset_module: adult     # which datamodule to load constants from
    hidden_dims: [32, 8]
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
      subsample_method: kmedoids
      k_per_class: 200
```

### Grid expansion

Any parameter whose value is a **list** is treated as a sweep axis.
All list parameters are expanded into a Cartesian product; the `run_name` gets a
`_param=value` suffix for each varied parameter.

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
        hidden_dims: [32, 8]
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
        dropout: 0.1
    method_overrides:           # optional: override individual params for this dataset
      certcf:
        eps_alpha: 0.35
```

### Auto-inherit for `certcf`

If `certcf` appears in the shared methods list **without** a `checkpoint` key,
`benchmark.py` automatically copies `model.params.checkpoint` (and `device`) into it.
This avoids repeating the checkpoint path in both `model` and `method_overrides`.

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

### Multi-dataset (run with `benchmark.py`)

| File | Datasets | Methods | Queries |
|------|----------|---------|---------|
| `benchmark_meeting_all.yaml` | adult, compas, german_credit, heloc, give_me_some_credit, lending_club | nn, dice, gs, face, certcf | 50 |
| `benchmark_meeting_all_200q.yaml` | adult, compas, german_credit, heloc, give_me_some_credit, lending_club | nn, dice, gs, face, certcf | 200 |
| `benchmark_smoke_all.yaml` | compas, german_credit, heloc, give_me_some_credit, lending_club | nn, gs, certcf | 50 |

### Single-dataset (run with `benchmark.py`)

| File | Dataset | Purpose |
|------|---------|---------|
| `benchmark_adult.yaml` | adult | CertCF grid search (eps_alpha × k_per_class), 200 queries |
| `benchmark_adult_main.yaml` | adult | All-methods comparison with grid expansion, 20 queries |
| `benchmark_adult_certcf.yaml` | adult | CertCF-only grid search, 20 queries |
| `benchmark_mnist.yaml` | mnist | Full benchmark on MNIST (balanced per-class sampling) |

### Other schemas

| File | Script | Purpose |
|------|--------|---------|
| `certcf_query_benchmark_adult.yaml` | `scripts/certcf_query_benchmark.py` | BVH vs sorted query method comparison |
| `certcf_query_benchmark_adult_smoke.yaml` | `scripts/certcf_query_benchmark.py` | Smoke version of the above |

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
