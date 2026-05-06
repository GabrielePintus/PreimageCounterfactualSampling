# Benchmark Configuration

This directory intentionally contains a single benchmark YAML:

- `final_benchmark.yaml` — the paper benchmark over all seven tabular datasets, with CertCF, Nearest Neighbor, Growing Spheres, DiCE, and FACE.

Run it with:

```bash
python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml
```

Useful overrides:

```bash
# Run only selected datasets
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --datasets compas heloc

# Run only selected methods
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --methods certcf face

# Write to a different parquet path
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --output results/my_benchmark.parquet

# Resume an interrupted run if the output parquet already exists
python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml

# Force recomputation from scratch
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --force
```

For multi-dataset runs, `scripts/benchmark.py` writes one combined parquet and one per-dataset parquet next to it. The configured output path is:

```text
results/final_benchmark.parquet
```

## Final Benchmark Contents

The config uses the same query set size and support cap for every dataset:

```yaml
sampling:
  n_queries: 1000
  n_train_per_class: 10000
```

The evaluated methods are:

| Method | Run name | Main settings |
| --- | --- | --- |
| CertCF | `certcf_alpha020_lambda1` | L1 distance, CROWN/backward LiRPA, `eps_alpha: 0.20`, adaptive shrinkage, group-aware reweighted-L1 sparsity with `sparsity_lambda: 1.0`, nearest-anchor query with 5 candidate anchors |
| Nearest Neighbor | `nn` | L1 distance |
| Growing Spheres | `growing_spheres` | L1 distance, `max_radius: 50.0`, `radius_step: 0.25`, `n_in_layer: 1000` |
| DiCE | `dice` | single CF, no diversity term, proximity weight 1, batch size 1 |
| FACE | `face` | L2 graph distance, kNN density estimator, `tp: 0.5`, dataset-specific graph radius |

FACE graph radii:

| Dataset | FACE `epsilon` |
| --- | ---: |
| `adult` | 2.9 |
| `compas` | 1.5 |
| `german_credit` | 3.6 |
| `give_me_some_credit` | 1.1 |
| `heloc` | 3.3 |
| `lending_club` | 2.6 |
| `wisconsin_breast_cancer` | 4.5 |

## Config Schema Notes

`final_benchmark.yaml` uses the multi-dataset schema:

```yaml
seed: 42
output: results/final_benchmark.parquet
timeout_per_sample: 120

sampling:
  n_queries: 1000
  n_train_per_class: 10000

methods:
  - name: certcf
    run_name: certcf_alpha020_lambda1
    params: {...}

datasets:
  - name: compas
    model:
      name: tabular_classifier_ckpt
      params:
        checkpoint: checkpoints/compas_classifier/best.ckpt
        device: auto
        dataset_module: compas
        hidden_dims: [64, 32]
        dropout: 0.2
    method_overrides:
      face:
        epsilon: 1.5
```

`method_overrides` can be keyed by method name or run name. The final config uses overrides only for FACE radii.

CertCF automatically inherits each dataset checkpoint and device from `model.params` when they are not specified directly in the method params.

Any method parameter whose value is a list is expanded as a grid sweep, except for atomic list parameters such as `certcf.params.cvxpy_solvers`. The final benchmark avoids sweeps so that each method appears once.

## Result Semantics

The saved parquet contains one row per `(dataset, method, query)` result. Important columns include:

- `method`: implementation key such as `certcf`, `face`, or `dice`
- `run_name`: configured variant label
- `success`: strict benchmark success, requiring both method success and target-class prediction
- `method_success`: whether the method reported a candidate
- `target_reached`: whether the model predicts the requested target class
- `distance`, `l1`, `l2`, and sparsity fields used by the analysis notebooks
- `meta__*` columns for method metadata, including resource-monitor fields when enabled
