# Configs

Lightning and experiment configuration files.

## Training (`configs/training/`)

Classifier and autoencoder training configs for `python train.py fit --config configs/training/<name>.yaml`.

- `mnist_classifier.yaml`
- `mnist_ae.yaml`
- `spiral_classifier.yaml`
- `adult_classifier.yaml`
- `compas_classifier.yaml`
- `german_credit_classifier.yaml`
- `heloc_classifier.yaml`
- `give_me_some_credit_classifier.yaml`
- `lending_club_classifier.yaml`

## Benchmarks (`configs/benchmarks/`)

Counterfactual benchmark configs for `python scripts/benchmark.py --config configs/benchmarks/<name>.yaml`.

- `benchmark_adult.yaml`
- `benchmark_adult_main.yaml`
- `benchmark_adult_cpp.yaml`
- `benchmark_compas_smoke.yaml`
- `benchmark_german_credit_smoke.yaml`
- `benchmark_heloc_smoke.yaml`
- `benchmark_give_me_some_credit_smoke.yaml`
- `benchmark_lending_club_smoke.yaml`
- `cpp_query_benchmark_adult.yaml`
- `cpp_query_benchmark_adult_smoke.yaml`
- `counterfactual_experiment.yaml`
