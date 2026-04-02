# Configs

Lightning and experiment configuration files.

## Training (`configs/training/`)

Classifier and autoencoder training configs for `python train.py fit --config configs/training/<name>.yaml`.

- `mnist_classifier.yaml`
- `mnist_classifier_benchmark.yaml`
- `mnist_ae.yaml`
- `spiral_classifier.yaml`
- `adult_classifier.yaml`
- `compas_classifier.yaml`
- `german_credit_classifier.yaml`
- `heloc_classifier.yaml`
- `give_me_some_credit_classifier.yaml`
- `lending_club_classifier.yaml`

## Benchmarks (`configs/benchmarks/`)

Benchmark config details live in `configs/benchmarks/README.md`.

### Benchmarks (`scripts/benchmark.py`)

- `benchmark_adult.yaml`
- `benchmark_adult_main.yaml`
- `benchmark_adult_certcf.yaml`
- `benchmark_mnist.yaml`
- `benchmark_meeting_all.yaml`
- `benchmark_smoke_all.yaml`

### CertCF query benchmarks (`scripts/certcf_query_benchmark.py`)

- `certcf_query_benchmark_adult.yaml`
- `certcf_query_benchmark_adult_smoke.yaml`
