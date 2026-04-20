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
- `wisconsin_breast_cancer_classifier.yaml`

## Benchmarks (`configs/benchmarks/`)

Benchmark config details live in `configs/benchmarks/README.md`.
Historical / superseded benchmark configs live in `archive/configs/benchmarks/`.

### Benchmarks (`scripts/benchmark.py`)

- `benchmark_meeting_all_200q.yaml`
- `benchmark_full_all_200q.yaml`
- `benchmark_smoke_all.yaml`
- `benchmark_meeting_certcf_input_vs_latent_200q.yaml`
- `benchmark_meeting_certcf_random_k_sweep_adult.yaml`
- `benchmark_meeting_certcf_boundary_random_k_sweep_adult.yaml`
- `benchmark_meeting_certcf_boundary_random_nearest_anchor_top3_200q.yaml`
- `benchmark_meeting_certcf_boundary_random_top3_plus_nn_gs_200q.yaml`
- `benchmark_mnist.yaml`
