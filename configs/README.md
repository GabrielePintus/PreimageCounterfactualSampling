# Configs

Lightning and experiment configuration files.

## Training (`configs/training/`)

Classifier and autoencoder training configs for
`python scripts/train_classifier.py fit --config configs/training/<name>.yaml`.

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

### Benchmarks (`scripts/benchmark.py`)

- `meeting_all_200q.yaml`
- `full_all_200q.yaml`
- `smoke_all.yaml`
- `meeting_certcf_input_vs_latent_200q.yaml`
- `meeting_certcf_random_k_sweep_adult.yaml`
- `meeting_certcf_boundary_random_k_sweep_adult.yaml`
- `meeting_certcf_boundary_random_nearest_anchor_top3_200q.yaml`
- `meeting_certcf_boundary_random_top3_plus_nn_gs_200q.yaml`
- `mnist.yaml`
