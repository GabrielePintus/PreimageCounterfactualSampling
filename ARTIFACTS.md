# Reproducibility Artifacts

The repository intentionally excludes generated datasets, checkpoints, and
benchmark outputs. The anonymous artifact bundle supplied with the reviewer
submission restores those files at their expected repository-relative paths.

## Paper-result provenance

`configs/paper/paper_results.yaml` is the authoritative machine-readable
manifest. `python scripts/prepare_paper_results.py` verifies each recorded
checksum and selects the exact method/run pairs reported in the paper.

The core raw result files are:

| Path | Purpose | SHA-256 |
| --- | --- | --- |
| `results/final_benchmark_noface.parquet` | NN, Growing Spheres, and DiCE quality metrics | `9a82352cc6663ad24dc26d4e3b09f782d07d14073a2ca139b46f5351a596df88` |
| `results/final_benchmark_noface_certcf_shrink_sparsity.parquet` | CertCF hyperparameter sweep and selected quality results | `1b98a976fb9d6da5b2e1f1a2918383d7d1dd9096264c481d8e473c44a341b1cb` |
| `results/final_benchmark_face.parquet` | FACE quality metrics | `c0f3d0fa93cf8b55b61e402c3d90c585bec35678bcc208d7527a7540eaf10be5` |
| `results/dice_query_batch1_20queries.parquet` | DiCE batch-size-one timing run | `9bd5e7319003f1b57c75f4108f9ffc97674361d6fdd17c24a84bc3f87de3f0d1` |
| `results/final_benchmark_certcf_parallel.parquet` | Optimized CertCF top-k projection timing run | `a96793b9e6fe5fd89acf17f4b5be83d36e753d6a5d82290ad6e20ff912f0ac1c` |

The preparation command produces two derived, ignored files:

- `results/paper_results.parquet`: one row per paper method and query;
- `results/paper_query_timing.parquet`: raw timing rows, with the targeted
  DiCE and CertCF timing runs substituted explicitly.

These derived files can always be recreated and should not be committed.

## Expected query counts

The paper uses 4,773 queries per method: 1,000 each for Adult, Give Me Some
Credit, HELOC, and Lending Club; 617 for COMPAS; 100 for German Credit; and 56
for Wisconsin Breast Cancer. The preparation script rejects missing, duplicate,
or unexpected query rows.

## Data and checkpoints

Processed tabular datasets are restored under `data/`, and trained classifiers
under `checkpoints/<dataset>_classifier/best.ckpt`. The exact paths are listed
in the root README and referenced directly by the benchmark configurations.
