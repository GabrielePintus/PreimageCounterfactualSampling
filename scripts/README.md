# Scripts

Executable entrypoints for training classifiers and running counterfactual benchmarks.

Run scripts from the repository root so relative config, checkpoint, data, and result paths resolve correctly.

The network-complexity grid has a resumable staged runner:

```bash
python scripts/network_complexity_grid.py all \
  --config configs/experiments/network_complexity_grid.yaml
```

It also exposes `prepare`, `train`, `benchmark`, `analyze`, and `status`.
`--architectures depth_01_width_016 ...` selects endpoint pilots or targeted
recovery; resume requires matching configuration fingerprints and checkpoint
hashes.

The `all` stage finishes training and accuracy-gating all 25 classifiers before
starting any CertCF atlas construction or query benchmark. For separate jobs,
run `train`, then `benchmark`, then `analyze`.

To remeasure the complete online grid with concurrent top-$k$ projections, while
keeping numerical-library threading disabled inside each worker, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/network_complexity_grid.py benchmark \
  --config configs/experiments/network_complexity_grid.yaml \
  --candidate-parallelism 8 --candidate-parallel-backend process --force
python scripts/network_complexity_grid.py analyze \
  --config configs/experiments/network_complexity_grid.yaml
python scripts/analyze_network_complexity_parallel_rerun.py
```

The process pool is warmed before the measured query loop. The last command
performs the paired comparison against the archived serial run and writes both
per-query and per-architecture artifacts under `results/network_complexity/`.

The corresponding 1,000-query targeted MNIST/LeNet-5 rerun uses a dedicated
output file so that the original serial artifact is preserved:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/benchmark.py \
  --config configs/benchmarks/mnist_certcf_lenet5_parallel.yaml --force
python scripts/analyze_certcf_parallel_rerun.py \
  --old results/mnist_certcf_lenet5.parquet --old-run certcf_lenet5 \
  --new results/mnist_certcf_lenet5_parallel.parquet \
  --output-json results/mnist_certcf_lenet5_parallel_comparison.json \
  --output-summary results/mnist_certcf_lenet5_parallel_comparison.parquet
```

The paired VERIX/CertCF MNIST experiment has a separate staged runner:

```bash
python scripts/verix_certcf_mnist.py prepare \
  --config configs/experiments/verix_certcf_mnist.yaml
python scripts/verix_certcf_mnist.py pilot \
  --config configs/experiments/verix_certcf_mnist.yaml
python scripts/verix_certcf_mnist.py verix \
  --config configs/experiments/verix_certcf_mnist.yaml
python scripts/verix_certcf_mnist.py certcf \
  --config configs/experiments/verix_certcf_mnist.yaml
```

Each query artifact is written atomically. `status` reports resumable progress;
`--force` starts a new prepared run.

For the faster paired validation on the existing 32D synthetic dataset:

```bash
python scripts/verix_certcf_synthetic32.py prepare \
  --config configs/experiments/verix_certcf_synthetic32.yaml
python scripts/verix_certcf_synthetic32.py verix \
  --config configs/experiments/verix_certcf_synthetic32.yaml
python scripts/verix_certcf_synthetic32.py certcf \
  --config configs/experiments/verix_certcf_synthetic32.yaml
python scripts/verix_certcf_synthetic32.py analyze \
  --config configs/experiments/verix_certcf_synthetic32.yaml
```

The VERIX stage checkpoints every completed feature, so rerunning the same
command resumes interrupted queries.

For the complete 25-architecture scaling grid, use the same staged CLI with
`configs/experiments/verix_certcf_synthetic32_grid.yaml`. Run `prepare`, then
`verix`, then `certcf`, and finally `analyze`. The grid configuration bounds
each complete VERIX query traversal to 120 cumulative seconds; the theoretical
VERIX wall-time ceiling is therefore 8 h 20 min for 25 architectures and 10
queries, before small preparation and analysis overheads.

The CIFAR-10 ResNet scaling pipeline is split by network:

```bash
python scripts/cifar_resnet_scaling.py prepare --config configs/experiments/cifar_resnet_scaling.yaml
python scripts/cifar_resnet_scaling.py pilot --config configs/experiments/cifar_resnet_scaling.yaml
python scripts/cifar_resnet_scaling.py benchmark --network resnet20 --config configs/experiments/cifar_resnet_scaling.yaml
python scripts/cifar_resnet_scaling.py benchmark --network resnet32 --config configs/experiments/cifar_resnet_scaling.yaml
python scripts/cifar_resnet_scaling.py benchmark --network resnet56 --config configs/experiments/cifar_resnet_scaling.yaml
python scripts/cifar_resnet_scaling.py aggregate --config configs/experiments/cifar_resnet_scaling.yaml
python scripts/cifar_resnet_scaling.py analyze --config configs/experiments/cifar_resnet_scaling.yaml
```

`status` reports resumable progress. The official build requires CUDA and is
capped at 12 hours per network; each online query is capped at 120 seconds.
Both `pilot` and the per-network benchmarks use all 10,000 selected CIFAR-10
training images as atlas anchors; the pilot only reduces the query count and
runs ResNet20 alone.

To rerun only the ten saved pilot queries with concurrent projections while
reusing its serialized atlas, use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/cifar_resnet_scaling.py query \
  --config configs/experiments/cifar_resnet_scaling.yaml \
  --network resnet20 --pilot-artifacts --force \
  --candidate-parallelism 8 --candidate-parallel-backend process
python scripts/analyze_cifar_parallel_rerun.py \
  --serial-root results/cifar_resnet_scaling/archive/serial_before_parallel_2026-09-05/pilot \
  --parallel-root results/cifar_resnet_scaling/pilot \
  --networks resnet20 \
  --output-prefix results/cifar_resnet_scaling/pilot_parallel_rerun
```

The LiRPA-refinement ablation has a separate resumable CLI:

```bash
python scripts/lirpa_refinement_ablation.py prepare --config configs/experiments/lirpa_refinement_ablation.yaml
python scripts/lirpa_refinement_ablation.py build --config configs/experiments/lirpa_refinement_ablation.yaml
python scripts/lirpa_refinement_ablation.py benchmark --methods anchor_ball anchor_pgd certcf --config configs/experiments/lirpa_refinement_ablation.yaml
python scripts/lirpa_refinement_ablation.py aggregate --config configs/experiments/lirpa_refinement_ablation.yaml
python scripts/lirpa_refinement_ablation.py analyze --config configs/experiments/lirpa_refinement_ablation.yaml
```

To time only the CertCF branch with concurrent top-$k$ projections while reusing
the saved atlas, run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/lirpa_refinement_ablation.py benchmark \
  --config configs/experiments/lirpa_refinement_ablation.yaml \
  --cases heloc --methods certcf --force \
  --candidate-parallelism 8 --candidate-parallel-backend process
```

The persistent process pool is warmed before the measured query loop. The
parallelism flags affect execution only and do not rebuild the atlas or change
the prepared query/anchor geometry.

`pilot` runs 50 queries on HELOC, Adult, and the depth-1/depth-5 Synthetic32
endpoints. `status` reports per-case and per-method completion. Post-hoc L1
certification is part of `analyze`; `--skip-certification` is intended only for
smoke checks.

The top-$k$ heuristic runner also measures intra-query projection parallelism:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/topk_heuristic_ablation.py parallel-benchmark \
  --config configs/experiments/topk_heuristic_ablation.yaml \
  --parallel-k-values 1,2,3,4,5,6,7,8 \
  --candidate-workers 1,2,4,8 \
  --candidate-backend process
python scripts/topk_heuristic_ablation.py parallel-analyze \
  --config configs/experiments/topk_heuristic_ablation.yaml
```

The process pool is warmed before timing. Each parallel result is checked
against the matching serial distance and selected anchor. Use
`--parallel-queries N` for a resumable smaller pilot.

The generic benchmark CLI also accepts execution-only candidate-parallelism
overrides. For example, the query-objective norm ablation can be rerun without
overwriting its serial artifact:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/benchmark.py \
  --config configs/benchmarks/certcf_query_norm_ablation.yaml \
  --output results/certcf_query_norm_ablation_parallel.parquet \
  --candidate-parallelism 8 --candidate-parallel-backend process --force
```

The 250 CertCF timings needed by the auxiliary VeriX comparison are the exact
matching subset of the parallel network-complexity grid. After validating
success, prediction, and L1 distance, update only those timing cells with:

```bash
python scripts/update_verix_certcf_parallel_timings.py
```

## Main Entrypoints

| Script | Purpose | Typical command |
| --- | --- | --- |
| `train_classifier.py` | Train or test a Lightning classifier from `configs/training/`. | `python scripts/train_classifier.py fit --config configs/training/adult_classifier.yaml` |
| `benchmark.py` | Run counterfactual benchmarks from `configs/benchmarks/final_benchmark.yaml`. | `python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml` |

## Benchmark Examples

```bash
# Full final benchmark
python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml

# Subset by dataset and method
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --datasets compas heloc \
  --methods certcf face

# Recompute even if result parquet files already exist
python scripts/benchmark.py \
  --config configs/benchmarks/final_benchmark.yaml \
  --force
```

The benchmark writes a combined parquet plus per-dataset parquet files next to the configured output path.

## Training Examples

```bash
# Train a tabular classifier
python scripts/train_classifier.py fit --config configs/training/compas_classifier.yaml

# Short smoke run with fewer epochs
python scripts/train_classifier.py fit \
  --config configs/training/compas_classifier.yaml \
  --trainer.max_epochs=2

# Test a checkpoint
python scripts/train_classifier.py test \
  --config configs/training/compas_classifier.yaml \
  --ckpt_path checkpoints/compas_classifier/best.ckpt
```

## Notes

- These scripts add `src/` to `sys.path`, so editable installation is helpful but not required for simple local runs.
- Generated outputs should stay out of git unless they are small documentation artifacts.
- Auxiliary profiling, incremental databank, and Slurm wrappers are not part of the cleaned repository.
