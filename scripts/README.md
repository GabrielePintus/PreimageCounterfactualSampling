# Scripts

Executable entrypoints for training classifiers and running counterfactual benchmarks.

Run scripts from the repository root so relative config, checkpoint, data, and result paths resolve correctly.

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
