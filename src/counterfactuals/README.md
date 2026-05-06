# Counterfactuals Framework

This package provides a modular framework for fitting, running, and evaluating counterfactual methods with a single interface.

## Design Goals

- Pluggable methods, models, datasets, and metrics
- Reproducible experiments via global seeding
- Config-driven execution via the benchmark pipeline (`python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml`)
- Easy extension through registries without modifying runner logic

## Core Contracts

- `BaseCounterfactualMethod.fit(x_train, y_train, model)`
- `BaseCounterfactualMethod.generate(example, model)`
- `BaseCounterfactualMethod.generate_batch(x, model, target_class)`
- `ModelInterface.predict(x)`
- `ModelInterface.predict_proba(x)`
- `DatasetInterface.load()`, `get_train()`, `get_test()`

## Extending

1. Implement a new method class in `counterfactuals/methods/`.
2. Register it in `counterfactuals/benchmarks/registry.py` or in a local experiment registry.
3. Add method-specific tests under `tests/counterfactuals/`.

For tabular datasets, schema-level metadata now lives in `dataset_specs/`.
Lightning datamodules still own data loading and fitted preprocessing, while the
benchmark stack consumes shared specs instead of importing datamodule constants.
