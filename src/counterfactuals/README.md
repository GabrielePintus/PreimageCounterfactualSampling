# Counterfactuals Framework

This package provides a modular, extensible framework for running counterfactual experiments with a single interface.

## Design Goals

- Pluggable methods, models, datasets, and metrics
- Reproducible experiments via global seeding
- Config-driven execution (`python run_experiment.py <config.yaml>`)
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
2. Register it in a method `Registry` in your experiment setup.
3. Add method-specific tests under `tests/counterfactuals/`.
