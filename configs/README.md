# Configs

Lightning and experiment configuration files.

## Training Configs

- `mnist_classifier.yaml`
- `mnist_ae.yaml`
- `spiral_classifier.yaml`
- `adult_classifier.yaml`

Use with:

```bash
python train.py fit --config configs/<name>.yaml
```

## Counterfactual Benchmark Config

- `counterfactual_experiment.yaml`

Use with:

```bash
python run_experiment.py --config configs/counterfactual_experiment.yaml
```
