# Notebooks

Interactive notebooks for training evaluation and counterfactual generation demos.

## Naming Convention

- `1.x`: model evaluation notebooks (MNIST classifier/autoencoder).
- `2`: preimage approximation and sampling walkthrough.
- `3.x`: spiral/MNIST sampling experiments.
- `5.x`: Adult dataset counterfactual baselines and modular method comparisons.
- `6.x`: benchmark analysis notebooks.

## Current Adult Counterfactual Series

- `5 - Adult counterfactual sampling.ipynb`: original CertCF workflow.
- `5.0 - Adult counterfactual sampling CertCF.ipynb`: modular novel method notebook.
- `5.1 - Adult counterfactual sampling FACE.ipynb`: FACE baseline in modular API.
- `5.2 - Adult counterfactual sampling DiCE.ipynb`: DiCE baseline in modular API.
- `5.3 - Adult counterfactual sampling 1-NN.ipynb`: nearest-neighbor baseline in modular API.

## Usage Notes

- Run cells top-to-bottom in a fresh kernel.
- Some notebooks expect checkpoints under `../checkpoints/`.
- For k-medoids cells, install `scikit-learn-extra`.

## Benchmark Analysis

- `6.1 - Benchmark analysis.ipynb`: earlier single-file benchmark analysis notebook.
- `6.2 - Multi-dataset benchmark analysis.ipynb`: new multi-dataset analysis notebook for the meeting results.
- `6.3 - MNIST benchmark analysis.ipynb`: MNIST-specific benchmark analysis with quantitative results and example images.
