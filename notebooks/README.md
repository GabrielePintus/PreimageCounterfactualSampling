# Notebooks

Interactive notebooks for training evaluation and counterfactual generation demos.

## Naming Convention

- `1.x`: model evaluation notebooks (MNIST classifier/autoencoder).
- `2`: preimage approximation and sampling walkthrough.
- `3.x`: spiral/MNIST sampling experiments.
- `5.x`: Adult dataset counterfactual baselines and modular method comparisons.

## Current Adult Counterfactual Series

- `5 - Adult counterfactual sampling.ipynb`: original CPP-style workflow.
- `5.0 - Adult counterfactual sampling CertifiedAtlas.ipynb`: modular novel method notebook.
- `5.1 - Adult counterfactual sampling FACE.ipynb`: FACE baseline in modular API.
- `5.2 - Adult counterfactual sampling DiCE.ipynb`: DiCE baseline in modular API.
- `5.3 - Adult counterfactual sampling 1-NN.ipynb`: nearest-neighbor baseline in modular API.

## Usage Notes

- Run cells top-to-bottom in a fresh kernel.
- Some notebooks expect checkpoints under `../checkpoints/`.
- For k-medoids cells, install `scikit-learn-extra`.
