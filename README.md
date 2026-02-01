# Preimage Counterfactual Sampling

A library for generating robust counterfactual explanations using **Certified Polyhedral Projection (CPP)**.

## Overview

This library implements a novel approach to counterfactual explanation generation that transforms the CE problem from a highly non-convex, unstable optimization task into a series of convex projections.

### Key Advantages

1. **Guaranteed Validity**: Counterfactuals are mathematically certified to lie deep within the target class distribution, not just barely across the decision boundary.

2. **Manifold Adherence**: Counterfactuals are constrained to stay close to the true data manifold, avoiding out-of-distribution artifacts.

3. **Convexity & Speed**: The linearized network enables solving convex QP problems with global optimality guarantees, eliminating local minima issues.

## The CPP Method

### Core Intuition

Standard counterfactual methods solve:
```
x' = argmin_z (||x - z||_p + λ · L_target(f(z)))
```

This involves navigating a non-convex loss landscape via gradient descent, often getting stuck in local minima.

**CPP Instead**: Since we decompose the target class geometry into a union of convex polytopes `P = {P_1, ..., P_N}` using LiRPA certification, finding the counterfactual simplifies to:
```
x' = proj_{∪ P_i}(x) = argmin_{z ∈ ∪ P_i} ||x - z||_2
```

### Two-Phase Algorithm

#### Phase I: Atlas Construction (Offline)

Using LiRPA bounds, construct certified polytopes around training samples:
1. For each sample x_i in target class
2. Define ε-ball B(x_i, ε)
3. Use auto_LiRPA to derive linear bounds A, b
4. Store polytope P_i = {z | Az + b ≥ 0} ∩ B(x_i, ε)

#### Phase II: Counterfactual Query (Online)

Given a query sample x_query:
1. **Candidate Filtering**: Find k-nearest neighbors in target class
2. **Convex Projection**: For each polytope, solve the QP:
   ```
   minimize    (1/2)||z - x_query||²
   subject to  Az ≥ -b              (LiRPA constraints)
               x_m - ε ≤ z ≤ x_m + ε (validity domain)
   ```
3. **Aggregation**: Return the projection with minimum distance

## Installation

```bash
# Clone the repository
git clone https://github.com/gabrielepintus/PreimageCounterfactualSampling.git
cd PreimageCounterfactualSampling

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import torch
from preimage_sampling import SimpleClassifier, PreimageApproximation, CounterfactualSampler
from preimage_sampling.utils import make_spiral

# 1. Create dataset and train model
X, y = make_spiral(n_samples_per_class=200, n_classes=10, noise=0.1)
dataset = torch.utils.data.TensorDataset(
    torch.tensor(X, dtype=torch.float32),
    torch.tensor(y, dtype=torch.long)
)

# 2. Compute LiRPA bounds (offline phase)
model = SimpleClassifier()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

approximator = PreimageApproximation(model, dataset, device)
all_bounds = approximator.compute_all_bounds(eps=0.05, norm=2)

# 3. Build polytope atlases
sampler = CounterfactualSampler(verbose=True)
sampler.build_all_atlases(all_bounds, eps=0.05)

# 4. Generate counterfactual (online phase)
x_query = X[0]  # Sample from class 0
result = sampler.generate_counterfactual(
    x_query,
    target_label=5,  # Want to move to class 5
    k_candidates=10
)

if result['success']:
    print(f"Found counterfactual at distance {result['distance']:.4f}")
    print(f"Counterfactual point: {result['counterfactual']}")

    # Verify with model
    verification = sampler.verify_counterfactual(
        result['counterfactual'],
        target_label=5,
        model=model,
        device=device
    )
    print(f"Model predicts class {verification['predicted_label']}")
```

## Library Structure

```
preimage_sampling/
├── models/          # Neural network architectures
├── certification/   # LiRPA-based certification
├── geometry/        # Polytope operations
├── sampling/        # CPP counterfactual generation
├── visualization/   # Plotting utilities
└── utils/           # Data generation and helpers
```

## Notebooks

- `1 - Training the NN.ipynb`: Train a classifier on spiral data
- `2 - Preimage approximation.ipynb`: Compute and visualize certified polytopes
- `3 - Counterfactual generation.ipynb`: Generate counterfactuals using CPP

## Citation

If you use this library in your research, please cite:

```bibtex
@software{preimage_sampling2024,
  author = {Pintus, Gabriele},
  title = {Preimage Sampling: Certified Polyhedral Projection for Counterfactuals},
  year = {2024},
  url = {https://github.com/gabrielepintus/PreimageCounterfactualSampling}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work builds on:
- [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) for neural network certification
- Research on robust counterfactual explanations and certified defenses
