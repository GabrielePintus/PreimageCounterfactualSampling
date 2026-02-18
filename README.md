# Preimage Counterfactual Sampling

**Certified Polyhedral Projection (CPP)** for generating valid, proximal, and robust counterfactual explanations.

---

## Overview

Given a classifier $f: \mathbb{R}^d \to \mathbb{R}^K$ and a query input $\mathbf{x}_0$ classified as class $l$, a **counterfactual explanation** is the closest point $\mathbf{x}'$ that the model classifies as a different target class $t$:

$$\mathbf{x}' = \arg\min_{\mathbf{z}} \|\mathbf{x}_0 - \mathbf{z}\|_2 \quad \text{s.t.} \quad f(\mathbf{z}) = t$$

Standard gradient-based approaches minimize a non-convex loss combining distance and classification confidence. This leads to local minima, boundary-hugging solutions with no validity guarantee, and fragility under small perturbations.

**CPP replaces gradient descent with projection onto certified convex regions**, converting a non-convex search into a series of tractable quadratic programs.

---

## The CPP Methodology

### Core Idea

Using **LiRPA** (Linear Relaxation based Perturbation Analysis), we can certify that a classifier predicts class $t$ everywhere inside a **convex polytope** around a training sample. The union of these polytopes across all training samples of class $t$ forms a **Certified Atlas** — a conservative but certified approximation of the class $t$ preimage.

Finding the counterfactual then reduces to:

$$\mathbf{x}' = \text{proj}_{\bigcup_i \mathcal{P}_i}(\mathbf{x}_0) = \arg\min_{\mathbf{z} \in \bigcup_i \mathcal{P}_i} \|\mathbf{x}_0 - \mathbf{z}\|_2$$

Each individual projection $\text{proj}_{\mathcal{P}_i}(\mathbf{x}_0)$ is a **convex QP with a global optimum**. There are no local minima, no trade-off hyperparameter $\lambda$, and every result is certified valid by construction.

---

### Phase I — Atlas Construction (Offline)

For each training sample $\mathbf{c}_i$ of target class $t$:

**1. Define the perturbation region.** Consider an $L_p$ ball of radius $\varepsilon$ centered at $\mathbf{c}_i$:
$$\mathcal{B}_p(\mathbf{c}_i, \varepsilon) = \{\mathbf{x} : \|\mathbf{x} - \mathbf{c}_i\|_p \leq \varepsilon\}$$

**2. Compute LiRPA bounds.** The classifier is wrapped with a one-vs-all constraint layer:
$$g_j(\mathbf{x}) = z_t(\mathbf{x}) - z_j(\mathbf{x}) \quad \forall j \neq t$$
where $z_k(\mathbf{x})$ is the logit for class $k$. The input is classified as $t$ iff $g_j(\mathbf{x}) \geq 0$ for all $j \neq t$.

LiRPA backward-mode bound propagation computes **linear lower bounds** on each $g_j$ valid throughout the ball:
$$\mathbf{a}_{ij}^\top \mathbf{x} + b_{ij} \leq g_j(\mathbf{x}) \quad \forall\, \mathbf{x} \in \mathcal{B}_p(\mathbf{c}_i, \varepsilon)$$

This gives us a matrix $A_i \in \mathbb{R}^{(K-1) \times d}$ and vector $\mathbf{b}_i \in \mathbb{R}^{K-1}$ (one row per competitor class) such that:
$$A_i \mathbf{x} + \mathbf{b}_i \geq \mathbf{0} \;\Longrightarrow\; g_j(\mathbf{x}) \geq 0 \;\forall j \;\Longrightarrow\; f(\mathbf{x}) = t$$

**3. Form the certified polytope.** Intersect the halfspace constraints with the perturbation ball:
$$\mathcal{P}_i = \bigl\{\mathbf{x} : A_i \mathbf{x} + \mathbf{b}_i \geq \mathbf{0}\bigr\} \cap \mathcal{B}_p(\mathbf{c}_i, \varepsilon)$$

Every point inside $\mathcal{P}_i$ is **certified** to be classified as class $t$ by $f$. The **Certified Atlas** for class $t$ is the collection $\{\mathcal{P}_i\}_{i=1}^{N_t}$.

**Solvers used during Phase I:**
- `auto_LiRPA` (CROWN/IBP) for bound propagation
- Supports $L_1$, $L_2$, and $L_\infty$ perturbation norms

---

### Phase II — Counterfactual Query (Online)

Given a query $\mathbf{x}_0$:

**1. Candidate pruning via BVH.** A **Bounding Volume Hierarchy** (BVH) binary tree is pre-built over the polytope centers. At query time, branch-and-bound search prunes subtrees whose bounding box is farther from $\mathbf{x}_0$ than the current best projection distance. This reduces $O(N)$ QP solves to $O(\log N)$ in practice.

**2. QP projection onto each candidate polytope.** For each candidate polytope $\mathcal{P}_i$:
$$\min_{\mathbf{z}}\; \|\mathbf{z} - \mathbf{x}_0\|_2^2 \quad \text{s.t.} \quad A_i \mathbf{z} + \mathbf{b}_i \geq \mathbf{0}, \quad \|\mathbf{z} - \mathbf{c}_i\|_p \leq \varepsilon$$

Solver dispatch depends on the norm:
- **$L_\infty$**: SLSQP (all constraints are linear, very fast)
- **$L_2$ / $L_1$**: CVXPY + CLARABEL (handles SOCP and $L_1$ ball constraints natively)

**3. Return the global minimum.** The counterfactual is the projection with smallest distance across all evaluated polytopes. BVH guarantees this is the global minimum over the full atlas.

---

### $\delta$-Robust Counterfactuals

A counterfactual at the polytope boundary is fragile: a perturbation of size $\epsilon$ can push it back to the original class. To guarantee robustness, we enforce that the **entire $L_q$ ball of radius $\delta$** around the counterfactual lies inside the certified polytope.

This is achieved by **eroding** the polytope inward before projection. Three constraints are modified:

**Halfspace erosion** (using the dual norm $q^*$ where $\frac{1}{q} + \frac{1}{q^*} = 1$):
$$\mathbf{a}_j^\top \mathbf{x} + b_j \geq \delta \cdot \|\mathbf{a}_j\|_{q^*}$$
This follows from Hölder's inequality: for any $\Delta\mathbf{x}$ with $\|\Delta\mathbf{x}\|_q \leq \delta$, we have $|\mathbf{a}_j^\top \Delta\mathbf{x}| \leq \|\mathbf{a}_j\|_{q^*} \cdot \delta$.

**Ball shrinkage** (to fit the robustness ball inside the certification ball):
$$\|\mathbf{x} - \mathbf{c}_i\|_p \leq \varepsilon - \delta \cdot \mathcal{C}(q \to p)$$
where $\mathcal{C}(q \to p) = d^{\max(0,\, 1/p - 1/q)}$ is the norm conversion factor from the norm inequality $\|\mathbf{x}\|_{p_2} \leq \|\mathbf{x}\|_{p_1}$ for $p_2 > p_1$.

**Box shrinkage**:
$$\|\mathbf{x} - \mathbf{c}_i\|_\infty \leq \varepsilon - \delta$$

The robustness norm $q$ can differ from the certification norm $p$ (**cross-norm robustness**). The conversion factor has a critical asymmetry:
- $q < p$ (e.g., $L_2$ robustness on $L_\infty$ atlas): $\mathcal{C} = 1$, **nearly free**
- $q > p$ (e.g., $L_\infty$ robustness on $L_2$ atlas): $\mathcal{C} = d^{1/p - 1/q} \gg 1$, **expensive in high $d$**

Setting $\delta = 0$ recovers the standard (non-robust) formulation.

---

### Latent-Space Extension (VAE)

For high-dimensional inputs like images, working directly in input space has two drawbacks: QP solves in 784D are slow, and LiRPA bounds tend to be loose (polytopes collapse to the unconstrained ball). The library supports an alternative: **build the atlas in the latent space of a VAE**.

**Architecture:**
1. A **Convolutional VAE** encodes images to a 32D Gaussian latent space ($\mu$, $\sigma$). The encoder's mean $\mu$ is used as a deterministic representation.
2. A **composite model** `decoder → classifier` maps flat latent vectors to class logits. This is the model that LiRPA certifies.
3. The atlas is built in 32D latent space using the composite model.
4. At query time, a query image is encoded to $\mu$, a counterfactual latent is found in 32D, then decoded back to image space.

**VAE training** uses the standard ELBO loss with consistent scaling:
$$\mathcal{L} = \underbrace{\frac{1}{N}\sum \|\mathbf{x} - \hat{\mathbf{x}}\|^2_{\text{sum}}}_{\text{reconstruction}} + \beta \cdot \underbrace{\left(-\frac{1}{2}\mathbb{E}\left[\sum_k (1 + \log\sigma_k^2 - \mu_k^2 - \sigma_k^2)\right]\right)}_{\text{KL divergence}}$$

The KL regularization encourages a structured latent space where class decision boundaries are closer to data points — directly targeting the polytope quality problem.

---

### Preimage Quality Diagnostics

A key challenge: if $\varepsilon$ is too small, the LiRPA constraints don't bind and polytopes collapse to the bare $\varepsilon$-ball (degenerate nearest-neighbor). If $\varepsilon$ is too large, polytopes become infeasible. The library includes metrics to diagnose this:

| Metric | Formula | Collapse signal |
|--------|---------|----------------|
| **Feasibility** | $\min_j (A_i \mathbf{c}_i + \mathbf{b}_i)_j \geq 0$ | Infeasible = no polytope |
| **Center margins** | $\min_j (A_i \mathbf{c}_i + \mathbf{b}_i)_j$ | Near 0 = boundary-touching |
| **Chebyshev radius ratio** | $r_\text{Cheb} / \varepsilon$ | $= 1$ means unconstrained |
| **Volume ratio** | $\text{vol}(\mathcal{P}_i) / \text{vol}(\text{box})$ via Monte Carlo | $= 1$ means unconstrained |
| **Value-add ratio** | $d(\mathbf{x}_0, \mathbf{x}') / d(\mathbf{x}_0, \text{NN}_t)$ | $\approx 1$ means CF = NN |

In 2D, exact polygon area via Shapely validates the Monte Carlo estimator, which then generalises to high-$d$.

---

## Library Structure

```
preimage_sampling/
├── atlas.py               # CertifiedAtlas: main API (build + query)
├── certification/
│   └── lirpa.py           # LiRPA bound propagation via auto_LiRPA
├── geometry/
│   ├── polytopes.py       # Ball/box constraints, Shapely polygon helpers
│   └── operations.py      # Polygon union, refinement
├── indexing/
│   └── bvh.py             # BVH spatial index with branch-and-bound
├── models/
│   ├── classifiers.py     # SimpleClassifier (spiral), MNISTClassifier (CNN)
│   └── ae_channels.py     # ConvAutoencoder (VAE): ConvEncoder + PixelShuffleDecoder
├── sampling/
│   └── sampler.py         # Low-level sampler (legacy API)
└── visualization/
    └── plotting.py        # Plotting utilities
```

---

## Quick Start

### Spiral toy dataset (2D)

```python
import torch
from preimage_sampling.models import SimpleClassifier
from preimage_sampling.atlas import CertifiedAtlas

# Load model and dataset
model = SimpleClassifier(num_classes=5)
dataset = torch.load('data/TOY Spiral/train_spiral.pt')

atlas = CertifiedAtlas(model, dataset, device='cuda')
atlas.build(eps=0.1, norm=2, build_unions=True)  # build_unions for 2D visualization

# Find counterfactual
x_query = dataset[0][0].numpy()
result = atlas.find_counterfactual(x_query, target_class=2, method='bvh')

if result.success:
    print(f"Counterfactual at distance {result.distance:.4f}")
    print(f"QP solves: {result.n_qp_solved}")
```

### MNIST in latent space (32D VAE)

```python
import torch
import torch.nn as nn
from preimage_sampling.models import MNISTClassifier
from preimage_sampling.models import ConvAutoencoderChannels as ConvAutoencoder
from preimage_sampling.atlas import CertifiedAtlas
from torch.utils.data import TensorDataset

# Load models
classifier = MNISTClassifier(num_classes=10).to(device)
vae = ConvAutoencoder(latent_dim=32).to(device)

# Composite model: decoder → classifier
class DecoderClassifier(nn.Module):
    def __init__(self, decoder, classifier):
        super().__init__()
        self.decoder = decoder
        self.classifier = classifier
    def forward(self, z):
        return self.classifier(self.decoder(z))

composite = DecoderClassifier(vae.decoder, classifier).to(device)

# Encode dataset to latent space (use mu for deterministic representation)
with torch.no_grad():
    mu, logvar = vae.encoder(images.to(device))
    latent_vectors = mu.cpu()

latent_dataset = TensorDataset(latent_vectors, labels)
atlas = CertifiedAtlas(composite, latent_dataset, device, cnn=False)
atlas.build(eps=0.1, norm=1)

# Query: encode → find counterfactual in latent space → decode
with torch.no_grad():
    mu_q, _ = vae.encoder(x_query.unsqueeze(0).to(device))
z_query = mu_q.cpu().numpy().flatten()

result = atlas.find_counterfactual(z_query, target_class=3)
x_cf_image = vae.decoder(torch.tensor(result.x_cf).unsqueeze(0).to(device))
```

### δ-Robust counterfactual

```python
# Standard (no robustness margin)
result = atlas.find_counterfactual(x_query, target_class=3)

# δ-robust: entire L2 ball of radius 0.05 around CF stays certified
result_robust = atlas.find_counterfactual(
    x_query, target_class=3,
    delta=0.05, robust_norm=2
)

# Cross-norm: atlas built with L1, robustness in L∞
result_cross = atlas.find_counterfactual(
    x_query, target_class=3,
    delta=0.01, robust_norm=float('inf')
)
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `1 - Training the NN.ipynb` | Train `SimpleClassifier` on 2D spiral data |
| `1.2 - Training the AE.ipynb` | Train convolutional VAE on MNIST with ELBO loss |
| `2 - Preimage approximation.ipynb` | Compute and visualize certified polytopes in 2D |
| `3.1 - MNIST-AE counterfactual sampling.ipynb` | Full pipeline in 32D VAE latent space |
| `3.2 - Spiral counterfactual sampling.ipynb` | Full pipeline on 2D spiral (ground-truth polytope visualization, value-add metrics, BVH benchmark) |

---

## Installation

```bash
git clone https://github.com/gabrielepintus/PreimageCounterfactualSampling.git
cd PreimageCounterfactualSampling
pip install -r requirements.txt
pip install -e .
```

**Dependencies:** `torch`, `auto_LiRPA`, `cvxpy`, `scipy`, `shapely`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`

**Recommended environment:** `py13` conda env (includes all dependencies + CUDA).

---

## Experimental Results

### 2D Spiral (`eps=0.1`, `L2`, 5 classes, 320 polytopes/class)

| Metric | Value |
|--------|-------|
| Feasibility | 71.3% (1141/1600) |
| Chebyshev r/eps (median) | 44.4% |
| Volume ratio (median) | 73.4% |
| CF closer than NN | 75% |
| NN distance ratio (median) | 0.85 |
| BVH speedup | 18.7× (17 vs 320 QPs) |

Polytopes are **genuinely non-trivial**: halfspace constraints bind, CFs are 15% closer than nearest-neighbor.

### MNIST latent space (`eps=0.1`, `L1`, 32D VAE)

| Metric | Value |
|--------|-------|
| Feasibility | 8.7% (87/1000) |
| Chebyshev r/eps (median) | 100% |
| CF success rate | 100% |
| CF validity | 100% (certified) |

The VAE's KL regularization brings decision boundaries closer (vs. plain AE: 95% feasibility at eps=0.1, 49D spatial), creating a harder feasibility tradeoff. **Key open problem**: finding the right (eps, norm) regime where polytopes are both feasible and non-trivially constrained in latent space.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SLSQP for $L_\infty$ | All constraints linear — fast and reliable |
| CVXPY/CLARABEL for $L_2$/$L_1$ | Native SOCP support, handles non-linear ball constraints exactly |
| BVH median split | Balances tree depth; $O(N \log N)$ construction, $O(\log N)$ query |
| VAE encoder mean $\mu$ as latent point | Deterministic, smooth manifold — avoids stochastic noise in atlas centers |
| Manual pixel shuffle (view+permute) | `nn.PixelShuffle` uses `onnx::DepthToSpace` which auto_LiRPA may not support |
| ELBO with sum/batch normalization | Prevents KL domination: reconstruction sums over pixels, KL sums over latent dims, both averaged over batch |

---

## Citation

```bibtex
@software{preimage_sampling2025,
  author = {Pintus, Gabriele},
  title  = {Preimage Counterfactual Sampling: Certified Polyhedral Projection for Counterfactual Explanations},
  year   = {2025},
  url    = {https://github.com/gabrielepintus/PreimageCounterfactualSampling}
}
```

---

## Acknowledgments

- [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) — neural network certification via linear relaxation
- [CVXPY](https://www.cvxpy.org/) / [CLARABEL](https://github.com/oxfordcontrol/Clarabel.jl) — convex optimization solver
- [Shapely](https://shapely.readthedocs.io/) — 2D polygon operations for ground-truth area computation
