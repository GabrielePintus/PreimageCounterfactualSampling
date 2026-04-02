# Preimage Counterfactual Sampling

This repository develops and benchmarks **CertCF**, our novel counterfactual method.
`certcf` is the benchmark-facing name used in configs and result tables; its underlying algorithm is based on **Certified Polyhedral Projection (CPP)**.

---

## Overview

The repository has two main roles:

- implement the CertCF method and its lower-level certification/query engine in `src/preimage_sampling/`
- benchmark CertCF against baselines such as DiCE, FACE, nearest-neighbor, and Growing Spheres via `scripts/benchmark.py`

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
- **$L_2$**: CVXPY with solver fallback chain CLARABEL → OSQP → SCS
- **$L_1$**: CVXPY with solver fallback chain OSQP → CLARABEL → SCS

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

### Adaptive ε via Clearance Strategy

A global fixed ε is a poor fit for heterogeneous datasets: points near the class boundary need a small ball (for feasibility); isolated interior points can tolerate a much larger one (richer polytope). The library provides pluggable **ε strategies** decoupled from atlas construction:

```
eps = strategy.compute_eps(X, y)   # shape (N,) — one value per training point
```

| Strategy | Formula | Use case |
|---|---|---|
| `ConstantEpsStrategy(eps)` | $\varepsilon_i = \varepsilon$ | Baseline, backward-compatible |
| `NearestOppositeClassClearanceStrategy(alpha)` | $\varepsilon_i = \alpha \cdot \min_{j:\, y_j \neq y_i} \|\mathbf{x}_i - \mathbf{x}_j\|_\infty$ | Adaptive; recommended for tabular data |

**NearestOppositeClassClearanceStrategy** sets each point's ε to a fraction $\alpha \in (0, 0.5)$ of its L∞ distance to the nearest opposite-class training point. At $\alpha < 0.5$ the certification ball never crosses a class boundary, guaranteeing feasibility. The default $\alpha = 0.25$ leaves a comfortable 50% safety margin. The O(N²) pairwise Chebyshev distance computation runs once offline via `scipy.spatial.distance.cdist`.

---

### Atlas Construction via k-Medoids

For large training sets, building one polytope per training point is expensive (each requires a LiRPA backward pass). The atlas is therefore built on a **representative subsample** rather than the full training set.

For each class $t$, the $k$ medoids of the class's training points are selected using **k-medoids clustering** (sklearn-extra). Medoids are actual data points (unlike k-means centroids), so every atlas center is a genuine training example. This preserves the semantic interpretation of each polytope center.

```python
# benchmark.py / atlas construction
from sklearn_extra.cluster import KMedoids
km = KMedoids(n_clusters=k_per_class, metric='euclidean')
km.fit(X_class)
X_atlas = X_class[km.medoid_indices_]   # k real training points per class
```

The atlas size is controlled by `k_per_class` (set to `None` to use the full shared train pool). A fallback to random subsampling is provided if sklearn-extra is not installed.

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

### Tabular Data Extension

The library supports tabular datasets with mixed numerical and categorical features. Categorical features are encoded as **one-hot vectors** (OHE), giving a well-defined continuous input space suitable for LiRPA certification and QP projection.

#### TabularClassifier Architecture

`TabularClassifier` (`src/models/classifiers.py`) uses a simple feedforward MLP operating directly on OHE-encoded inputs:

```
raw x  ──►  [OHE encoding by datamodule]  ──►  Dropout  ──►  Linear  ──►  ReLU  ──►  Dropout  ──►  Linear  ──►  ReLU  ──►  Linear  ──►  logits
```

Numerical features are **StandardScaler-normalised** by the datamodule (fit on the training split) before being concatenated with the OHE categorical features. There is no BatchNorm layer in the model — standardisation happens offline in the data pipeline, so the 104D input vector is already in a clean, geometrically meaningful space.

LiRPA certifies the full network (Dropout-stripped in eval mode) directly on the 104D OHE input.

#### Adult Dataset Features

The UCI Adult dataset has 14 features — 6 numerical and 8 categorical. After OHE encoding, the total input dimension is **104D** (6 numerical + 98 OHE categorical dims):

| Feature | Type | Encoding |
|---|---|---|
| age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week | Numerical | StandardScaler (fit on train split) |
| workclass (9), education (16), marital-status (7), occupation (15), relationship (6), race (5), sex (2), native-country (42) | Categorical | One-hot encoding (cardinality dims per feature) |

#### OHE Simplex Constraints

Each categorical block in the OHE vector must satisfy $\sum_{k \in \text{block}} x_k = 1$ and $x_k \geq 0$. These are enforced as **equality constraints in the QP** via `atlas.ohe_slices` — a list of `(start, end)` index pairs for each categorical block. After projection, `model.decode()` argmax-snaps each block to the nearest valid one-hot vertex.

#### The Atlas → Decode Pipeline

At query time:

1. **Find CF in raw OHE space**: `atlas.find_counterfactual(x_query, target_class=t)` — projects `x_query` onto the nearest opposite-class polytope, with OHE simplex constraints enforced in the QP.
2. **Snap to valid OHE**: `x_cf = model.decode(x_cf_raw)` — argmax-snaps each categorical block to a one-hot vertex, producing a valid discrete feature vector.

Distances (L1, L2, L0) are measured in raw feature space after snapping, on the same footing as all other methods.

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
src/
├── preimage_sampling/
│   ├── atlas.py               # CertifiedAtlas: main CPP API (build + query)
│   ├── certification/         # LiRPA orchestration and wrapped models
│   ├── geometry/              # Polytope helpers and 2D unions
│   ├── indexing/              # BVH spatial index
│   ├── sampling/              # Low-level legacy sampler
│   └── visualization/         # Plotting utilities
├── counterfactuals/           # Modular benchmarking framework
├── models/                    # NN architectures (spiral, MNIST, tabular, AE/VAE)
└── training/                  # Lightning modules and datamodules
```

---

## Quick Start

### Spiral toy dataset (2D)

```python
import torch
from models import SimpleClassifier
from preimage_sampling import CertifiedAtlas, ConstantEpsStrategy

# Load model and dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleClassifier(num_classes=5)
dataset = torch.load('data/TOY Spiral/train_spiral.pt')

atlas = CertifiedAtlas(
    model,
    dataset,
    device=device,
    norm=2,
    eps_strategy=ConstantEpsStrategy(0.1),
)
atlas.build(build_unions=True)  # build_unions for 2D visualization

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
from models import MNISTClassifier, ConvAutoencoderChannels as ConvAutoencoder
from preimage_sampling import CertifiedAtlas, ConstantEpsStrategy
from torch.utils.data import TensorDataset

# Assume images, labels, and x_query have already been loaded as tensors.
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    mu, _ = vae.encoder(images.to(device))
    latent_vectors = mu.cpu()

latent_dataset = TensorDataset(latent_vectors, labels)
atlas = CertifiedAtlas(
    composite,
    latent_dataset,
    device=device,
    cnn=False,
    norm=1,
    eps_strategy=ConstantEpsStrategy(0.1),
)
atlas.build()

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

## Benchmark Pipeline

Counterfactual methods can be evaluated against each other using the benchmark pipeline.
A single config file specifies the dataset(s), model checkpoint, methods, and hyperparameter grids.
The official benchmark entrypoint is `scripts/benchmark.py`.

```bash
# Single dataset
python scripts/benchmark.py --config configs/benchmarks/benchmark_adult.yaml

# Multiple datasets → one combined parquet
python scripts/benchmark.py --config configs/benchmarks/benchmark_meeting_all.yaml
```

Results are written as `.parquet` files and loaded directly by the `notebooks/6.x` analysis notebooks.
In benchmark configs and output tables, the novel method appears as `certcf`.

See **[configs/benchmarks/README.md](configs/benchmarks/README.md)** for the full pipeline documentation: config schema, grid expansion, multi-dataset format, and the list of available config files.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `1.2 - MNIST Classifier Evaluation.ipynb` | Evaluate the MNIST classifier training path |
| `1.3 - MNIST Autoencoder Evaluation.ipynb` | Evaluate the convolutional autoencoder / VAE path |
| `2 - Preimage approximation + CF sampling.ipynb` | Visualize certified regions and counterfactual sampling |
| `3.1 - Spiral counterfactual sampling.ipynb` | Spiral CertCF/CPP workflow in 2D |
| `3.3 - MNIST counterfactual sampling.ipynb` | Pixel-space MNIST counterfactual sampling |
| `3.4 - MNIST-AE counterfactual sampling.ipynb` | Latent-space MNIST counterfactual sampling |
| `5.0 - Adult counterfactual sampling CertCF.ipynb` | Adult tabular CertCF workflow |
| `6.1 - Benchmark analysis.ipynb` | Single-dataset benchmark analysis |
| `6.2 - Multi-dataset benchmark analysis.ipynb` | Combined tabular benchmark analysis |
| `6.3 - MNIST benchmark analysis.ipynb` | MNIST benchmark analysis |

---

## Installation

```bash
git clone https://github.com/gabrielepintus/PreimageCounterfactualSampling.git
cd PreimageCounterfactualSampling
pip install -e .
```

For notebook and test tooling:

```bash
pip install -e .[dev]
```

**Dependencies:** declared in `setup.py` and include `torch`, `auto_LiRPA`, `cvxpy`, `scipy`, `shapely`, `scikit-learn`, `numpy`, `matplotlib`, `pandas`, `pyarrow`, and `lightning`

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

### Adult tabular dataset (OHE input space, `L1`, adaptive ε, n=500 queries, 5 000 train medoids)

Benchmark comparing CertCF against DiCE, FACE, nearest_neighbor, and growing_spheres on the UCI Adult dataset. All methods operate and are evaluated in the same raw OHE feature space.

| Method | Validity | L1 (mean) | L2 (mean) | Sparsity (mean) | Runtime (mean) |
|---|---|---|---|---|---|
| **CertCF** | **98.8%** | 7.34 | 2.65 | **8.5%** | 0.86 s |
| nearest_neighbor | 100% | 6.91 | 2.57 | — | <1 ms |
| FACE | 100% | 7.44 | 2.71 | — | — |
| DiCE | 100% | 8.23 | 1.69 | 60% | — |
| growing_spheres | 82.6% | 6.13 | 0.99 | 100% | — |

Key findings:
- **Validity**: CertCF achieves 98.8% with certified guarantees. growing_spheres fails on 17.4% of queries.
- **Sparsity**: CertCF changes only 8.5% of features on average — the sparsest of all methods — consistent with its QP minimising the L1 norm with OHE simplex constraints.
- **Proximity**: CertCF's L1/L2 distances are competitive with retrieval-based methods (nearest_neighbor, FACE) and better than DiCE.
- **Runtime**: 0.86 s/query for CertCF reflects online BVH traversal + a small number of CVXPY solves. The atlas is built once offline.

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
