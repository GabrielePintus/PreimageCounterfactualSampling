Here is a comprehensive synthesis of all evaluation aspects and metrics across the 7 papers.

## 1. Core Properties / Desiderata

These are the high-level qualities that CFs are expected to satisfy. They appear across all papers.

| Property | Description |
|----------|------------|
| Validity | CF achieves the desired target class prediction |
| Proximity | CF is close to the original instance |
| Sparsity | CF changes as few features as possible |
| Plausibility | CF lies on or near the real data manifold |
| Actionability | Changes are feasible for the user to act upon |
| Feasibility | CF is reachable from the original instance via a plausible path |
| Diversity | Multiple CFs are distinct from each other |
| Causal Consistency | CF respects known causal relationships between features |

---

## 2. Metrics — Full Reference

### ✅ Validity / Success Rate

Definition: percentage of instances for which a valid CF (achieving target label) is found.

Sources:  
- NICE: An Algorithm for Nearest Instance Counterfactual Explanations  
- CARLA: A Python Library to Benchmark Algorithmic Recourse and Counterfactual Explanation Algorithms  
- Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations  
- Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers  

DiCE condition:

\[
f(c) > 0.75
\]

for the desired class.

---

### 📏 Proximity / Distance

Different papers use different distance functions.

| Metric | Formula | Source |
|--------|---------|---------|
| L2 (Euclidean) | \(\|x - \tilde{x}\|_2\) | Counterfactual Explanations without Opening the Black Box |
| L1 (Manhattan) | \(\|x - \tilde{x}\|_1\) | Same + causal constraints paper |
| Normalized L1 | \(c_1 = \frac{1}{d}\|\delta_x\|_1\) | CARLA |
| MAD-normalized L1 | \(\text{dist}_{cont}(c,x) = \frac{1}{d_{cont}} \sum_p \frac{|c_p - x_p|}{MAD_p}\) | DiCE + causal constraints |
| Categorical mismatch | \(\text{dist}_{cat}(c,x) = \frac{1}{d_{cat}} \sum_p \mathbf{1}[c_p \neq x_p]\) | DiCE + causal constraints |
| HEOM | heterogeneous L1 distance | NICE / tabular metrics |

HEOM per feature distance:

\[
d_F(a,b) =
\begin{cases}
1 & \text{if categorical and } a \neq b \\
0 & \text{if categorical and } a = b \\
\frac{|a-b|}{range} & \text{if continuous}
\end{cases}
\]

Practical note:  
MAD-normalized L1 is usually the most robust choice for tabular data.

---

### 🔢 Sparsity (L0)

Definition: number of features changed.

\[
c_0 = \frac{1}{d} \|x - \tilde{x}\|_0
\]

Sources:

- NICE  
- CARLA  
- Wachter et al.

---

### 🌐 Plausibility / Data Manifold Proximity

Three main approaches.

#### a) Autoencoder reconstruction error

Train AE on training data.

\[
\|cf - AE(cf)\|^2
\]

Low error = on manifold.

Used by NICE.

---

#### b) IM1 / IM2 (prototype paper)

Train two autoencoders:

- \(AE_i\) on target class  
- \(AE_{t_0}\) on original class  

IM1:

\[
IM1 = \frac{\|x_{cf} - AE_i(x_{cf})\|_2^2}
{\|x_{cf} - AE_{t_0}(x_{cf})\|_2^2 + \varepsilon}
\]

Lower = better match to target manifold.

Source: Interpretable Counterfactual Explanations Guided by Prototypes.

IM2 = single autoencoder variant.

---

#### c) yNN (CARLA)

\[
yNN =
1 -
\frac{1}{n_k n}
\sum_{i \in \breve{H}^{-}}
\sum_{j \in kNN(\breve{x}_i)}
|f^b(\breve{x}_i) - f^b(x_j)|
\]

Measures whether CF lies in region of same-label points.

Range: [0,1], higher = more plausible.

---

#### d) kNN distance

Average distance to nearest neighbors.

Example: 5NN distance.

Used by NICE.

---

### 🎲 Diversity

Average pairwise distance:

\[
\Delta =
\frac{1}{\binom{k}{2}}
\sum_{i=1}^{k-1}
\sum_{j=i+1}^{k}
d(c_i, c_j)
\]

Distance can be continuous or categorical metric.

DiCE uses DPP internally to encourage diversity.

Source: Diverse Counterfactual Explanations.

---

### ⚙️ Actionability / Constraint Satisfaction

#### Immutable feature violations

Example: age, race cannot change.

Metric: violation rate.

Used by CARLA, FACE.

---

#### Causal constraints

From causal constraints paper.

Metrics:

- Constraint feasibility score
- Causal-edge score
- Causal-graph score

Example:

log-likelihood under causal model.

---

#### FACE feasibility

CF must lie on path of high-density points in graph.

Graph-based reachability constraint.

Source: FACE.

---

### ⏱️ Runtime

Average time per CF.

Used by NICE, CARLA.

---

### 📦 Coverage

Percentage of instances with at least one CF.

Used by NICE, CARLA.

---

### 🔄 Redundancy

How many changes are unnecessary.

Idea:

Revert features and check validity.

Used by CARLA.

---

### 🛡️ Cross-model robustness

Generate CF for model A  
Test on model B.

Measures transferability.

Used by NICE.

---

## 3. Quick Implementation Checklist

| Metric | Implementation |
|--------|--------------|
| Validity | mean(f(cf) == target) |
| L0 Sparsity | mean((cf != x).sum() / d) |
| L1 proximity | MAD-normalized L1 + categorical mismatch |
| AE error | train AE, compute reconstruction |
| IM1 | two AEs, ratio |
| yNN | kNN label agreement |
| Diversity | pairwise distance |
| Constraint violation | check immutable / causal rules |
| Coverage | % with ≥1 valid CF |
| Runtime | wall-clock time |
