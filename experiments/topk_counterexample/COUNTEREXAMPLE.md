# A controlled counterexample for CertCF's nearest-anchor top-$k$ heuristic

## 1. Purpose and status of the example

This document describes a controlled counterexample in which CertCF's
nearest-anchor top-5 heuristic almost always returns a valid but suboptimal
counterfactual relative to search over the complete certified atlas.

The example is **end-to-end with respect to CertCF**: anchor sampling,
Equation (1), auto-LiRPA, adaptive radius reduction, certified-polytope
construction, convex projection, top-5 retrieval, and exhaustive search all use
the production implementation in this repository. No epsilon radius, LiRPA
halfspace, certified polytope, or projection distance is inserted manually.

The data distribution and classifier are deliberately constructed. In
particular, the classifier is a frozen ReLU network with analytically specified
weights, not a network obtained through ordinary training. The example is
therefore a constructive counterexample to a universal optimality claim for
nearest-anchor top-$k$. It is **not** evidence that this failure occurs with the
same frequency for conventionally trained models on natural datasets.

## 2. Assumptions and simplifications

The construction makes the following assumptions explicit.

| Item | Assumption in this experiment |
|---|---|
| Task | Binary classification in two continuous dimensions |
| Model | Frozen, hand-constructed affine/ReLU network |
| Model training | None |
| Source support | 120 samples near the origin |
| Target support | 120 samples, balanced across six target components |
| Queries | 100 independent source samples per seed |
| Preprocessing | None |
| Atlas sampling | 24 randomly selected anchors per supplied class |
| Certification norm | $L_1$ |
| Projection norm | $L_1$ |
| Equation (1) scale | $\alpha=0.2$ |
| LiRPA method | Backward LiRPA |
| Adaptive radius reduction | Factor $1/2$, at most eight reductions |
| Query heuristic | Five nearest anchor centers |
| Reference search | Production `sorted` search over the certified atlas |
| Classification margin | Zero |
| Feasibility/actionability constraints | None |
| Categorical features and decoding | None |
| Sparsity refinement | None |
| Robustness erosion | None, i.e. $\delta=0$ |

These choices isolate the relationship between anchor distance and certified
projection distance. In particular, no query-specific constraint can make an
anchor center infeasible, and no categorical or sparsity post-processing can
change which certified region is preferable.

## 3. Data distribution

Let the input space be $\mathbb{R}^2$. Class 0 is the source class and class 1
is the counterfactual target class.

### 3.1 Source support and queries

The source support contains $n_0=120$ independent samples

$$
X_i^{(0)} \sim \mathcal{N}(0,\sigma_s^2 I_2),
\qquad \sigma_s=0.02.
$$

The queries are sampled independently from the same distribution:

$$
X_q \sim \mathcal{N}(0,\sigma_s^2 I_2).
$$

Consequently, the counterexample does not rely on a distribution shift between
the source support and the queries.

### 3.2 Target components

The target class consists of six disconnected $L_1$ balls. Five are called
**decoy components**. Their centers are

$$
\begin{aligned}
c_1 &= (-9,0), &
c_2 &= (0,9), &
c_3 &= (0,-9),\\
c_4 &= (-4.5,4.5), &
c_5 &= (-4.5,-4.5),
\end{aligned}
$$

and all have radius

$$
\rho_1=\cdots=\rho_5=0.03.
$$

The sixth, **useful component**, is centered at

$$
c_6=(9.1,0)
$$

and has radius

$$
\rho_6=5.
$$

The complete target region is therefore

$$
\mathcal{T}
=
\bigcup_{j=1}^{6} B_1(c_j,\rho_j),
\qquad
B_1(c,\rho)=\{x:\lVert x-c\rVert_1<\rho\}.
$$

Twenty target support points are generated around each center. First,

$$
Z_{j,r}\sim\mathcal{N}(c_j,\sigma_t^2I_2),
\qquad \sigma_t=0.005.
$$

Writing $u=Z_{j,r}-c_j$, the implemented sampler clips the offset to half the
component radius:

$$
X_{j,r}^{(1)}
=
c_j
+
u\min\left\{1,\frac{\rho_j/2}{\lVert u\rVert_1}\right\}.
$$

Thus every target support point lies strictly inside its assigned target
component. There are $6\times20=120$ target points, so the complete support set
is balanced between the two classes.

## 4. Exact ReLU classifier

For every component define its signed target margin

$$
m_j(x)=\rho_j-\lVert x-c_j\rVert_1.
$$

The network computes

$$
M(x)=\max_{1\leq j\leq6}m_j(x)
$$

and returns the two logits

$$
f(x)=\begin{bmatrix}0 & M(x)\end{bmatrix}.
$$

Therefore, away from ties,

$$
\operatorname{argmax} f(x)=1
\quad\Longleftrightarrow\quad
M(x)>0
\quad\Longleftrightarrow\quad
x\in\mathcal{T}.
$$

At $M(x)=0$, PyTorch's first-index tie breaking assigns class 0. The plotted
diamonds are the model decision boundary $M(x)=0$ and enclose the strict
target-class interiors.

The function is implemented only with affine layers and ReLUs. Coordinatewise
absolute values use

$$
|v|=\operatorname{ReLU}(v)+\operatorname{ReLU}(-v),
$$

and the maximum is accumulated using

$$
\max(u,v)=\operatorname{ReLU}(u-v)+v.
$$

The resulting computation has 226 frozen scalar parameters and 29 ReLU
nonlinearities. Its weights are sparse and analytically assigned. All 240
support samples receive their intended labels, and every evaluated query is
predicted as class 0 before counterfactual generation.

## 5. Atlas construction through the production pipeline

### 5.1 Random anchors

CertCF receives the balanced support set and randomly selects 24 anchors per
class using the experiment seed. The supplied labels and model predictions
coincide exactly. The analysis below concerns the class-1 atlas.

### 5.2 Equation (1)

For every selected target anchor $a_i$, the production epsilon strategy uses
the complete support set and computes

$$
d_i
=
\min_{z:\,y(z)\neq1}\lVert a_i-z\rVert_1,
\qquad
\epsilon_i^{(0)}=\alpha d_i,
\qquad
\alpha=0.2.
$$

The opposite-class support is concentrated near the origin. Since both the
decoy and useful centers have $L_1$ distance close to 9 from the origin,
Equation (1) assigns similar initial radii to both types of anchor. Across the
ten runs, the means are

$$
\overline{\epsilon}_{\mathrm{decoy}}^{(0)}=1.791,
\qquad
\overline{\epsilon}_{\mathrm{useful}}^{(0)}=1.812.
$$

These values are data-dependent outputs of Equation (1), not parameters of the
constructed model.

### 5.3 LiRPA and adaptive reduction

For an anchor $a_i$, backward auto-LiRPA computes affine lower bounds for the
target-vs-source logit margin within

$$
B_1(a_i,\epsilon_i).
$$

The corresponding certified region has the form

$$
P_i
=
B_1(a_i,\epsilon_i)
\cap
\{x:A_ix+b_i\geq0\},
$$

where $A_i$ and $b_i$ are generated by auto-LiRPA. If the anchor center cannot
be certified, the implementation applies

$$
\epsilon_i\leftarrow\frac{1}{2}\epsilon_i
$$

and recomputes the bounds, for at most eight reductions.

For every decoy anchor in the ten-seed experiment, seven reductions are
required:

$$
\epsilon_{mathrm{decoy}}^{(\mathrm{final})}
=
\frac{\epsilon_{mathrm{decoy}}^{(0)}}{2^7}
\approx0.014.
$$

The useful anchors certify without reduction:

$$
\epsilon_{mathrm{useful}}^{(\mathrm{final})}
=
\epsilon_{mathrm{useful}}^{(0)}
\approx1.812.
$$

This asymmetry is not manually inserted into the atlas. It follows from the
constructed model boundary and the actual certification procedure: the decoy
target components are extremely small, whereas the useful component contains
the entire initial certification ball.

## 6. Online search and failure event

Let the target atlas contain $m$ certified regions $P_i$ with anchors $a_i$.
For a query $x_q$, order the anchors by $L_1$ distance:

$$
\lVert x_q-a_{(1)}\rVert_1
\leq
\cdots
\leq
\lVert x_q-a_{(m)}\rVert_1.
$$

Define the projection distance onto region $i$ as

$$
D_i(x_q)=\min_{x\in P_i}\lVert x-x_q\rVert_1.
$$

The nearest-anchor top-5 result is

$$
D_5(x_q)=\min_{1\leq i\leq5}D_{(i)}(x_q),
$$

whereas the certified-atlas optimum is

$$
D^\star(x_q)=\min_{1\leq i\leq m}D_i(x_q).
$$

We record a heuristic failure when

$$
D_5(x_q)>D^\star(x_q)+10^{-5}.
$$

The decoy anchors are normally closer to the query because their centers have
distance approximately 9, compared with 9.1 for the useful component. Their
certified regions, however, extend only about 0.014 away from their centers.
The useful anchors are slightly farther in center distance but their certified
regions extend approximately 1.81 toward the query. Anchor ordering and
projection ordering are consequently reversed.

The top-5 result remains target-valid. Therefore, CertCF's fallback is not
activated: fallback handles the absence of a feasible candidate among the
first five regions, not the possibility that an unvisited region improves an
existing candidate.

## 7. Visual explanation

![Geometry and projection-distance explanation](outputs/counterexample_explained.png)

Panel A shows the actual model decision geometry. The large blue diamond is
the useful target component. The five decoy components have true radius 0.03
and are nearly point-sized at the global scale, so the inset displays one of
their boundaries without enlarging it in the main axes. The orange path is the
top-5 counterfactual; the blue path is the exhaustive-atlas counterfactual.

Panel B expresses the same run as distance from the query. Pale bars show the
initial balls from Equation (1), while colored bars show the query-facing reach
after certification. Although the useful anchor center is farther away, its
certified region yields the closer projection.

## 8. Concrete trace for seed 0

For query 0,

$$
x_q=(0.010942,\;0.035275).
$$

The top-5 procedure selects a decoy-2 region whose anchor is

$$
a_{\mathrm{top}}=(0.001934,\;8.986934).
$$

Its nearest observed opposite-class point is

$$
z_{\mathrm{top}}=(0.002715,\;0.046207),
$$

giving

$$
\lVert a_{\mathrm{top}}-z_{\mathrm{top}}\rVert_1=8.941507,
\qquad
\epsilon_{\mathrm{top}}^{(0)}=1.788301,
\qquad
\epsilon_{\mathrm{top}}^{(\mathrm{final})}=0.013971.
$$

The returned top-5 counterfactual is

$$
x_{\mathrm{top}}^{\mathrm{CF}}
=(0.010942,\;8.986066),
\qquad
D_5(x_q)=8.950791.
$$

The exhaustive-optimal useful anchor is

$$
a_\star=(9.095542,\;0.005862),
$$

and has rank 23 by anchor distance. Its nearest observed opposite-class point
is

$$
z_\star=(0.036330,\;-0.000996),
$$

so

$$
\lVert a_\star-z_\star\rVert_1=9.066071,
\qquad
\epsilon_\star^{(0)}
=
\epsilon_\star^{(\mathrm{final})}
=1.813214.
$$

The exhaustive counterfactual is

$$
x_\star^{\mathrm{CF}}
=(7.298417,\;0.021951),
\qquad
D^\star(x_q)=7.300799.
$$

Both points are predicted as class 1. Nevertheless,

$$
D_5(x_q)-D^\star(x_q)=1.649992,
\qquad
\frac{D_5(x_q)}{D^\star(x_q)}=1.226002.
$$

No fallback occurs.

## 9. Aggregate result

The complete experiment uses ten independent seeds. Each seed resamples the
support set and queries and changes CertCF's random anchor selection. There are
100 queries per seed.

| Quantity | Result |
|---|---:|
| Seeds | 10 |
| Queries | 1,000 |
| Top-5 misses of the certified-atlas optimum | 998 (99.8%) |
| Top-5 target validity | 100% |
| Exhaustive target validity | 100% |
| Fallback rate | 0% |
| Mean top-5 $L_1$ distance | 8.960 |
| Mean exhaustive $L_1$ distance | 7.298 |
| Mean additive gap | 1.662 |
| Smallest observed nonzero gap | 1.469 |
| Mean ratio $D_5/D^\star$ | 1.228 |
| Mean rank of the exhaustive-optimal anchor | 21.3 |

The only two cases in which top-5 recovers the exhaustive optimum have queries
with unusually large positive first coordinates. In those cases a useful
anchor enters the first five positions, at ranks 5 and 1 respectively.

### Aggregate diagnostic plots

![Aggregate radius, distance, and rank diagnostics](outputs/counterexample_diagnostics.png)

Panel A shows that Equation (1) assigns comparable initial radii to decoy and
useful anchors, but certification reduces every decoy radius by a factor of
$128$ while leaving useful radii unchanged. Panel B compares the returned
distance for every query: almost every point lies strictly above the equality
line $D_5=D^\star$. Panel C shows why fixed top-5 retrieval misses these
regions: the exhaustive-optimal anchor is usually ranked between approximately
19 and 24 by center distance.

## 10. Role of Equation (1) in limiting the gap

The construction does not produce an arbitrarily large multiplicative gap at
$\alpha=0.2$. This is a consequence of Equation (1), given the assumptions of
this experiment.

Let $a$ be a target anchor, let

$$
r=\lVert x_q-a\rVert_1,
$$

and suppose an observed opposite-class point $z$ satisfies

$$
\lVert x_q-z\rVert_1\leq\delta.
$$

By the triangle inequality and Equation (1),

$$
\epsilon(a)
\leq
\alpha\lVert a-z\rVert_1
\leq
\alpha(r+\delta).
$$

Since every certified region remains inside its initial ball,

$$
P_a\subseteq B_1(a,\epsilon(a)),
$$

which implies

$$
d(x_q,P_a)
\geq
r-\epsilon(a)
\geq
(1-\alpha)r-\alpha\delta.
$$

With no actionability constraints, a certified anchor center itself provides a
candidate at distance at most $r$. When useful and decoy anchors have nearly
equal center distance and $\delta$ is small, the largest attainable ratio is
therefore approximately

$$
\frac{1}{1-\alpha}.
$$

For $\alpha=0.2$, this value is 1.25. The observed mean ratio 1.228 is close to
this finite-range ceiling. This argument is not a universal bound for arbitrary
CertCF queries: it uses local opposite-class support, feasible anchor centers,
no query-specific actionability constraints, and nearly equal useful/decoy
anchor distances.

## 11. What the example establishes

The experiment establishes the following point:

> Ranking certified regions by their anchor-center distance does not, in
> general, rank those regions by their distance from the query. Consequently,
> a fixed nearest-anchor top-$k$ search need not recover the optimal
> counterfactual available in the certified atlas, even when it returns a valid
> counterfactual and never activates fallback.

The example also demonstrates that the relevant mismatch can be generated by
the actual interaction between Equation (1), the model boundary, and adaptive
LiRPA certification rather than by manually assigning different atlas radii.

## 12. What the example does not establish

It does not establish that 99.8% failure is representative of trained networks.
The following aspects are intentionally adversarial:

1. The model weights analytically encode the desired disconnected target set.
2. The five decoy target components are extremely small relative to the useful
   component: $0.03$ versus $5$ in $L_1$ radius.
3. Target samples lie inside the tiny decoy components, but the observed source
   distribution contains no opposite-class samples near their boundaries.
4. The useful component extends far beyond the region occupied by its observed
   target support.
5. The sparse max-of-component computational graph is not the standard dense
   MLP architecture used in the paper.
6. The problem has no categorical variables, feasibility constraints,
   actionability constraints, sparsity refinement, or robustness erosion.

A conventionally trained network would generally need negative examples near
the decoy components to learn such small target islands. Those same negative
examples would reduce the initial radii assigned by Equation (1), potentially
removing the effect. The next experiment must therefore train an ordinary ReLU
MLP and determine empirically whether a comparable mismatch survives without
constructing the decision boundary directly.

## 13. Reproduction

From the repository root, run one seed with:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/end_to_end.py \
    --seed 0
```

Regenerate the ten-seed experiment and aggregate it with:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/run_seeds.py

MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/summarize.py
```

Regenerate the explanatory figure with:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/make_clear_figure.py
```

Regenerate the aggregate diagnostic plots with:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/make_diagnostic_plots.py
```

The principal artifacts are:

- `outputs/counterexample_explained.png`;
- `outputs/counterexample_explained.pdf`;
- `outputs/counterexample_diagnostics.png`;
- `outputs/counterexample_diagnostics.pdf`;
- `outputs/aggregate/aggregate_summary.json`;
- `outputs/aggregate/seed_summary.csv`;
- `outputs/seeds/seed_<n>/summary.json`;
- `outputs/seeds/seed_<n>/anchors.csv`;
- `outputs/seeds/seed_<n>/queries.csv`;
- `outputs/seeds/seed_<n>/data.npz`.
