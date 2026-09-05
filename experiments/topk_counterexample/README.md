# End-to-end counterexample for nearest-anchor top-k

This directory contains a persistent, reproducible stress test for CertCF's
nearest-anchor top-$k$ heuristic. It replaces the temporary copy that was lost
after a reboot.

For the complete mathematical construction, assumptions, numerical trace, and
interpretation, see [`COUNTEREXAMPLE.md`](COUNTEREXAMPLE.md).

## Scope

The experiment does **not** manually assign atlas radii, LiRPA halfspaces, or
projection costs. It uses the production implementation for:

- random class-wise anchor sampling;
- the Equation (1) epsilon strategy, evaluated on the full support set;
- backward auto-LiRPA bounds;
- adaptive epsilon shrinking;
- certified-atlas construction;
- nearest-anchor top-5 and exhaustive sorted querying.

The controlled part is the binary data geometry and its classifier. The model
is an explicit frozen ReLU network rather than a trained classifier, allowing
its complete decision boundary to be known exactly.

## Data and model

Source-class support points and held-out queries are independently sampled
from the same two-dimensional Gaussian around the origin. The target class has
six components:

- five small decoy $L_1$ balls, centered at $(-9,0)$, $(0,9)$, $(0,-9)$,
  $(-4.5,4.5)$, and $(-4.5,-4.5)$, each with radius $0.03$;
- one useful $L_1$ ball centered at $(9.1,0)$ with radius $5$.

The support set is balanced with 120 points per class. CertCF randomly selects
24 anchors per predicted class. The classifier computes the union of these six
target balls using only `Linear` and `ReLU` operators. All support labels are
verified before atlas construction, and every query begins in the source
class.

## Failure mechanism

The decoy anchor centers are slightly closer to the queries and dominate the
top-5 ranking. Equation (1) initially gives similar radii to decoy and useful
anchors. LiRPA and adaptive shrinking then reduce the decoy radii seven times,
from approximately $1.79$ to $0.014$, while preserving the useful radius at
approximately $1.81$.

The top-5 path finds a valid candidate in a decoy region and therefore does not
activate fallback. Exhaustive search also examines the more distant useful
anchor, whose certified region reaches substantially closer to the query.

The original ten-seed experiment produced 998 misses of the exhaustive optimum
over 1,000 queries, with 100% target validity for both methods, a mean top-5
distance of 8.960, an exhaustive distance of 7.298, and a mean ratio of 1.228.
The commands below regenerate these values rather than treating them as stored
ground truth.

## Commands

Run one seed:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/end_to_end.py \
    --seed 0
```

Run the ten-seed experiment and aggregate it:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/run_seeds.py

.venv/bin/python experiments/topk_counterexample/summarize.py
```

Generate the explanatory PNG and vector PDF:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/make_clear_figure.py
```

Generate the aggregate radius, query-distance, and anchor-rank plots:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/topk_counterexample/make_diagnostic_plots.py
```

All generated artifacts are placed under `outputs/`. The explanatory figure
shows the actual model boundary: the large blue diamond is the useful target
component, while a true-scale inset makes one of the five very small decoy
boundaries visible.

## Interpretation

With a nearby opposite-class support point, Equation (1) constrains the
relative damage possible in this unconstrained construction. For a target
anchor at query distance $r$ and an opposite support point within distance
$\delta$ of the query,

$$
\epsilon_i \leq \alpha(r+\delta),
$$

and every certified region $P_i$ remains inside its initial ball, so

$$
d(x_q,P_i) \geq (1-\alpha)r-\alpha\delta.
$$

When useful and decoy anchors have nearly equal distance and $\delta$ is small,
the attainable ratio is approximately bounded by $1/(1-\alpha)$. For the paper
setting $\alpha=0.2$, this is $1.25$; the observed mean ratio of $1.228$ is near
that ceiling. This is not a universal top-$k$ guarantee: it depends on local
support coverage and the absence of query-specific actionability constraints.

## Remaining realism step

A separate follow-up should train a conventional binary ReLU MLP on the same
distribution, including negative samples around the target components, and
test whether optimization preserves enough certification asymmetry to produce
the same behavior. That experiment should remain distinct from this controlled
case.
