# CertCF Privacy Post-Processing Idea

## Motivation

Nearest-neighbor counterfactual methods can directly disclose training samples because the returned counterfactual is itself a point from the training set. CertCF can also sometimes return a training anchor, especially when nearest-anchor initialization is used. This suggests a simple privacy metric:

```text
training-point disclosure rate =
    fraction of returned counterfactuals that match a training sample
```

The goal is to reduce this rate, ideally to zero, while preserving CertCF validity and paying only a small proximity cost.

This should not be called formal differential privacy unless we later prove a DP guarantee. For now, the correct framing is:

```text
certified training-point disclosure mitigation
```

or:

```text
privacy-preserving CertCF post-processing
```

## Core Idea

CertCF returns a point inside a certified target-class polytope. If the returned point is exactly a training sample, or too close to one, we can move it slightly while keeping it inside the same certified polytope.

Let:

- `x_query` be the original factual point.
- `x_cf` be the counterfactual returned by CertCF.
- `P` be the certified target-class polytope containing `x_cf`.
- `x_train` be the nearest training sample to `x_cf`.
- `rho` be the desired privacy radius.

We want a new counterfactual `z` such that:

```text
z in P
distance(z, x_train) >= rho
z remains close to x_query
```

Because `z` stays inside `P`, the target-class validity should remain certified.

## Deterministic Formulation

A natural privacy constraint is an L-infinity separation from the leaked training point:

```text
||z - x_train||_inf >= rho
```

The direct constraint is non-convex because it means that at least one coordinate must differ by at least `rho`:

```text
exists j such that |z_j - x_train_j| >= rho
```

We can avoid sampling by enumerating the disjunction. For each feature `j` and sign `s in {-1, +1}`, solve:

```text
minimize_z       ||z - x_query||_1

subject to       z in P
                 s * (z_j - x_train_j) >= rho
```

Then choose the feasible solution with the smallest L1 distance to `x_query`.

This is deterministic and only requires up to `2 * d` small convex solves for a `d`-dimensional dataset. Since this post-processing is only needed when the returned CF is too close to a training point, the extra cost should be acceptable.

## Practical Algorithm

For each CertCF query:

1. Run CertCF normally and obtain `(x_cf, P)`.
2. Compute nearest training distance:

   ```text
   d_train = min_x in D_train ||x_cf - x||_inf
   ```

3. If `d_train > rho`, return `x_cf` unchanged.
4. Otherwise, let `x_train` be the nearest training point.
5. Enumerate all coordinate/sign constraints:

   ```text
   z_j >= x_train_j + rho
   z_j <= x_train_j - rho
   ```

6. For each feasible branch, solve the certified projection problem inside `P`.
7. Return the feasible `z` with smallest L1 distance to `x_query`.
8. Store metadata.

## Metadata To Save

The benchmark parquet should expose enough fields to analyze the tradeoff:

```text
privacy_postprocess_enabled
privacy_radius
privacy_triggered
privacy_original_train_linf_distance
privacy_new_train_linf_distance
privacy_original_l1_distance
privacy_new_l1_distance
privacy_l1_delta
privacy_branch_attempts
privacy_feasible_branches
privacy_selected_feature
privacy_selected_sign
privacy_success
```

If the post-processing fails to find a feasible point, the safest behavior is configurable:

```text
privacy_on_failure: return_original | mark_failed
```

For paper experiments, `mark_failed` is cleaner if we want the privacy constraint to be part of the algorithm's contract.

## Metrics

The main privacy metric should be:

```text
training-point disclosure rate at tolerance eps
```

where a returned CF is considered a training-point disclosure if:

```text
min_x in D_train ||x_cf - x||_inf <= eps
```

Useful companion metrics:

```text
median nearest-training L_inf distance
mean nearest-training L_inf distance
validity
normalized L1 AUC
mean L1 proximity
privacy post-processing trigger rate
privacy post-processing success rate
```

Expected qualitative behavior:

```text
NN10000: disclosure rate near 100%
CertCF NN-init 10k: possibly nonzero disclosure rate
Private CertCF: disclosure rate near 0%, validity preserved, proximity slightly worse
DiCE / Growing Spheres: likely near 0%, but without certified validity
FACE: depends on whether it returns graph nodes or interpolated points
```

## Experiment Plan

Run CertCF NN-init 10k with several privacy radii:

```text
rho in {0, 1e-8, 1e-6, 1e-4, 1e-3}
```

Report:

```text
method
rho
validity
normalized L1 AUC
training-point disclosure rate
median nearest-training L_inf distance
mean L1 distance
post-processing trigger rate
```

The key table should show the tradeoff:

```text
rho increases -> disclosure decreases, proximity worsens slightly, validity remains certified
```

## Paper Claim

A careful claim would be:

```text
CertCF supports a deterministic certified post-processing step that prevents exact training-point disclosure by moving returned counterfactuals away from training samples while remaining inside the certified target-class region.
```

Avoid claiming formal differential privacy unless a separate proof is added.

