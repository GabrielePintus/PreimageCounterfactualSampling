# Boundary Manifold Sampling Ideas

Working note for future CertCF research directions around explicit use of the
decision boundary.

## Core Idea

Treat the binary decision boundary as the implicit manifold

\[
g(x) = f_1(x) - f_0(x) = 0
\]

and use this manifold directly to generate better atlas support.

The main motivation is:
- larger isotropic certified regions do not necessarily improve proximity
- anchor placement near useful boundary regions may matter more than simply
  increasing `alpha`
- boundary-adjacent support is especially relevant if we want to emphasize
  certified robustness / post-erosion coverage

## Boundary Walk

For a boundary point \(x\) with \(\nabla g(x) \neq 0\):

- normal direction: \(\nabla g(x)\)
- tangent space:
  \[
  T_x \mathcal{M} = \{v : \nabla g(x)^\top v = 0\}
  \]

In 2D, we implemented a predictor-corrector walk in notebook `6.16`:
- find a seed point on the boundary by bisection
- predictor: step along the tangent
- corrector: project back to `g(x)=0` with a Newton-style update

This is useful for:
- visualizing one connected boundary component
- understanding where the boundary is flat vs curved
- building intuition before moving to higher-dimensional methods

## High-Dimensional Generalization

In higher dimensions, the boundary is a \((d-1)\)-dimensional manifold rather
than a curve, so there is no unique walk direction.

A formal tangent-flow formulation is:

\[
P_T(x) = I - \frac{\nabla g(x)\nabla g(x)^\top}{\|\nabla g(x)\|^2},
\qquad
\dot x = P_T(x) u(x)
\]

where \(u(x)\) is any ambient direction field.

This gives:
- deterministic boundary walking / continuation
- projected Langevin on the boundary if \(u(x)\) is stochastic
- constrained HMC / manifold sampling as a more advanced variant

## Why This Matters For CertCF

Boundary information could help CertCF in several non-exclusive ways:

1. Boundary-seeded anchors
- sample or trace points on the decision boundary
- move slightly into the target side
- certify local regions around those points

2. Better support density near critical frontier regions
- add more anchor points where the boundary is highly curved
- keep fewer anchors where the boundary is flat

3. Better diagnostics
- compare certified unions / anchor placement to actual boundary structure
- identify holes in boundary-adjacent coverage

4. Input to anisotropic certification ideas
- the boundary normal/tangent gives directional geometric information
- this may be useful for future non-isotropic trust-region design

## Practical Research Direction

If we revisit this idea, the most realistic next step is **not** full
constrained HMC.

The best first candidate is:
- projected Langevin or tangent-projected stochastic exploration on the boundary

Reason:
- simpler than HMC
- scalable to higher dimensions
- can generate diverse boundary samples offline
- naturally fits the CertCF “expensive offline, fast online” philosophy

## Important Caveat

Boundary points alone are not automatically good anchors.

Risks:
- they may be too thin / fragile
- they may be off-manifold
- certified regions around them may be very small

So any future use of this idea should evaluate:
- validity
- proximity
- robustness erosion survivability
- plausibility / manifoldness

## Suggested Future Experiment

Minimal experimental path:

1. Use the boundary-walk notebook (`6.16`) to visualize and validate the idea.
2. Implement a simple high-dimensional projected-Langevin boundary sampler.
3. Generate candidate target-side boundary-adjacent anchor points.
4. Compare against standard CertCF anchors on:
   - validity
   - proximity
   - runtime
   - robust success after erosion
