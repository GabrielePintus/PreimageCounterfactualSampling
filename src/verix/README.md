# VERIX

This package is an isolated, paper-faithful implementation of Algorithm 1 in
Wu, Wu, and Barrett, *VERIX: Towards Verified Explainability of Deep Neural
Networks* (NeurIPS 2023).

Sources:

- Paper: <https://arxiv.org/abs/2212.01051>
- Official implementation: <https://github.com/NeuralNetworkVerification/VeriX>

## Why this is a separate package

`verix` implements the original explanation algorithm. It is intentionally not
placed in `counterfactuals.methods` yet, because a benchmark adapter will need
additional policy that is not part of VERIX: target-class selection, choosing
one witness, tabular feasibility, actionability, and metric reporting.

The separation lets us distinguish results of the original method from the
selection and pairing policy in `experiments.verix_certcf_mnist`.

## Preserved algorithmic behavior

- Features are visited exactly once in a specified traversal.
- At step `i`, the verifier checks perturbations of `B union {i}`, while every
  other feature remains fixed.
- A proven invariant feature is added to the irrelevant set `B`.
- A violated check adds the feature to explanation `A` and stores the concrete
  verifier counterexample in `C`.
- An unknown or timed-out check is conservatively added to `A`, but has no
  counterfactual witness.
- The sensitivity traversal ranks features from least to most sensitive using
  the original-class score difference.
- Explanation optimality is local/subset-minimal, and is reported only when a
  complete checker returns no unknown results.

The structured result and verifier protocol are engineering changes; they do
not alter Algorithm 1.

## Marabou backend

The optional `MarabouClassificationChecker` follows the official source:

- ONNX Runtime obtains the original prediction.
- Marabou reads pre-softmax logits when the ONNX graph ends in Softmax.
- Free coordinates receive clipped `[x_i - epsilon, x_i + epsilon]` bounds.
- Other coordinates are fixed to the query.
- Each competing logit is checked separately.
- SAT returns a concrete counterfactual; UNSAT proves invariance; timeout is
  treated as unknown.
- Worker count, timeout, and classification margin match the reference
  repository: 16 workers, 300 seconds, and `1e-6`.
- The native Marabou solver is the default. Set `solve_with_milp=True` only
  when Marabou has been compiled with Gurobi support and a license is active.

Although the paper defines norms `p in {1, 2, infinity}`, the released Marabou
implementation realizes only the L-infinity case with coordinate-wise bounds.
The generic core carries the norm in every check request so another faithful
backend can implement the other cases; the Marabou backend rejects them
explicitly.

Install optional dependencies on a Marabou-supported Python version:

```bash
pip install -e '.[verix]'
```

Gurobi is optional in current Marabou releases. The original VERIX authors
compiled Marabou with Gurobi enabled, but this changes the solver backend and
performance rather than Algorithm 1.

## Not implemented here

- No optimization of counterfactual proximity.
- No target-directed multiclass search.
- No anchor-centered regions.
- No automatic one-hot decoding or actionability constraints.
- No registration as a generic counterfactual method.

The dedicated MNIST experiment keeps that additional policy outside the VERIX
algorithm: it validates every native witness on the canonical classifier,
selects the closest valid witness in $L_1$, and uses its predicted class as the
paired CertCF target.
