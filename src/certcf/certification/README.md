# certification

Offline certification strategies and bound-computation utilities. Online search
consumes certified geometry and does not need to know which backend produced it.

## Contents

- `wrapping.py`: model wrappers used before handing networks to auto_LiRPA.
- `lirpa.py`: CROWN/optimized-CROWN/alpha-CROWN orchestration and `PreimageApproximation`.
- `bounds.py`: small helpers for extracting lower/upper bound values.
- `backend.py`: `CertificationBackend` protocol, `BuildRequest`, `InputDomain`,
  and `BuildResult` contracts.
- `local_lirpa.py`: strategy adapter for the existing batched LiRPA/shrinkage path.
- `premap.py`: optional, process-isolated original PREMAP adapter.
- `premap_worker.py`: standalone worker, executed by PREMAP's own Python.
- `artifacts.py`: versioned numeric JSON/NPZ artifacts; no Python pickle.

The final benchmark uses the standard backward CROWN path through `lirpa_method: backward`.

## Example

```python
from certcf.certification import PreimageApproximation
```

## Architecture and compatibility

The strategy boundary is `build(request) -> BuildResult`, not a replacement for
the online projection solver. Domain selection is a separate responsibility:
`InputDomain` contains explicit geometry and a target class. A backend may return
zero, one, or multiple regions per domain. Every region retains its source-domain
ID, target, complete geometry, and provenance.

`RegionGeometry` (in `geometry/regions.py`) represents `A @ x + b >= 0`, finite
coordinate bounds, and an optional norm ball. Box faces are part of the
certificate: they must not be discarded after PREMAP splits a domain. The
classification margin is already absorbed into `b`; online projection must not
apply it again. Retrieval representatives need not be contained in a region and
are never accepted without checking the actual certified geometry.

Existing `CertCFAtlas(model, dataset, ...).build()` calls use `LocalLiRPABackend`
and preserve the existing batching, adaptive shrinkage, checkpointing, and
anchor-inclusion policy. Existing benchmark configurations are unchanged. Direct
strategy calls accept explicit ball domains; PREMAP accepts explicit box domains.
No implicit conversion from an L1 ball to its enclosing box is performed.

`CertCFAtlas.from_regions(result, ...)` and `from_directory(path, ...)` allow
query-only use without a model, reference data, or importing auto_LiRPA. The
existing projection, OHE decoding, sparsity refinement, and parallel candidate
evaluation are reused. Generic norm-ball projection currently supports L1, L2,
and Linf; box-only PREMAP regions do not impose a ball norm. Geometry containers
support arbitrary p >= 1. Mixed-norm erosion of a ball is conservative.

## Optional PREMAP environment

This adapter targets the **original PREMAP**, not PREMAP2:

- Paper: [PREMAP, JMLR 2025](https://www.jmlr.org/papers/v26/24-1297.html).
- Code: [PreimageApproxForNNs](https://github.com/Zhang-Xiyue/PreimageApproxForNNs).
- Required revision: `0bbbdb99d5813549d7d0656a05a99073eb4a48ed` (clean tracked files).

Its bundled auto_LiRPA needs an older PyTorch API. Do not replace CertCF's
dependencies to accommodate it. From the project root, choose separate paths:

```bash
git clone https://github.com/Zhang-Xiyue/PreimageApproxForNNs.git /path/to/premap-source
git -C /path/to/premap-source checkout --detach 0bbbdb99d5813549d7d0656a05a99073eb4a48ed
uv venv --python 3.10 /path/to/premap-env
uv pip install --python /path/to/premap-env/bin/python -r configs/premap/requirements-cpu.txt
```

The CPU dependency versions above have been smoke-tested on Linux x86_64. Some
recent Linux loaders reject the old PyTorch wheel with `cannot enable executable
stack`. On the development host, backing up `torch/lib/libtorch_cpu.so` and using
`patchelf --clear-execstack` on that library **inside the external environment**
resolved it. This is a platform-specific compatibility workaround, not a change
to PREMAP source or a reason to disable system security. CUDA has not been tested.

Initial supported scope:

- Binary continuous tabular inputs and explicit finite, full-dimensional boxes.
- Float32 `nn.Sequential` graphs composed of Linear/ReLU (eval-mode Dropout and
  Identity are ignored). Pass the actual inference graph explicitly; the adapter
  does not infer arbitrary custom `forward` implementations.
- Under-approximation with **input splitting**. No ReLU splitting, multiclass
  specification, CNN, or latent encoder/decoder integration in this first version.
- Positive target logit margin; box endpoints exactly representable in float32.

The worker uses upstream volume-oriented bound optimization, split selection,
and branch queue. Adapter hooks replace benchmark-specific model/data loading
in the sampling routines and export every leaf's affine bounds **and box**.
Coverage estimates use uniform samples in each box relative to the margin-based
target set. Volumes are normalized to the source box to avoid high-dimensional
underflow. The smooth split reward uses target samples, as upstream does. Purely
affine networks use the upstream CROWN path because no relaxation slopes exist.
Thus this is an adapted PREMAP backend, not an unmodified reproduction of its
published benchmark protocol.

## Usage

With an already trained `model` in eval mode and `d` numerical input features:

```python
import numpy as np
from certcf import CertCFAtlas, BuildRequest, InputDomain, RegionGeometry
from certcf.certification.premap import PremapBackend, PremapConfig

# Explicit experimental domain: choose bounds appropriate for the trained model.
# These are not automatically inferred from CertCF's existing L1 neighborhoods.
box = RegionGeometry(np.empty((0, d)), np.empty(0),
                     -np.ones(d), np.ones(d))
request = BuildRequest(
    model=model,
    domains=(InputDomain("domain-0", target_class=1, geometry=box),),
    class_labels=(0, 1),  # in model-output order, not just the requested classes
    input_shape=(d,),
    classification_margin=1e-3,
)
backend = PremapBackend(PremapConfig(
    python="/path/to/premap-env/bin/python",
    repository="/path/to/premap-source",
    timeout_s=120, max_regions=128, coverage_samples=2000,
    coverage_threshold=0.9, seed=0,
))
atlas = CertCFAtlas(certification_backend=backend, distance_norm=1)
atlas.build(request=request)
print(atlas.build_result.status, atlas.build_result.diagnostics)
cf = atlas.find_counterfactual(x_query, target_class=1)
atlas.save_bounds("artifacts/premap-atlas")

online = CertCFAtlas.from_directory("artifacts/premap-atlas")
# Point-by-region geometry membership; reusable for nested coverage estimates.
membership = online.membership_matrix(held_out_points, label=1)
```

`membership_matrix` checks certificate geometry only, not actionability or model
labels. To estimate held-out class-conditional coverage, first select held-out
points predicted as the target class, then take the row-wise union of membership
columns. Group regions by `source_domain_id` when varying the number of input
domains: a PREMAP leaf is not an independent anchor. Internal uniform-sampling
coverage is **not** held-out data coverage and is not a formal volume certificate.

`timeout_s` caps the entire worker request including startup. `max_regions` caps
leaf regions **per input domain**; domains are processed sequentially in the
provided order. Complete iterations are checkpointed atomically. A timeout
returns only completed checkpoints (possibly no regions), never guessed bounds.
Worker errors raise with a log excerpt, with no silent fallback to LocalLiRPA.
`budget_exhausted` means a partial construction; `complete` means the sampled
stopping criterion was met for every domain, **not** exact preimage coverage.
An empty or partial atlas is not evidence that no counterfactual exists.

### Sampling and reproducibility

`coverage_samples` controls the uniform sampling used for coverage estimates and
split rewards. The old `samples` name remains an alias; conflicting values are
rejected. Each half of a candidate bisection receives `coverage_samples // 2`
samples (an odd count therefore drops one sample for pairwise estimates).
Both the initial and pairwise coverage calls reset the RNG to `seed + domain
index`, matching the original reset-per-call policy for each adapted domain.
Changing the order of domains changes these seeds.

The pinned upstream **volume optimizer is separate**: it uses 1,000 samples and
hard-codes seed 1 inside its objective. Neither `coverage_samples` nor `seed`
overrides those optimizer settings. This behavior is deliberately preserved,
not silently patched; runs with different coverage seeds are not independent
replications of the optimizer's random draws. Plain CROWN and affine models do
not use this volume optimizer.

`BuildResult.diagnostics["sampling"]` records the full sampling contract;
per-domain diagnostics record the actual coverage seed and bound method.
These settings survive atlas serialization and enter the request fingerprint.
Protocol `input-coverage-v2` also fixes the missing initial-coverage RNG reset
in the first adapter, uses the upstream argmax tie rule on margin-shifted logits,
and removes the extra per-half samples previously used for counts below four.
Historical pilot outputs are not rewritten: recompute into a new output
directory before comparing results across sampling protocols.

For a matched-domain experiment, `PremapConfig(optimize_bounds=False,
max_regions=1)` runs plain backward CROWN in the same external environment,
without PREMAP's slope optimization or domain splitting. This is a control,
not the PREMAP method; it can return empty regions. The default remains
`optimize_bounds=True`.

Version-2 artifacts store region geometry, provenance, status, query settings,
and checksums. Read them only from trusted sources: checksums detect corruption,
not forged certificates. Old version-1 atlases still load; very old manifests
without a stored classification margin need that original margin supplied to
`from_directory(..., classification_margin=...)`. Saved norm/margin metadata
defines the certificate and takes precedence over conflicting v1 overrides.
Atomic v2 saves keep older immutable payloads, which can be cleaned separately
when no reader uses the older manifest.

Both artifact formats now save solver order, solver options and accepted
statuses, iteration limits, and query/candidate parallelism. `from_directory`
restores these settings, with explicit query overrides taking precedence. Old
artifacts missing these fields still use the normal defaults: settings never
recorded cannot be recovered retroactively. `load_bounds` instead replaces
geometry and certificate norm/margin while retaining the existing instance's
query preferences; use `from_directory` for a full query-configuration restore.

## Tests

Normal geometry/strategy/regression tests need no external PREMAP installation:

```bash
.venv/bin/pytest tests/counterfactuals/test_certification_backends.py -q
```

Enable real upstream smoke and differential tests explicitly:

```bash
CERTCF_PREMAP_PYTHON=/path/to/premap-env/bin/python \
CERTCF_PREMAP_REPOSITORY=/path/to/premap-source \
.venv/bin/pytest tests/counterfactuals/test_certification_backends.py \
  tests/counterfactuals/test_premap_upstream_equivalence.py -q
```

The differential tests run the original sampling functions and branch-and-bound
implementation independently of the worker on a two-hidden-ReLU-layer fixture.
Only model/data loaders and singleton-axis layout are adapted in the oracle;
the original leaf budget is mapped to the same maximum leaf count. Tests compare
coverage, affine coefficients, leaf boxes/split outcomes, and stopping status
for both target classes, with and without splitting. They also check bound and
target-margin invariants on a 2D grid and exercise the optimizer's actual sample
count and seed behavior. This is regression evidence for these fixtures, not a
proof of general equivalence, benchmark quality, or PREMAP2 compatibility.
