# Experiments

`network_complexity_grid.yaml` is the unified protocol for the 25-cell CertCF
ReLU MLP width/depth grid. Run it with:

```bash
python scripts/network_complexity_grid.py all \
  --config configs/experiments/network_complexity_grid.yaml
```

Artifacts use `depth_DD_width_WWW` identifiers. The runner resumes only when
the protocol fingerprint and checkpoint SHA-256 match, and it writes each
architecture result atomically before producing the validated combined files.
All classifier training and the shared accuracy gate finish before the first
CertCF benchmark begins.

`verix_certcf_mnist.yaml` defines the paired MNIST benchmark in which the
nearest canonically valid native VERIX witness determines the target class
subsequently used by CertCF. The resumable CLI is
`scripts/verix_certcf_mnist.py`.

`verix_certcf_synthetic32.yaml` is the faster paired VERIX/CertCF validation
on the existing standardized 32-dimensional dataset. It initially compares
the smallest and largest network-grid endpoints using 10 balanced queries and
resumes VERIX after every completed feature check. The common VERIX radius is
0.5 in standardized feature space; the smaller 0.05 pilot produced no native
witnesses.

`verix_certcf_synthetic32_grid.yaml` extends that protocol to the Cartesian
product of depths 1--5 and widths 16--256. It reuses the same 10 balanced
queries and enforces a cumulative 120-second VERIX budget per
architecture/query pair. Timeout outcomes are terminal, explicitly recorded,
and retained in the aggregate instead of blocking later grid cells.

`cifar_resnet_scaling.yaml` defines the scaling experiment on pretrained
CIFAR-10 ResNet20/32/56 models. It selects 10,000 balanced images from the
training split and uses all of them as atlas anchors, with 100 shared balanced
queries, $L_1$ certification/projection, and top-$k=3$. Each network is
benchmarked independently before strict aggregation.

`lirpa_refinement_ablation.yaml` compares Anchor-Ball, Anchor-PGD, and CertCF
on HELOC, Adult, and width-128 Synthetic32 networks of depths 1--5. The runner
materializes one shared full-support Eq. (1) geometry per case, uses exactly
500 random anchors per predicted class, and evaluates 500 shared test queries.
The tabular cases use top-$k=5$ and the depth slice uses top-$k=3$.

`topk_heuristic_ablation.yaml` evaluates strict nearest-anchor prefixes against
the exhaustive atlas optimum. Its additional `parallel-benchmark` stage times
each fixed $k$ independently with configurable candidate worker counts. The
process backend keeps a warm projection pool and is the appropriate backend
for the small CVXPY/CLARABEL problems; the thread backend is retained as a
diagnostic option.
