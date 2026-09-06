# CertCF offline-build parallelism benchmark

## Purpose

This benchmark isolates the two independent forms of parallelism used during
certified-atlas construction:

1. parallel computation of the initial radii from Equation (1), using eight CPU
   workers;
2. parallel LiRPA certification of class shards, using two workers and two CUDA
   streams.

The experiment is intended to support a short implementation note and a
representative serial-versus-parallel comparison in the paper appendix. It does
not replace the original scalability experiments.

## Protocol

- Implementation commit measured: `44c3cdbcad15d6d31651dae6df3ce432a8810626`.
- Hardware: Intel Core i7-14700K, 31 GiB RAM, NVIDIA GeForce RTX 4070 SUPER
  with 12 GiB VRAM.
- Every case/variant is run in a fresh Python process.
- Serialization and comparison with the reference atlas are outside the timed
  region.
- `serial`: one radius worker and one LiRPA worker.
- `epsilon8`: eight radius workers and one LiRPA worker.
- `lirpa2`: one radius worker and two LiRPA workers.
- `combined`: eight radius workers and two LiRPA workers.
- The three cases use the same data, checkpoints, selected anchors, and CertCF
  hyperparameters across all four variants.

The cases are:

- HELOC: binary tabular classifier, 1,000 atlas anchors;
- synthetic-32 FCNN: depth 5, width 256, 1,000 atlas anchors selected from
  10,000 training points;
- CIFAR-10 ResNet-20: ten classes and 10,000 atlas anchors selected from the
  training set.

The controlled CLI and configuration are:

- `scripts/offline_build_parallelism.py`;
- `configs/experiments/offline_build_parallelism.yaml`.

Raw metadata are stored under `results/offline_build_parallelism/`, and the
aggregated table is
`results/offline_build_parallelism/offline_build_summary.parquet`.

Two follow-up experiments were run with the CNN batching implementation at
commits `6544ea3` and `4831edc`:

- a three-repetition sweep of the number of CPU workers used by Equation (1);
- sound batching of CNN anchors whose initial radii differ by at most 5%.

The radius-sweep measurements are stored in
`results/offline_build_parallelism/epsilon_parallelism_sweep.parquet`. The
CNN-batch build metadata and its 100-query probe are stored under
`results/offline_build_parallelism/cifar_resnet20/cnn_batch64/`.

## Results

The following are single-run wall-clock measurements. `RSS delta` is the peak
increase over the process RSS at entry to atlas construction. `CUDA allocated`
is the peak allocated CUDA memory during the same region.

| Case | Variant | Build (s) | Radius (s) | LiRPA (s) | Speedup | RSS delta (MiB) | CUDA allocated (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| HELOC | serial | 0.856 | 0.022 | 0.833 | 1.00x | 543.9 | 12.4 |
| HELOC | epsilon8 | 0.872 | 0.009 | 0.862 | 0.98x | 562.5 | 12.4 |
| HELOC | lirpa2 | 0.923 | 0.022 | 0.900 | 0.93x | 528.2 | 23.5 |
| HELOC | combined | 0.812 | 0.010 | 0.801 | 1.05x | 529.8 | 22.8 |
| FCNN depth 5, width 256 | serial | 4.213 | 0.004 | 4.207 | 1.00x | 415.6 | 205.2 |
| FCNN depth 5, width 256 | epsilon8 | 4.330 | 0.003 | 4.324 | 0.97x | 417.7 | 205.2 |
| FCNN depth 5, width 256 | lirpa2 | 2.992 | 0.004 | 2.986 | **1.41x** | 408.6 | 298.6 |
| FCNN depth 5, width 256 | combined | 3.044 | 0.003 | 3.038 | 1.38x | 410.0 | 321.8 |
| CIFAR-10 ResNet-20 | serial | 339.028 | 76.150 | 262.853 | 1.00x | 2,851.8 | 443.1 |
| CIFAR-10 ResNet-20 | epsilon8 | 284.689 | 27.297 | 257.373 | 1.19x | 2,969.4 | 443.1 |
| CIFAR-10 ResNet-20 | lirpa2 | 333.727 | 77.735 | 255.972 | 1.02x | 3,099.6 | 461.6 |
| CIFAR-10 ResNet-20 | combined | 275.468 | 27.197 | 248.251 | **1.23x** | 3,176.4 | 461.6 |

### Equation (1) worker sweep

Each configuration computed the initial radii of the same 10,000 CIFAR-10
anchors three times. Worker order was randomized within each repetition. Every
run reproduced the serial reference radii exactly, with zero maximum absolute
difference.

| CPU workers | Mean (s) | Std. (s) | Speedup |
|---:|---:|---:|---:|
| 1 | 82.152 | 0.472 | 1.00x |
| 2 | 46.030 | 0.087 | 1.78x |
| 4 | 32.662 | 0.107 | 2.51x |
| 8 | 26.923 | 0.533 | 3.05x |
| 12 | 26.760 | 0.284 | **3.07x** |
| 16 | 26.773 | 0.225 | 3.07x |

The computation scales usefully up to approximately eight workers and then
plateaus. The operational default remains eight workers because 12 and 16 do
not provide a meaningful additional reduction.

### Sound CNN radius batching

The legacy CNN path certifies one anchor at a time because auto-LiRPA accepts a
single scalar radius for convolutional batches. The new opt-in path sorts the
anchors of each target class by radius and groups at most 64 anchors whose
largest radius is no more than 5% above the smallest. LiRPA is evaluated once
at the largest radius in each bucket. This is sound for every original ball
because the larger ball contains all of them. Each returned polytope is still
intersected with its anchor's original ball. If the conservative bucket bound
does not certify an anchor center, that anchor is recomputed at its exact radius
and then follows the existing adaptive-shrink procedure.

On CIFAR-10 ResNet-20, 10,000 anchors were grouped into 282 buckets, or 35.5
anchors per LiRPA call on average. No exact-radius fallback or adaptive shrink
was needed.

| Variant | Build (s) | Radius (s) | LiRPA (s) | Build speedup | LiRPA speedup | CUDA allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Serial reference | 339.028 | 76.150 | 262.853 | 1.00x | 1.00x | 443.1 |
| Radius workers only | 284.689 | 27.297 | 257.373 | 1.19x | 1.02x | 443.1 |
| Radius workers + CNN batches | **43.289** | 26.796 | **16.474** | **7.83x** | **15.96x** | 3,768.3 |

The batching optimization therefore exchanges GPU memory for a substantial
reduction in offline latency. It is disabled by default and controlled
independently from class-level LiRPA parallelism.

The batched atlas was also evaluated on the same 100 ResNet-20 queries used by
the original scalability benchmark. It returned a valid target-class
counterfactual for all 100 queries and selected the same anchor as the serial
atlas in every case. Mean L1 distance was 356.2740 for both atlases; the maximum
absolute per-query distance change was `6.10e-5`. The mean post-hoc certified L1
radius changed from `4.2251e-5` to `4.2195e-5` (-0.13%), with a median
batch-to-serial ratio of 0.998. Query runtime is not compared here because the
historical query artifacts used eight projection processes whereas this
correctness probe intentionally used one.

### Final optimized CIFAR-10 architecture sweep

The selected CNN configuration uses eight CPU workers for the initial radii,
radius buckets containing at most 32 anchors, and a maximum relative radius
inflation of 5%. Three complete builds were measured for every architecture.

| Network | Build (s) | Initial radii (s) | LiRPA (s) | Regions | Fallbacks | Adaptive shrinks |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-20 | 41.795 +/- 0.500 | 26.545 | 15.231 | 10,000 | 0 | 0 |
| ResNet-32 | 49.590 +/- 0.162 | 25.944 | 23.617 | 10,000 | 0 | 0 |
| ResNet-56 | 65.606 +/- 0.290 | 25.971 | 39.586 | 10,000 | 0 | 0 |

Times are the mean and, for complete build time, sample standard deviation over
three repetitions. Every run preserved exactly the reference atlas's anchor
set, initial and final radii, shrink decisions, and center-certification
decisions. Raw measurements are stored in
`results/offline_build_parallelism/lirpa_batch_size_sweep.parquet`.

## Correctness checks

For HELOC and the synthetic FCNN, every saved numeric atlas array is exactly
equal across the serial and parallel variants.

For ResNet-20, all variants select exactly the same anchors and produce exactly
the same initial/final radii, shrink counts, binary-search counts, and
center-certification decisions. The affine LiRPA coefficients vary by at most
`4.33e-3` across independent GPU executions. This is not caused by the parallel
path: comparing the old canonical serial atlas with the fresh serial atlas gives
the same scale of variation (`4.32e-3` maximum), while all anchor/radius and
certification decisions remain exactly equal. The CNN result should therefore
be described as structurally identical with ordinary CUDA floating-point
variation, not as bitwise-identical affine bounds.

The CNN radius-batched atlas deliberately does not reproduce the serial affine
coefficients: its affine relaxation is computed over a radius up to 5% larger
than the individual anchor radius and is therefore potentially more
conservative. Soundness follows from domain containment, and the exact-radius
fallback protects center certification. In the measured run, region count,
anchors, initial/final radii, shrink decisions, and center-certification
decisions all matched the serial atlas exactly. The full automated suite also
passed (360 tests passed, one skipped), including a convolutional-network test
that samples points from every original ball and checks the returned affine
lower bounds against the true network margin.

## Interpretation for the paper

- Small tabular builds are too short for worker overhead to be informative.
- On the largest synthetic FCNN, parallel LiRPA certification provides the
  useful gain: about 1.41x, at the cost of a higher CUDA-memory peak.
- On ResNet-20, two concurrent LiRPA shards contend for the same GPU and provide
  little benefit. Parallel radius computation is responsible for most of the
  end-to-end reduction, from 339.0 s to 275.5 s in the combined configuration.
- Radius-aware CNN batching is much more effective on the same ResNet-20:
  LiRPA falls from 262.9 s to 16.5 s and the complete build from 339.0 s to
  43.3 s, at the cost of a roughly 3.8 GiB CUDA-allocation peak.
- The appendix should report these representative examples rather than imply
  that every atlas build scales uniformly with the number of workers.
