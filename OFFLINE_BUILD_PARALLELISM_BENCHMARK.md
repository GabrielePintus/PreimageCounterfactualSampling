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

## Interpretation for the paper

- Small tabular builds are too short for worker overhead to be informative.
- On the largest synthetic FCNN, parallel LiRPA certification provides the
  useful gain: about 1.41x, at the cost of a higher CUDA-memory peak.
- On ResNet-20, two concurrent LiRPA shards contend for the same GPU and provide
  little benefit. Parallel radius computation is responsible for most of the
  end-to-end reduction, from 339.0 s to 275.5 s in the combined configuration.
- The appendix should report these representative examples rather than imply
  that every atlas build scales uniformly with the number of workers.

