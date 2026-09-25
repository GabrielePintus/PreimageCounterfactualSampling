# Experiment Protocol Notes

This directory contains implementation-facing protocol notes for the more
expensive appendix and scalability experiments. Reviewer reproduction should
start from the root README and the commands in `scripts/README.md`; these notes
provide additional design and provenance detail.

- `APPENDIX_D_TABULAR_EXPERIMENT_PLAN.md` records the common seven-dataset
  ablation protocol.
- `OFFLINE_BUILD_PARALLELISM_BENCHMARK.md` records the offline atlas-build
  parallelism protocol.

The completed top-k projection protocol is represented by
`configs/benchmarks/final_benchmark_certcf_parallel.yaml` and the corresponding
entry in `configs/paper/paper_results.yaml`; its former internal progress log is
not part of the reviewer-facing repository.
