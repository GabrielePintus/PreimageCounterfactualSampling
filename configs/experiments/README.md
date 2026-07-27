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
