# Tests

Automated tests for the modular counterfactual framework.

## Layout

- `counterfactuals/`: unit and smoke tests for methods, metrics, registry, and experiment runner.

## Run

```bash
pytest tests/counterfactuals -q
```

## Notes

- Tests focus on API contracts and lightweight integration behavior.
- Core CertCF/preimage workflows are primarily validated with notebooks and targeted scripts.
