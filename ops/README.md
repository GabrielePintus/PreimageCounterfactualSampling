# Operational Docs

This folder contains Markdown documents that are operationally useful for
contributors and coding agents working in this repository.

## Files

- `00-repository-map.md`: architecture map, ownership boundaries, API entrypoints
- `01-training-playbook.md`: LightningCLI workflows, configs, and training validations
- `02-certified-atlas-playbook.md`: CertCF/certification/query pipeline and safe edit guardrails
- `03-subprojects-and-dependencies.md`: boundaries for embedded dependencies
- `04-agent-operating-guidelines.md`: practical editing and validation rules
- `05-repository-tidy-checklist.md`: repository cleanup and organization checklist
- `06-history.md`: lightweight operational history of major repository decisions
- `07-absolute-certification-report.md`: benchmark/result semantics for certification reporting
- `08-method-validation-playbook.md`: benchmarking, metrics, and plot guidelines for method evaluation

## Why This Exists

The repository contains multiple packages, notebooks, and artifact-heavy folders.
These docs reduce context-loading time and help both humans and agents route
directly to the right subsystem.

## Maintenance Rule

When behavior changes, update the nearest file here in the same change set.
