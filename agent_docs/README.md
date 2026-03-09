# Agent Docs Index

This folder contains task-specific repository context for coding agents.

## Files

- `00-repository-map.md`: architecture map, ownership boundaries, API entrypoints
- `01-training-playbook.md`: LightningCLI workflows, configs, and training validations
- `02-certified-atlas-playbook.md`: CPP/certification/query pipeline and safe edit guardrails
- `03-subprojects-and-dependencies.md`: boundaries for embedded dependencies
- `04-agent-operating-guidelines.md`: practical editing and validation rules

## Why This Exists

The repository contains multiple packages, notebooks, and artifact-heavy folders.
These docs reduce context-loading time and help agents route directly to the right subsystem.

## Maintenance Rule

When behavior changes, update the nearest file here in the same change set.
