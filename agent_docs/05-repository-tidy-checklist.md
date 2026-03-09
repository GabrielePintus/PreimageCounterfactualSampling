# Repository Tidy Checklist (Phase 2)

Date: 2026-03-09
Scope: non-destructive tidy pass and commit preparation.

This checklist is designed so each item can be approved/rejected before execution.
Defaults are pre-filled from current repository context.

## 1) Guardrails

- [x] Do not rewrite history.
- [x] Do not use destructive reset commands.
- [x] Keep research/code changes unless explicitly marked as noise.
- [x] Keep generated artifact paths ignored going forward.

## 2) Current Working Tree Snapshot

Tracked modified:
- `.gitignore`
- `data/Trained Models/conv_ae.pth`
- `notebooks/5.1 - Adult counterfactual sampling FACE.ipynb`
- `src/counterfactuals/experiments/runner.py`
- `src/counterfactuals/methods/README.md`
- `src/counterfactuals/methods/__init__.py`
- `src/counterfactuals/methods/dice.py`
- `src/counterfactuals/metrics/proximity.py`
- `tests/counterfactuals/test_methods.py`

Tracked deleted:
- `vicreg-loss`

Untracked:
- `config.yaml`
- `notebooks/5.0 - Adult counterfactual sampling CertifiedAtlas.ipynb`
- `notebooks/5.2 - Adult counterfactual sampling DiCE.ipynb`
- `notebooks/5.3 - Adult counterfactual sampling 1-NN.ipynb`
- `papers/Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations.pdf`
- `src/counterfactuals/methods/face.py`
- `src/counterfactuals/methods/nearest_neighbor.py`

## 3) Defaults (Can Be Revised)

### Keep as intentional work
- [x] `src/counterfactuals/experiments/runner.py`
- [x] `src/counterfactuals/methods/__init__.py`
- [x] `src/counterfactuals/methods/dice.py`
- [x] `src/counterfactuals/methods/face.py` (new file)
- [x] `src/counterfactuals/methods/nearest_neighbor.py` (new file)
- [x] `tests/counterfactuals/test_methods.py`
- [x] `notebooks/5.0 - Adult counterfactual sampling CertifiedAtlas.ipynb`
- [x] `notebooks/5.2 - Adult counterfactual sampling DiCE.ipynb`
- [x] `notebooks/5.3 - Adult counterfactual sampling 1-NN.ipynb`

### Keep but review before commit
- [ ] `notebooks/5.1 - Adult counterfactual sampling FACE.ipynb` (confirm only intended edits)
- [ ] `src/counterfactuals/methods/README.md` (confirm consistency with new methods)
- [ ] `src/counterfactuals/metrics/proximity.py` (confirm if this change was intentional)

### Needs explicit decision
- [ ] `data/Trained Models/conv_ae.pth`: keep tracked update or revert local change.
- [ ] `vicreg-loss`: restore or keep deletion.
- [ ] `config.yaml`: track or add to ignore.
- [ ] `papers/Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations.pdf`: track or move out of repo.

## 4) Ignore Policy (Already Applied)

Current additions in `.gitignore`:
- `wandb/`
- `checkpoints/`
- `tmp.ipynb`
- `lightning_logs/`
- `outputs/`

Review:
- [x] Keep these rules.
- [ ] Optional: add `config.yaml` if it is local-only.

## 5) Execution Sequence (When Approved)

### A. Verify intent-sensitive files
1. Inspect diffs:
   - `git diff -- src/counterfactuals/methods/README.md`
   - `git diff -- src/counterfactuals/metrics/proximity.py`
   - `git diff -- notebooks/5.1 - Adult counterfactual sampling FACE.ipynb`
2. Decide keep/revert per file.

### B. Resolve ambiguous assets
1. Binary model:
   - keep: leave as is
   - revert: `git restore -- "data/Trained Models/conv_ae.pth"`
2. Deleted path:
   - restore: `git restore -- vicreg-loss`
   - keep deleted: leave as is
3. Local config:
   - track: leave untracked and add later
   - ignore: append `config.yaml` to `.gitignore`
4. Paper PDF:
   - track: leave untracked and add later
   - ignore/archive: move externally or add ignore rule

### C. Validation gates
1. `python -m pytest tests/counterfactuals -q`
2. Optional import smoke test for methods package.

### D. Commit grouping (recommended)
1. Commit 1: `repo hygiene`
   - `.gitignore` and any explicit cleanup-only decisions.
2. Commit 2: `counterfactual methods`
   - `nearest_neighbor`, method registry wiring, tests.
3. Commit 3: `adult notebooks`
   - `5.0`, `5.2`, `5.3`, and any approved notebook updates.

## 6) Quick Approval Matrix

Mark each line with `approve` / `revise`:

- [ ] keep `.gitignore` cleanup additions
- [ ] keep new `1-NN` method files and registry wiring
- [ ] keep notebook additions (`5.0`, `5.2`, `5.3`)
- [ ] keep or revert `data/Trained Models/conv_ae.pth`
- [ ] keep or restore `vicreg-loss`
- [ ] track or ignore `config.yaml`
- [ ] track or exclude the new PDF in `papers/`

Once approved, execute exactly this checklist in order and produce a clean final status report.
