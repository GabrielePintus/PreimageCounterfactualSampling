"""Separate numerical/positive-control diagnostics, not a coverage benchmark."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from certcf import BuildRequest, CertCFAtlas, InputDomain, RegionGeometry
from certcf.certification.premap import PremapBackend, PremapConfig
from experiments.atlas_premap import explicit_graph, region_feasibility
from experiments.network_complexity import sha256_file
from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner


def affine_box_maxima(geometry):
    """Exact per-row maxima of A x + b over a box (no model inference)."""
    return (np.maximum(geometry.A, 0) @ geometry.upper
            + np.minimum(geometry.A, 0) @ geometry.lower + geometry.b)


def run_diagnostics(config_path, python, repository):
    config = yaml.safe_load(Path(config_path).read_text())
    root = Path(config["output_dir"])
    torch.set_num_threads(1)
    source = TopKHeuristicAblationRunner.from_yaml(config["source_config"])
    if not source._valid_build():
        raise RuntimeError("Source artifacts missing or stale")
    original_hashes = source._atlas_file_hashes()
    with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
        test = data["x_test"].copy()
    with np.load(source.paths.atlas_support, allow_pickle=False) as data:
        anchors, labels = data["x_support"].copy(), data["y_support"].copy()
    model = source.source_runner._load_model(source.depth, source.width, "cpu")
    graph = explicit_graph(model, torch.tensor(anchors[:128], dtype=torch.float32))
    with torch.no_grad():
        predictions = graph(torch.tensor(test, dtype=torch.float32)).argmax(1).numpy()
    rows, input_hashes = [], {}
    for budget in config["premap"]["region_budgets"]:
        for label in (0, 1):
            path = root / f"regions_{budget:03d}" / f"class_{label}" / "atlas"
            input_hashes[str(path / "manifest.json")] = sha256_file(path / "manifest.json")
            atlas = CertCFAtlas.from_directory(path)
            maxima = np.concatenate([affine_box_maxima(r.geometry) for r in atlas.regions])
            # One margin constraint per leaf in this binary pilot.
            assert all(len(r.geometry.b) == 1 for r in atlas.regions)
            sigmoid_max = float(torch.sigmoid(torch.tensor(maxima.max(), dtype=torch.float32)))
            rows.append({"region_budget": budget, "class_label": label,
                         "smallest_leaf_max_margin": float(maxima.min()),
                         "largest_leaf_max_margin": float(maxima.max()),
                         "float32_sigmoid_of_largest_margin": sigmoid_max})
    controls = []
    for label in (0, 1):
        # Predetermined selection from existing anchors; no test-set tuning.
        selected = int(np.flatnonzero(labels == label)[0])
        center = anchors[selected]
        # This small box tests operation on the 32D model, not generalization.
        radius = np.float32(.05)
        lo, hi = (center - radius).astype(np.float32), (center + radius).astype(np.float32)
        box = RegionGeometry(np.empty((0, len(center))), [], lo, hi)
        backend = PremapBackend(PremapConfig(python, repository, timeout_s=30, max_regions=1,
                                            samples=1000, seed=42, coverage_threshold=.9))
        request = BuildRequest(graph, (InputDomain(f"control-anchor-{selected}", label, box),),
                               (0, 1), (len(center),), source.config["certcf"]["classification_margin"])
        started = time.perf_counter()
        built = backend.build(request)
        seconds = time.perf_counter() - started
        atlas = CertCFAtlas.from_regions(built, distance_norm=1)
        folder = root / "positive_control" / f"class_{label}"
        atlas.save_bounds(folder)
        feasible = [region_feasibility(r, graph, request.classification_margin) for r in built.regions]
        membership = atlas.membership_matrix(test, label)
        controls.append({"class_label": label, "anchor_index": selected, "halfwidth": float(radius),
                         "status": built.status, "build_seconds": seconds,
                         "n_regions": len(built.regions), "n_feasible_regions": sum(r["feasible"] for r in feasible),
                         "anchor_in_certified_union": bool(atlas.membership_matrix(center[None, :], label).any()),
                         "n_heldout_covered": int(membership[predictions == label].any(1).sum()),
                         "region_witnesses": feasible})
    result = {"source_run": source._run_fingerprint(), "global_artifact_hashes": input_hashes,
              "diagnostic_code_sha256": sha256_file(Path(__file__)),
              "global_margin_diagnostics": rows, "local_positive_controls": controls,
              "note": "Local controls use the first stored anchor per class and halfwidth 0.05, solely to check functionality on the same 32D classifier. They are not a coverage-improvement experiment."}
    (root / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    if source._atlas_file_hashes() != original_hashes:
        raise AssertionError("Source atlas was modified")
    return result
